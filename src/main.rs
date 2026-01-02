mod bff;
mod checkpoint;
#[cfg(feature = "cuda")]
mod cuda;
mod energy;
mod fitness;
mod gpu;
mod islands;
mod metrics;
mod simulation;

use bff::SINGLE_TAPE_SIZE;
use serde::{Deserialize, Serialize};
use simulation::{Simulation, SimulationParams, Topology};
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};

/// Simulation configuration (can be loaded from YAML)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Grid dimensions
    pub grid: GridConfig,
    /// Simulation parameters
    pub simulation: SimConfig,
    /// Output settings
    pub output: OutputConfig,
    /// Energy system settings
    pub energy: EnergySettings,
    /// Checkpoint settings
    pub checkpoint: CheckpointConfig,
    /// Metrics settings (compression ratio tracking for phase transitions)
    #[serde(default)]
    pub metrics: MetricsSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GridConfig {
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SimConfig {
    pub seed: u64,
    /// Mutation rate as "1 in N" (e.g., 4096 means 1/4096 chance)
    pub mutation_rate: usize,
    pub steps_per_run: usize,
    pub max_epochs: usize,
    pub neighbor_range: usize,
    /// Auto-terminate if all programs are dead for N epochs (0 = disabled)
    pub auto_terminate_dead_epochs: usize,
    /// Run N simulations in parallel on GPU (1 = single sim)
    pub parallel_sims: usize,
    /// Layout of parallel sims as [columns, rows] for mega-simulation
    /// e.g., [4, 4] = 4x4 grid of sub-sims that can interact at borders
    pub parallel_layout: [usize; 2],
    /// Enable border interaction between adjacent sub-simulations
    pub border_interaction: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CheckpointConfig {
    /// Enable checkpointing
    pub enabled: bool,
    /// Save checkpoint every N epochs (0 = only at end)
    pub interval: usize,
    /// Directory for checkpoint files
    pub path: String,
    /// Resume from this checkpoint file (empty = start fresh)
    pub resume_from: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    pub frame_interval: usize,
    pub frames_dir: String,
    /// Frame format: "png" (compressed) or "ppm" (uncompressed)
    pub frame_format: String,
    /// Downscale factor for regular frames (1 = full, 4 = 1/4 size)
    pub thumbnail_scale: usize,
    /// Save raw soup data (fast binary dumps)
    #[serde(default)]
    pub save_raw: bool,
    /// Directory for raw data files
    #[serde(default = "default_raw_dir")]
    pub raw_dir: String,
    /// Save in background thread (non-blocking)
    #[serde(default = "default_true")]
    pub async_save: bool,
    /// Also render frames during simulation
    #[serde(default)]
    pub render_frames: bool,
}

fn default_raw_dir() -> String {
    "raw_data".to_string()
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EnergySettings {
    pub enabled: bool,
    pub sources: usize,
    pub radius: usize,
    pub reserve_epochs: u32,
    pub death_epochs: u32,  // 0 = infinite (never dies from timeout)
    /// Spontaneous generation rate (1 in N chance per dead tape in energy zone per epoch, 0 = disabled)
    pub spontaneous_rate: u32,
    /// Shape of energy zones: "circle", "strip_h", "strip_v", "half_circle", "ellipse", "random"
    pub shape: String,
    /// Dynamic energy options
    pub dynamic: DynamicEnergySettings,
    /// Per-simulation group configs (optional - allows different death_epochs per sim)
    /// If empty, all sims use the global death_epochs and reserve_epochs values
    #[serde(default)]
    pub sim_groups: Vec<SimGroupConfig>,
}

/// Configuration for a group of simulations with shared energy parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimGroupConfig {
    /// Number of simulations in this group (if not specified, remaining sims are split evenly)
    #[serde(default)]
    pub count: Option<usize>,
    /// Death timer for this group (0 = infinite)
    pub death_epochs: u32,
    /// Reserve duration for this group (if not specified, uses global value)
    #[serde(default)]
    pub reserve_epochs: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DynamicEnergySettings {
    pub random_placement: bool,
    pub max_sources: usize,
    /// Epochs until source expires (0 = infinite)
    pub source_lifetime: usize,
    /// Spawn new source every N epochs (0 = disabled)
    pub spawn_rate: usize,
}

/// Metrics settings for tracking phase transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MetricsSettings {
    /// Enable metrics collection (Brotli compression ratio tracking)
    pub enabled: bool,
    /// Collect metrics every N epochs
    pub interval: usize,
    /// Path to CSV output file (optional, metrics also printed to stdout)
    pub output_file: String,
    /// Brotli compression quality (1-11, lower = faster, default 4)
    pub brotli_quality: u32,
}

impl Default for MetricsSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            interval: 1000,
            output_file: String::new(),
            brotli_quality: 4,
        }
    }
}

impl Default for GridConfig {
    fn default() -> Self {
        Self { width: 512, height: 256 }
    }
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            mutation_rate: 4096,
            steps_per_run: 8192,
            max_epochs: 10000,
            neighbor_range: 2,
            auto_terminate_dead_epochs: 0, // Disabled by default (has GPU overhead)
            parallel_sims: 1,
            parallel_layout: [1, 1],  // Default: single sim or independent sims
            border_interaction: false, // Disabled by default
        }
    }
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interval: 10000,
            path: "checkpoints".to_string(),
            resume_from: String::new(),
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            frame_interval: 64,
            frames_dir: "frames".to_string(),
            frame_format: "png".to_string(),
            thumbnail_scale: 1, // Full resolution by default
            save_raw: false,
            raw_dir: "raw_data".to_string(),
            async_save: true,
            render_frames: true,
        }
    }
}

impl Default for EnergySettings {
    fn default() -> Self {
        Self {
            enabled: false,
            sources: 4,
            radius: 64,
            reserve_epochs: 5,
            death_epochs: 10,
            spontaneous_rate: 0,  // Disabled by default
            shape: "circle".to_string(),
            dynamic: DynamicEnergySettings::default(),
            sim_groups: Vec::new(),  // Empty = all sims use global values
        }
    }
}

impl EnergySettings {
    /// Expand sim_groups into per-simulation configs (death_timer, reserve_duration pairs)
    /// Returns a Vec of (death_timer, reserve_duration) for each simulation
    /// 
    /// # Behavior
    /// - Groups with explicit `count` are allocated first (capped to remaining sims)
    /// - Groups without `count` split the remaining sims evenly
    /// - Any remaining sims after all groups use global defaults
    /// - Warns if total explicit counts exceed num_sims
    pub fn expand_sim_configs(&self, num_sims: usize) -> Vec<(u32, u32)> {
        if num_sims == 0 {
            return Vec::new();
        }
        
        if self.sim_groups.is_empty() {
            // No groups defined - all sims use global values
            return vec![(self.death_epochs, self.reserve_epochs); num_sims];
        }
        
        let mut configs = Vec::with_capacity(num_sims);
        
        // First pass: count sims with explicit counts
        let explicit_count: usize = self.sim_groups.iter()
            .filter_map(|g| g.count)
            .sum();
        
        // Warn if explicit counts exceed total sims
        if explicit_count > num_sims {
            eprintln!("Warning: sim_groups explicit counts ({}) exceed parallel_sims ({}). Some groups will be truncated.", 
                explicit_count, num_sims);
        }
        
        // Groups without explicit counts split the remaining sims evenly
        let groups_without_count = self.sim_groups.iter()
            .filter(|g| g.count.is_none())
            .count();
        
        let remaining_after_explicit = num_sims.saturating_sub(explicit_count);
        let per_group_default = if groups_without_count > 0 {
            remaining_after_explicit / groups_without_count
        } else {
            0
        };
        
        // Distribute remainder across groups without explicit counts (first groups get extras)
        let mut extra_sims = if groups_without_count > 0 {
            remaining_after_explicit % groups_without_count
        } else {
            0
        };
        
        let mut remaining_sims = num_sims;
        
        for group in &self.sim_groups {
            if remaining_sims == 0 {
                break; // No more sims to allocate
            }
            
            let count = match group.count {
                Some(c) => c.min(remaining_sims),
                None => {
                    // Distribute evenly with extras going to first groups
                    let extra_for_this = if extra_sims > 0 { 
                        extra_sims -= 1; 
                        1 
                    } else { 
                        0 
                    };
                    (per_group_default + extra_for_this).min(remaining_sims)
                }
            };
            
            let reserve = group.reserve_epochs.unwrap_or(self.reserve_epochs);
            for _ in 0..count {
                configs.push((group.death_epochs, reserve));
            }
            remaining_sims = remaining_sims.saturating_sub(count);
        }
        
        // Fill any remaining sims with global defaults
        if remaining_sims > 0 {
            eprintln!("Note: {} sims not covered by sim_groups, using global defaults (death_epochs={}, reserve_epochs={})",
                remaining_sims, self.death_epochs, self.reserve_epochs);
            for _ in 0..remaining_sims {
                configs.push((self.death_epochs, self.reserve_epochs));
            }
        }
        
        configs
    }
}

impl Default for DynamicEnergySettings {
    fn default() -> Self {
        Self {
            random_placement: false,
            max_sources: 8,
            source_lifetime: 0,
            spawn_rate: 0,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            grid: GridConfig::default(),
            simulation: SimConfig::default(),
            output: OutputConfig::default(),
            energy: EnergySettings::default(),
            checkpoint: CheckpointConfig::default(),
            metrics: MetricsSettings::default(),
        }
    }
}

impl Config {
    /// Load config from a YAML file
    pub fn from_yaml(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let config: Config = serde_yaml::from_str(&contents)?;
        Ok(config)
    }
    
    /// Save config to a YAML file
    pub fn to_yaml(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let yaml = serde_yaml::to_string(self)?;
        std::fs::write(path, yaml)?;
        Ok(())
    }
    
    /// Validate configuration and return errors/warnings
    /// Returns Err if there are fatal configuration errors
    pub fn validate(&self) -> Result<Vec<String>, String> {
        let mut warnings = Vec::new();
        
        // Check grid dimensions
        if self.grid.width == 0 || self.grid.height == 0 {
            return Err("Grid dimensions must be non-zero".to_string());
        }
        
        // Check parallel_layout matches parallel_sims when border_interaction is enabled
        let [layout_cols, layout_rows] = self.simulation.parallel_layout;
        let layout_product = layout_cols * layout_rows;
        
        if self.simulation.border_interaction {
            if layout_product != self.simulation.parallel_sims {
                return Err(format!(
                    "parallel_layout [{}, {}] = {} does not match parallel_sims = {}. \
                    When border_interaction is enabled, layout must multiply to parallel_sims.",
                    layout_cols, layout_rows, layout_product, self.simulation.parallel_sims
                ));
            }
            if layout_cols <= 1 && layout_rows <= 1 {
                return Err(
                    "border_interaction requires parallel_layout with at least 2 rows or columns".to_string()
                );
            }
        } else if layout_product != self.simulation.parallel_sims && layout_product > 1 {
            warnings.push(format!(
                "parallel_layout [{}, {}] = {} differs from parallel_sims = {}. \
                This is only meaningful when border_interaction is enabled.",
                layout_cols, layout_rows, layout_product, self.simulation.parallel_sims
            ));
        }
        
        // Check mutation_rate
        if self.simulation.mutation_rate == 0 {
            warnings.push("mutation_rate is 0, will be treated as 1 (maximum mutation)".to_string());
        }
        
        // Check energy radius vs grid size
        if self.energy.enabled {
            let min_dim = self.grid.width.min(self.grid.height);
            if self.energy.radius * 2 >= min_dim {
                warnings.push(format!(
                    "energy radius {} is large relative to grid size {}x{}. \
                    Sources may overlap significantly.",
                    self.energy.radius, self.grid.width, self.grid.height
                ));
            }
            
            // Check death_epochs fits in 15 bits (max 32767)
            if self.energy.death_epochs > 32767 {
                return Err(format!(
                    "death_epochs {} exceeds maximum of 32767 (15-bit limit in GPU state packing)",
                    self.energy.death_epochs
                ));
            }
            
            // Check reserve_epochs fits in 16 bits (max 65535)
            if self.energy.reserve_epochs > 65535 {
                return Err(format!(
                    "reserve_epochs {} exceeds maximum of 65535 (16-bit limit in GPU state packing)",
                    self.energy.reserve_epochs
                ));
            }
            
            // Also check sim_groups for out-of-range values
            for (i, group) in self.energy.sim_groups.iter().enumerate() {
                if group.death_epochs > 32767 {
                    return Err(format!(
                        "sim_groups[{}].death_epochs {} exceeds maximum of 32767",
                        i, group.death_epochs
                    ));
                }
                if let Some(reserve) = group.reserve_epochs {
                    if reserve > 65535 {
                        return Err(format!(
                            "sim_groups[{}].reserve_epochs {} exceeds maximum of 65535",
                            i, reserve
                        ));
                    }
                }
            }
        }
        
        // Check steps_per_run is reasonable
        if self.simulation.steps_per_run == 0 {
            return Err("steps_per_run must be greater than 0".to_string());
        }
        
        // Check frame_interval
        if self.output.frame_interval > self.simulation.max_epochs {
            warnings.push(format!(
                "frame_interval {} is greater than max_epochs {}, no frames will be saved",
                self.output.frame_interval, self.simulation.max_epochs
            ));
        }
        
        Ok(warnings)
    }
    
    /// Generate a template config file
    pub fn write_template(path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let config = Config::default();
        config.to_yaml(path)
    }
    
    /// Convert mutation_rate (1/N) to internal mutation_prob
    /// Returns max probability if mutation_rate is 0 (prevents division by zero)
    pub fn mutation_prob(&self) -> u32 {
        let rate = self.simulation.mutation_rate.max(1);
        ((1u64 << 30) / rate as u64) as u32
    }
}

/// Command-line arguments (internal, maps from Config)
/// Note: Some fields are only used with specific feature flags (cuda vs wgpu)
#[allow(dead_code)]
struct Args {
    grid_width: usize,
    grid_height: usize,
    seed: u64,
    mutation_prob: u32,
    steps_per_run: usize,
    max_epochs: usize,
    neighbor_range: usize,
    frame_interval: usize,
    frames_dir: String,
    frame_format: String,
    thumbnail_scale: usize,
    auto_terminate_dead_epochs: usize,
    parallel_sims: usize,
    // Mega-simulation options
    parallel_layout: [usize; 2],
    border_interaction: bool,
    // Checkpoint options
    checkpoint_enabled: bool,
    checkpoint_interval: usize,
    checkpoint_path: String,
    checkpoint_resume_from: String,
    // Raw data / async save options
    save_raw: bool,
    raw_dir: String,
    async_save: bool,
    render_frames: bool,
    // Special modes
    render_raw_path: Option<String>,  // --render-raw <path>
    // Energy system options
    energy_enabled: bool,
    energy_sources: usize,
    energy_radius: usize,
    energy_reserve: u32,
    energy_death: u32,  // 0 = infinite (never dies from timeout)
    energy_spontaneous_rate: u32,
    energy_shape: String,
    // Dynamic energy options
    energy_random: bool,
    energy_max_sources: usize,
    energy_source_lifetime: usize,
    energy_spawn_rate: usize,
    // Per-simulation energy configs (for running different death_epochs in same run)
    energy_sim_groups: Vec<SimGroupConfig>,
    // Metrics options (compression ratio tracking for phase transitions)
    metrics_enabled: bool,
    metrics_interval: usize,
    metrics_output_file: String,
    metrics_brotli_quality: u32,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            grid_width: 512,
            grid_height: 256,
            seed: 42,
            mutation_prob: 1 << 18, // ~1/4096
            steps_per_run: 8192,
            max_epochs: 10000,
            neighbor_range: 2,
            frame_interval: 64,
            frames_dir: "frames".to_string(),
            frame_format: "png".to_string(),
            thumbnail_scale: 1,
            auto_terminate_dead_epochs: 0,
            parallel_sims: 1,
            // Mega-simulation defaults
            parallel_layout: [1, 1],
            border_interaction: false,
            // Checkpoint defaults
            checkpoint_enabled: false,
            checkpoint_interval: 10000,
            checkpoint_path: "checkpoints".to_string(),
            checkpoint_resume_from: String::new(),
            // Raw data / async save defaults
            save_raw: false,
            raw_dir: "raw_data".to_string(),
            async_save: true,
            render_frames: true,
            render_raw_path: None,
            // Energy defaults
            energy_enabled: false,
            energy_sources: 4,
            energy_radius: 64,
            energy_reserve: 5,
            energy_death: 10,
            energy_spontaneous_rate: 0,
            energy_shape: "circle".to_string(),
            // Dynamic energy defaults
            energy_random: false,
            energy_max_sources: 8,
            energy_source_lifetime: 0,  // 0 = infinite
            energy_spawn_rate: 0,       // 0 = disabled
            // Per-sim configs
            energy_sim_groups: Vec::new(),
            // Metrics defaults
            metrics_enabled: false,
            metrics_interval: 1000,
            metrics_output_file: String::new(),
            metrics_brotli_quality: 4,
        }
    }
}

impl From<Config> for Args {
    fn from(c: Config) -> Self {
        Self {
            grid_width: c.grid.width,
            grid_height: c.grid.height,
            seed: c.simulation.seed,
            mutation_prob: c.mutation_prob(),
            steps_per_run: c.simulation.steps_per_run,
            max_epochs: c.simulation.max_epochs,
            neighbor_range: c.simulation.neighbor_range,
            frame_interval: c.output.frame_interval,
            frames_dir: c.output.frames_dir,
            frame_format: c.output.frame_format,
            thumbnail_scale: c.output.thumbnail_scale.max(1),
            auto_terminate_dead_epochs: c.simulation.auto_terminate_dead_epochs,
            parallel_sims: c.simulation.parallel_sims,
            parallel_layout: c.simulation.parallel_layout,
            border_interaction: c.simulation.border_interaction,
            checkpoint_enabled: c.checkpoint.enabled,
            checkpoint_interval: c.checkpoint.interval,
            checkpoint_path: c.checkpoint.path,
            checkpoint_resume_from: c.checkpoint.resume_from,
            save_raw: c.output.save_raw,
            raw_dir: c.output.raw_dir,
            async_save: c.output.async_save,
            render_frames: c.output.render_frames,
            render_raw_path: None,  // Only set via CLI
            energy_enabled: c.energy.enabled,
            energy_sources: c.energy.sources,
            energy_radius: c.energy.radius,
            energy_reserve: c.energy.reserve_epochs,
            energy_death: c.energy.death_epochs,
            energy_spontaneous_rate: c.energy.spontaneous_rate,
            energy_shape: c.energy.shape,
            energy_random: c.energy.dynamic.random_placement,
            energy_max_sources: c.energy.dynamic.max_sources,
            energy_source_lifetime: c.energy.dynamic.source_lifetime,
            energy_spawn_rate: c.energy.dynamic.spawn_rate,
            energy_sim_groups: c.energy.sim_groups.clone(),
            // Metrics
            metrics_enabled: c.metrics.enabled,
            metrics_interval: c.metrics.interval,
            metrics_output_file: c.metrics.output_file,
            metrics_brotli_quality: c.metrics.brotli_quality,
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let argv: Vec<String> = env::args().collect();
    
    // First pass: check for --config or --generate-config
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--config" | "-c" => {
                i += 1;
                let config_path = &argv[i];
                match Config::from_yaml(config_path) {
                    Ok(config) => {
                        println!("Loaded config from: {}", config_path);
                        
                        // Validate configuration
                        match config.validate() {
                            Ok(warnings) => {
                                for warning in warnings {
                                    eprintln!("Config warning: {}", warning);
                                }
                            }
                            Err(e) => {
                                eprintln!("Config validation error: {}", e);
                                std::process::exit(1);
                            }
                        }
                        
                        args = Args::from(config);
                    }
                    Err(e) => {
                        eprintln!("Error loading config file '{}': {}", config_path, e);
                        std::process::exit(1);
                    }
                }
            }
            "--generate-config" => {
                i += 1;
                let output_path = if i < argv.len() && !argv[i].starts_with('-') {
                    argv[i].clone()
                } else {
                    // No argument provided, use default
                    "config.yaml".to_string()
                };
                match Config::write_template(&output_path) {
                    Ok(_) => {
                        println!("Generated config template: {}", output_path);
                        std::process::exit(0);
                    }
                    Err(e) => {
                        eprintln!("Error writing config template: {}", e);
                        std::process::exit(1);
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }
    
    // Second pass: CLI args override config file values
    i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--config" | "-c" => {
                i += 1; // skip, already processed
            }
            "--grid-width" | "-w" => {
                i += 1;
                args.grid_width = argv[i].parse().expect("Invalid grid-width");
            }
            "--grid-height" | "-h" => {
                i += 1;
                args.grid_height = argv[i].parse().expect("Invalid grid-height");
            }
            "--seed" | "-s" => {
                i += 1;
                args.seed = argv[i].parse().expect("Invalid seed");
            }
            "--mutation-prob" | "-m" => {
                i += 1;
                args.mutation_prob = argv[i].parse().expect("Invalid mutation-prob");
            }
            "--steps-per-run" => {
                i += 1;
                args.steps_per_run = argv[i].parse().expect("Invalid steps-per-run");
            }
            "--max-epochs" | "-e" => {
                i += 1;
                args.max_epochs = argv[i].parse().expect("Invalid max-epochs");
            }
            "--neighbor-range" | "-n" => {
                i += 1;
                args.neighbor_range = argv[i].parse().expect("Invalid neighbor-range");
            }
            "--frame-interval" | "-f" => {
                i += 1;
                args.frame_interval = argv[i].parse().expect("Invalid frame-interval");
            }
            "--frames-dir" | "-d" => {
                i += 1;
                args.frames_dir = argv[i].clone();
            }
            "--energy" => {
                args.energy_enabled = true;
            }
            "--energy-sources" => {
                i += 1;
                let count: usize = argv[i].parse().expect("Invalid energy-sources");
                if count > 8 {
                    eprintln!("Warning: energy-sources capped at 8");
                    args.energy_sources = 8;
                } else {
                    args.energy_sources = count;
                }
            }
            "--energy-radius" => {
                i += 1;
                args.energy_radius = argv[i].parse().expect("Invalid energy-radius");
            }
            "--energy-reserve" => {
                i += 1;
                args.energy_reserve = argv[i].parse().expect("Invalid energy-reserve");
            }
            "--energy-death" => {
                i += 1;
                args.energy_death = argv[i].parse().expect("Invalid energy-death");
            }
            "--energy-random" => {
                args.energy_random = true;
            }
            "--energy-max-sources" => {
                i += 1;
                args.energy_max_sources = argv[i].parse().expect("Invalid energy-max-sources");
            }
            "--energy-source-lifetime" => {
                i += 1;
                args.energy_source_lifetime = argv[i].parse().expect("Invalid energy-source-lifetime");
            }
            "--energy-spawn-rate" => {
                i += 1;
                args.energy_spawn_rate = argv[i].parse().expect("Invalid energy-spawn-rate");
            }
            "--save-raw" => {
                args.save_raw = true;
            }
            "--no-save-raw" => {
                args.save_raw = false;
            }
            "--raw-dir" => {
                i += 1;
                args.raw_dir = argv[i].clone();
            }
            "--async-save" => {
                args.async_save = true;
            }
            "--no-async-save" => {
                args.async_save = false;
            }
            "--render-frames" => {
                args.render_frames = true;
            }
            "--no-render-frames" => {
                args.render_frames = false;
            }
            "--render-raw" => {
                i += 1;
                args.render_raw_path = Some(argv[i].clone());
            }
            "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                print_help();
                std::process::exit(1);
            }
        }
        i += 1;
    }
    
    args
}

fn print_help() {
    println!("BFF Primordial Soup Simulation");
    println!();
    println!("USAGE:");
    println!("    energetic-primordial-soup [OPTIONS]");
    println!("    energetic-primordial-soup --config config.yaml");
    println!("    energetic-primordial-soup --generate-config [output.yaml]");
    println!();
    println!("CONFIG FILE:");
    println!("    -c, --config <FILE>       Load settings from YAML config file");
    println!("    --generate-config [FILE]  Generate template config (default: config.yaml)");
    println!();
    println!("OPTIONS (override config file values):");
    println!("    -w, --grid-width <N>      Grid width (default: 512)");
    println!("    -h, --grid-height <N>     Grid height (default: 256)");
    println!("    -s, --seed <N>            Random seed (default: 42)");
    println!("    -m, --mutation-prob <N>   Mutation probability (default: 262144)");
    println!("    --steps-per-run <N>       Steps per BFF run (default: 8192)");
    println!("    -e, --max-epochs <N>      Maximum epochs (default: 10000)");
    println!("    -n, --neighbor-range <N>  Neighbor range (default: 2)");
    println!("    -f, --frame-interval <N>  Save frame every N epochs (0 = disabled)");
    println!("    -d, --frames-dir <PATH>   Frames output directory (default: frames)");
    println!();
    println!("ENERGY SYSTEM:");
    println!("    --energy                  Enable energy sources");
    println!("    --energy-sources <N>      Initial sources 1-8 (default: 4)");
    println!("                              4=corners, 5=+center, 6=+edges, 8=all");
    println!("    --energy-radius <N>       Radius of each source (default: 64)");
    println!("    --energy-reserve <N>      Reserve epochs when leaving zone (default: 5)");
    println!("    --energy-death <N>        Epochs until program death (default: 10)");
    println!();
    println!("DYNAMIC ENERGY:");
    println!("    --energy-random           Randomize source positions");
    println!("    --energy-max-sources <N>  Max simultaneous sources (default: 8)");
    println!("    --energy-source-lifetime <N>");
    println!("                              Epochs until source expires (0=infinite)");
    println!("    --energy-spawn-rate <N>   Spawn new source every N epochs (0=disabled)");
    println!();
    println!("RAW DATA / ASYNC SAVE:");
    println!("    --save-raw                Save raw soup data (fast binary dumps)");
    println!("    --no-save-raw             Disable raw data saving");
    println!("    --raw-dir <PATH>          Raw data output directory (default: raw_data)");
    println!("    --async-save              Save in background thread (non-blocking)");
    println!("    --no-async-save           Save synchronously (blocking)");
    println!("    --render-frames           Render frames during simulation");
    println!("    --no-render-frames        Skip frame rendering (only save raw data)");
    println!();
    println!("POST-PROCESSING:");
    println!("    --render-raw <PATH>       Render frames from saved raw data directory");
    println!("                              (runs rendering only, no simulation)");
    println!();
    println!("    --help                    Print this help message");
}

fn main() {
    let mut args = parse_args();
    
    // Generate unique run ID based on timestamp
    let run_timestamp = {
        use std::time::{SystemTime, UNIX_EPOCH};
        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        format!("{}", duration.as_secs())
    };
    
    // Append run ID to frames_dir to make it unique per run
    // Format: frames_<timestamp> (e.g., frames_1735412345)
    if !args.frames_dir.contains(&run_timestamp) {
        args.frames_dir = format!("{}_{}", args.frames_dir, run_timestamp);
    }
    
    // Handle --render-raw mode (render frames from raw data and exit)
    if let Some(ref raw_path) = args.render_raw_path {
        println!("BFF Raw Data Renderer");
        println!("=====================\n");
        
        if let Err(e) = render_raw_directory(
            raw_path,
            &args.frames_dir,
            &args.frame_format,
            args.thumbnail_scale,
        ) {
            eprintln!("Error rendering raw data: {}", e);
            std::process::exit(1);
        }
        return;
    }
    
    let num_programs = args.grid_width * args.grid_height;
    let save_frames = args.frame_interval > 0 && args.render_frames;
    #[allow(unused_variables)]
    let save_raw = args.save_raw && args.frame_interval > 0;
    
    println!("BFF Primordial Soup Simulation");
    println!("==============================\n");
    
    println!("Configuration:");
    println!("  Programs: {} ({}x{} grid)", num_programs, args.grid_width, args.grid_height);
    println!("  Seed: {}", args.seed);
    println!("  Mutation prob: {} (~1/{})", args.mutation_prob, (1u64 << 30) / args.mutation_prob as u64);
    println!("  Steps per run: {}", args.steps_per_run);
    println!("  Max epochs: {}", args.max_epochs);
    println!("  Neighbor range: ±{}", args.neighbor_range);
    if args.parallel_sims > 1 {
        println!("  Parallel simulations: {}", args.parallel_sims);
    }
    
    // Energy system info
    if args.energy_enabled {
        println!("  Energy system: ENABLED");
        if args.energy_random {
            println!("    - {} random sources (radius: {})", args.energy_sources, args.energy_radius);
        } else {
            println!("    - {} fixed sources (radius: {})", args.energy_sources, args.energy_radius);
        }
        println!("    - Reserve epochs: {}", args.energy_reserve);
        println!("    - Death timer: {} epochs", args.energy_death);
        if args.energy_source_lifetime > 0 || args.energy_spawn_rate > 0 {
            println!("    - Dynamic mode:");
            println!("      - Max sources: {}", args.energy_max_sources);
            if args.energy_source_lifetime > 0 {
                println!("      - Source lifetime: {} epochs", args.energy_source_lifetime);
            }
            if args.energy_spawn_rate > 0 {
                println!("      - Spawn rate: every {} epochs", args.energy_spawn_rate);
            }
        }
    } else {
        println!("  Energy system: disabled");
    }
    
    // Create frames directory
    if save_frames {
        if let Err(e) = fs::create_dir_all(&args.frames_dir) {
            eprintln!("Warning: Could not create frames directory: {}", e);
        } else {
            println!("  Frame interval: {}", args.frame_interval);
            println!("  Frames dir: {}/", args.frames_dir);
            println!("  Image size: {}x{} pixels", args.grid_width * 8, args.grid_height * 8);
        }
    }
    
    // Build energy config for GPU backends
    #[allow(unused_variables)]
    let gpu_energy_config = if args.energy_enabled {
        Some(energy::EnergyConfig::full_with_options(
            args.grid_width,
            args.grid_height,
            args.energy_radius,
            args.energy_sources,
            args.energy_reserve,
            args.energy_death,
            args.energy_random,
            args.energy_max_sources,
            args.energy_source_lifetime,
            args.energy_spawn_rate,
            args.seed,
            args.energy_spontaneous_rate,
            args.energy_shape.clone(),
        ))
    } else {
        None
    };

    // Note: We no longer freeze_dynamic() here since CUDA now supports dynamic energy updates

    // Build per-sim configs if sim_groups specified
    #[allow(unused_variables)]
    let per_sim_configs: Option<Vec<(u32, u32)>> = if !args.energy_sim_groups.is_empty() {
        let energy_settings = EnergySettings {
            enabled: args.energy_enabled,
            sources: args.energy_sources,
            radius: args.energy_radius,
            reserve_epochs: args.energy_reserve,
            death_epochs: args.energy_death,
            spontaneous_rate: args.energy_spontaneous_rate,
            shape: args.energy_shape.clone(),
            dynamic: DynamicEnergySettings::default(),
            sim_groups: args.energy_sim_groups.clone(),
        };
        let configs = energy_settings.expand_sim_configs(args.parallel_sims);
        println!("  Per-sim energy configs: {} groups across {} sims", 
            args.energy_sim_groups.len(), args.parallel_sims);
        for (i, group) in args.energy_sim_groups.iter().enumerate() {
            let count_str = group.count.map(|c| format!("{}", c)).unwrap_or("auto".to_string());
            let death_str = if group.death_epochs == 0 { "infinite".to_string() } else { format!("{}", group.death_epochs) };
            println!("    Group {}: {} sims, death_epochs={}", i, count_str, death_str);
        }
        
        // Write sim_groups.txt
        if let Err(e) = std::fs::create_dir_all(&args.frames_dir) {
            eprintln!("Warning: Could not create frames directory: {}", e);
        } else {
            let groups_file = format!("{}/sim_groups.txt", args.frames_dir);
            let mut content = String::new();
            content.push_str("# Simulation Group Configuration\n");
            content.push_str("# Generated at run start\n");
            content.push_str(&format!("# Total simulations: {}\n\n", args.parallel_sims));
            
            let mut sim_idx = 0usize;
            for (i, group) in args.energy_sim_groups.iter().enumerate() {
                let count = group.count.unwrap_or_else(|| {
                    let explicit: usize = args.energy_sim_groups.iter().filter_map(|g| g.count).sum();
                    let remaining = args.parallel_sims.saturating_sub(explicit);
                    let without_count = args.energy_sim_groups.iter().filter(|g| g.count.is_none()).count();
                    remaining / without_count.max(1)
                });
                let death_str = if group.death_epochs == 0 { 
                    "infinite (never dies from timeout)".to_string() 
                } else { 
                    format!("{} epochs", group.death_epochs) 
                };
                let reserve = group.reserve_epochs.unwrap_or(args.energy_reserve);
                
                content.push_str(&format!("## Group {} - death_epochs: {}\n", i, death_str));
                content.push_str(&format!("   reserve_epochs: {}\n", reserve));
                content.push_str(&format!("   sim_{} to sim_{}\n\n", sim_idx, sim_idx + count.saturating_sub(1)));
                sim_idx += count;
            }
            
            if sim_idx < args.parallel_sims {
                let death_str = if args.energy_death == 0 {
                    "infinite (never dies from timeout)".to_string()
                } else {
                    format!("{} epochs", args.energy_death)
                };
                content.push_str(&format!("## Remaining (global defaults) - death_epochs: {}\n", death_str));
                content.push_str(&format!("   reserve_epochs: {}\n", args.energy_reserve));
                content.push_str(&format!("   sim_{} to sim_{}\n", sim_idx, args.parallel_sims - 1));
            }
            
            if let Err(e) = std::fs::write(&groups_file, &content) {
                eprintln!("Warning: Could not write sim_groups.txt: {}", e);
            } else {
                println!("  Wrote group config to: {}", groups_file);
            }
        }
        
        Some(configs)
    } else {
        None
    };
    
    // Try CUDA first (no 4GB buffer limit)
    #[cfg(feature = "cuda")]
    {
        if args.parallel_sims > 1 && cuda::cuda_available() {
            println!("\n  Backend: CUDA (no buffer size limit)\n");
            
            match cuda::CudaMultiSimulation::new(
                args.parallel_sims,
                num_programs,
                args.grid_width,
                args.grid_height,
                args.seed,
                args.mutation_prob,
                args.steps_per_run as u32,
                gpu_energy_config.as_ref(),
                per_sim_configs.clone(),
            ) {
                Ok(mut cuda_sim) => {
                    let [layout_cols, layout_rows] = args.parallel_layout;
                    let effective_border_interaction = args.border_interaction
                        && layout_cols * layout_rows == args.parallel_sims
                        && (layout_cols > 1 || layout_rows > 1);
                    
                    if args.border_interaction && !effective_border_interaction {
                        eprintln!("Warning: parallel_layout {:?} does not match parallel_sims {}",
                            args.parallel_layout, args.parallel_sims);
                        eprintln!("         Border interaction disabled.");
                    }
                    
                    if effective_border_interaction {
                        let mega_pairs = generate_mega_pairs(
                            num_programs, args.grid_width, args.grid_height, args.neighbor_range,
                            args.parallel_sims, layout_cols, layout_rows,
                        );
                        cuda_sim.set_mega_mode(true);
                        cuda_sim.set_pairs_mega(&mega_pairs);
                        
                        println!("Mega-simulation mode: {}x{} grid ({} sub-sims)",
                            layout_cols, layout_rows, args.parallel_sims);
                        println!("  Total grid: {}x{} programs",
                            args.grid_width * layout_cols, args.grid_height * layout_rows);
                        println!("  Total pairs per epoch: {} (including {} cross-border)",
                            mega_pairs.len(),
                            mega_pairs.iter().filter(|(a, b)| a / num_programs as u32 != b / num_programs as u32).count());
                    } else {
                        let pairs = generate_2d_pairs(num_programs, args.grid_width, args.grid_height, args.neighbor_range);
                        cuda_sim.set_pairs_all(&pairs);
                    }
                    
                    // Run CUDA simulation
                    run_cuda_simulation(
                        &mut cuda_sim,
                        args.max_epochs,
                        &args.frames_dir,
                        save_frames,
                        args.frame_interval,
                        &args.frame_format,
                        args.thumbnail_scale,
                        args.parallel_layout,
                        effective_border_interaction,
                        args.checkpoint_enabled,
                        args.checkpoint_interval,
                        &args.checkpoint_path,
                        &args.checkpoint_resume_from,
                        args.seed,
                        save_raw,
                        &args.raw_dir,
                        args.async_save,
                        gpu_energy_config,
                        args.neighbor_range,
                        metrics::MetricsConfig {
                            enabled: args.metrics_enabled,
                            interval: args.metrics_interval,
                            output_path: if args.metrics_output_file.is_empty() { None } else { Some(args.metrics_output_file.clone()) },
                            brotli_quality: args.metrics_brotli_quality,
                        },
                    );
                    return;
                }
                Err(e) => {
                    eprintln!("CUDA initialization failed: {}", e);
                    println!("Falling back to wgpu...");
                }
            }
        }
    }
    
    // Try wgpu (cross-platform, 4GB buffer limit)
    #[cfg(feature = "wgpu-compute")]
    {
        if args.parallel_sims > 1 {
            // Multi-simulation mode (gpu_energy_config and per_sim_configs already built above)
            if let Some(mut multi_sim) = gpu::wgpu_sim::MultiWgpuSimulation::new(
                args.parallel_sims,
                num_programs,
                args.grid_width,
                args.grid_height,
                args.seed,
                args.mutation_prob,
                args.steps_per_run as u32,
                gpu_energy_config.as_ref(),
                per_sim_configs,
            ) {
                let mode_desc = if args.border_interaction {
                    format!("mega-sim ({}x{} grid)", args.parallel_layout[0], args.parallel_layout[1])
                } else {
                    format!("{} parallel simulations", args.parallel_sims)
                };
                println!("\n  Backend: GPU (wgpu/Vulkan) - {}\n", mode_desc);
                run_multi_gpu_simulation(
                    &mut multi_sim,
                    gpu_energy_config,
                    num_programs,
                    args.grid_width,
                    args.grid_height,
                    args.max_epochs,
                    &args.frames_dir,
                    save_frames,
                    args.frame_interval,
                    &args.frame_format,
                    args.thumbnail_scale,
                    args.neighbor_range,
                    args.parallel_layout,
                    args.border_interaction,
                    args.checkpoint_enabled,
                    args.checkpoint_interval,
                    &args.checkpoint_path,
                    &args.checkpoint_resume_from,
                    args.seed,
                    save_raw,
                    &args.raw_dir,
                    args.async_save,
                    metrics::MetricsConfig {
                        enabled: args.metrics_enabled,
                        interval: args.metrics_interval,
                        output_path: if args.metrics_output_file.is_empty() { None } else { Some(args.metrics_output_file.clone()) },
                        brotli_quality: args.metrics_brotli_quality,
                    },
                );
                return;
            }
            println!("  Multi-GPU simulation failed, trying single...\n");
        }
        
        if let Some(mut gpu_sim) = gpu::wgpu_sim::WgpuSimulation::new(
            num_programs,
            args.grid_width,
            args.grid_height,
            args.seed,
            args.mutation_prob,
            args.steps_per_run as u32,
            gpu_energy_config.as_ref(),
        ) {
            println!("\n  Backend: GPU (wgpu/Vulkan)\n");
            let completed = run_gpu_simulation(
                &mut gpu_sim,
                gpu_energy_config,
                num_programs,
                args.grid_width,
                args.grid_height,
                args.max_epochs,
                &args.frames_dir,
                save_frames,
                args.frame_interval,
                args.neighbor_range,
                args.auto_terminate_dead_epochs,
                args.seed,
            );
            if !completed {
                std::process::exit(2); // Exit code 2 = terminated early
            }
            return;
        }
        println!("  GPU not available, falling back to CPU...\n");
    }
    
    // CPU fallback
    run_cpu_simulation(
        num_programs,
        args.grid_width,
        args.grid_height,
        args.seed,
        args.mutation_prob,
        args.steps_per_run,
        args.max_epochs,
        &args.frames_dir,
        save_frames,
        args.frame_interval,
        args.neighbor_range,
        args.energy_enabled,
        args.energy_sources,
        args.energy_radius,
        args.energy_reserve,
        args.energy_death,
        args.energy_spontaneous_rate,
        &args.energy_shape,
        args.energy_random,
        args.energy_max_sources,
        args.energy_source_lifetime,
        args.energy_spawn_rate,
    );
}

#[allow(dead_code)]
fn run_cpu_simulation(
    num_programs: usize,
    grid_width: usize,
    grid_height: usize,
    seed: u64,
    mutation_prob: u32,
    steps_per_run: usize,
    max_epochs: usize,
    frames_dir: &str,
    save_frames: bool,
    frame_interval: usize,
    neighbor_range: usize,
    energy_enabled: bool,
    energy_sources: usize,
    energy_radius: usize,
    energy_reserve: u32,
    energy_death: u32,  // 0 = infinite (never dies from timeout)
    energy_spontaneous_rate: u32,
    energy_shape: &str,
    energy_random: bool,
    energy_max_sources: usize,
    energy_source_lifetime: usize,
    energy_spawn_rate: usize,
) {
    println!("  Backend: CPU ({} threads)\n", rayon::current_num_threads());
    
    // Build energy config if enabled
    let energy_config = if energy_enabled {
        Some(energy::EnergyConfig::full_with_options(
            grid_width,
            grid_height,
            energy_radius,
            energy_sources,
            energy_reserve,
            energy_death,
            energy_random,
            energy_max_sources,
            energy_source_lifetime,
            energy_spawn_rate,
            seed,
            energy_spontaneous_rate,
            energy_shape.to_string(),
        ))
    } else {
        None
    };
    
    let params = SimulationParams {
        num_programs,
        seed,
        mutation_prob,
        callback_interval: 64,
        steps_per_run,
        zero_init: false,
        permute_programs: true,
        topology: Topology::Grid2D {
            width: grid_width,
            height: grid_height,
            neighbor_range,
        },
        energy_config,
    };
    
    let mut sim = Simulation::new(params);
    
    println!("Initial programs:");
    for i in 0..5 {
        sim.print_program(i);
    }
    println!();
    
    if save_frames {
        if let Err(e) = sim.save_frame(frames_dir, 0) {
            eprintln!("Warning: Could not save initial frame: {}", e);
        }
    }
    
    sim.run(|sim, state| {
        if save_frames && state.epoch % frame_interval == 0 {
            if let Err(e) = sim.save_frame(frames_dir, state.epoch) {
                eprintln!("Warning: Could not save frame {}: {}", state.epoch, e);
            }
        }
        
        if state.epoch % 256 == 0 {
            println!("\nSample programs at epoch {}:", state.epoch);
            for i in 0..5 {
                sim.print_program(i);
            }
            println!();
        }
        
        state.epoch >= max_epochs
    });
    
    if save_frames {
        if let Err(e) = sim.save_frame(frames_dir, max_epochs) {
            eprintln!("Warning: Could not save final frame: {}", e);
        }
    }
    
    println!("\nSimulation complete!");
}

#[cfg(feature = "cuda")]
fn run_cuda_simulation(
    cuda_sim: &mut cuda::CudaMultiSimulation,
    max_epochs: usize,
    frames_dir: &str,
    save_frames: bool,
    frame_interval: usize,
    frame_format: &str,
    thumbnail_scale: usize,
    parallel_layout: [usize; 2],
    mega_mode: bool,
    checkpoint_enabled: bool,
    checkpoint_interval: usize,
    checkpoint_path: &str,
    checkpoint_resume_from: &str,
    seed: u64,
    save_raw: bool,
    raw_dir: &str,
    async_save: bool,
    mut energy_config: Option<energy::EnergyConfig>,
    neighbor_range: usize,
    metrics_config: metrics::MetricsConfig,
) {
    use std::time::Instant;
    
    let num_sims = cuda_sim.num_sims();
    let grid_width = cuda_sim.grid_width();
    let grid_height = cuda_sim.grid_height();
    
    // Create output directories
    if save_frames {
        std::fs::create_dir_all(frames_dir).ok();
        for sim_idx in 0..num_sims {
            std::fs::create_dir_all(format!("{}/sim_{}", frames_dir, sim_idx)).ok();
        }
    }
    if save_raw {
        if let Err(e) = fs::create_dir_all(raw_dir) {
            eprintln!("Warning: Could not create raw data directory: {}", e);
        }
    }

    // Show save mode info
    if save_raw {
        if async_save {
            println!("Raw data saving: ENABLED (async, non-blocking)");
        } else {
            println!("Raw data saving: ENABLED (sync)");
        }
        println!("  Directory: {}", raw_dir);
    }
    if save_frames {
        if async_save {
            println!("Frame rendering: ENABLED (async, format: {})", frame_format);
        } else {
            println!("Frame rendering: ENABLED (format: {})", frame_format);
        }
    } else if save_raw {
        println!("Frame rendering: DISABLED (will render later with --render-raw)");
    }
    println!();

    // Create async writer if needed
    let async_writer = if (save_raw || save_frames) && async_save {
        Some(AsyncWriter::new())
    } else {
        None
    };

    // Check for checkpoint resume
    let mut start_epoch = 0usize;
    if !checkpoint_resume_from.is_empty() {
        match checkpoint::Checkpoint::load(checkpoint_resume_from) {
            Ok(ckpt) => {
                if let Err(e) = ckpt.validate(grid_width, grid_height, num_sims, parallel_layout) {
                    eprintln!("Checkpoint validation failed: {}", e);
                    eprintln!("Starting fresh simulation instead.");
                } else {
                    println!("Resuming from checkpoint: {}", checkpoint_resume_from);
                    println!("  - Epoch: {}", ckpt.header.epoch);
                    println!("  - Saved at: {}",
                        chrono_time_str(ckpt.header.timestamp));
                    cuda_sim.set_all_soup(&ckpt.soup);
                    cuda_sim.set_all_energy_states(&ckpt.energy_states);
                    cuda_sim.set_epoch(ckpt.header.epoch as u64);
                    start_epoch = ckpt.header.epoch;
                }
            }
            Err(e) => {
                eprintln!("Failed to load checkpoint '{}': {}", checkpoint_resume_from, e);
                eprintln!("Starting fresh simulation instead.");
            }
        }
    }
    
    let start_time = Instant::now();
    let mut total_ops: u64 = 0;
    let mut last_print = Instant::now();
    let mut last_checkpoint = start_epoch;
    
    let num_programs = cuda_sim.num_programs();
    
    // Create metrics tracker if enabled
    let mut metrics_tracker = if metrics_config.enabled {
        match metrics::MetricsTracker::new(metrics_config.clone()) {
            Ok(tracker) => {
                println!("Metrics tracking: ENABLED (interval: {} epochs)", metrics_config.interval);
                if !metrics_config.output_path.as_ref().map_or(true, |p| p.is_empty()) {
                    println!("  Output file: {}", metrics_config.output_path.as_ref().unwrap());
                }
                Some(tracker)
            }
            Err(e) => {
                eprintln!("Warning: Could not create metrics tracker: {}", e);
                None
            }
        }
    } else {
        None
    };
    
    println!("Running {} epochs across {} simulations...", max_epochs - start_epoch, num_sims);
    
    let layout_cols = parallel_layout[0];
    let layout_rows = parallel_layout[1];
    
    for epoch in start_epoch..max_epochs {
        // Regenerate pairs for this epoch (dynamic pairing for more variety)
        if mega_mode {
            let pairs = generate_mega_pairs_seeded(
                num_programs, grid_width, grid_height, neighbor_range,
                num_sims, layout_cols, layout_rows, seed, epoch
            );
            cuda_sim.set_pairs_mega(&pairs);
        } else {
            let pairs = generate_2d_pairs_seeded(
                num_programs, grid_width, grid_height, neighbor_range, seed, epoch
            );
            cuda_sim.set_pairs_all(&pairs);
        }
        
        // Update dynamic energy sources if needed
        if let Some(ref mut config) = energy_config {
            if config.is_dynamic() && config.update_sources(epoch) {
                cuda_sim.update_energy_config(config);
            }
        }

        let next_epoch_needs_save = frame_interval > 0
            && (save_raw || save_frames)
            && (epoch + 1) % frame_interval == 0
            && epoch + 1 < max_epochs;
        let this_epoch_needs_save = frame_interval > 0 && (save_raw || save_frames) && epoch % frame_interval == 0;

        let ops = cuda_sim.step();
        total_ops += ops;

        if next_epoch_needs_save && !cuda_sim.has_pending_readback() {
            cuda_sim.begin_async_readback();
        }
        
        // Collect metrics if enabled
        if let Some(ref mut tracker) = metrics_tracker {
            if tracker.should_collect(epoch) {
                let all_soup = cuda_sim.get_all_soup();
                let m = tracker.collect(epoch, &all_soup);
                // Print brief status
                print!("\r[Metrics] Epoch {:>8} | Compression: {:.2}x | Commands: {:.1}% | Zeros: {:.1}%    ",
                    epoch, m.compression_ratio, m.command_fraction * 100.0, m.zero_byte_fraction * 100.0);
                std::io::stdout().flush().ok();
            }
        }
        
        // Save checkpoint
        let will_checkpoint = checkpoint_enabled && checkpoint_interval > 0
            && epoch > 0 && (epoch - last_checkpoint) >= checkpoint_interval;
        if will_checkpoint {
            let soup = cuda_sim.get_all_soup();
            let energy_states = cuda_sim.get_all_energy_states();
            save_checkpoint_from_data(
                epoch + 1,
                grid_width,
                grid_height,
                num_sims,
                parallel_layout,
                mega_mode,
                seed,
                soup,
                energy_states,
                checkpoint_path,
            );
            last_checkpoint = epoch + 1;
        }

        // Save frames/raw data
        if this_epoch_needs_save {
            let all_soup = if cuda_sim.has_pending_readback() {
                cuda_sim.finish_async_readback().unwrap_or_else(|| cuda_sim.get_all_soup())
            } else {
                cuda_sim.get_all_soup()
            };

            if async_save && (save_raw || save_frames) {
                if let Some(ref writer) = async_writer {
                    writer.save_bundle(
                        all_soup,
                        epoch,
                        if save_raw { Some(raw_dir) } else { None },
                        if save_frames { Some(frames_dir) } else { None },
                        grid_width,
                        grid_height,
                        num_sims,
                        parallel_layout,
                        frame_format,
                        thumbnail_scale,
                        mega_mode,
                    );
                } else {
                    if save_raw {
                        if let Err(e) = save_raw_data_sync(
                            &all_soup,
                            epoch,
                            raw_dir,
                            grid_width,
                            grid_height,
                            num_sims,
                            parallel_layout,
                        ) {
                            eprintln!("Warning: Could not save raw data {}: {}", epoch, e);
                        }
                    }
                    if save_frames {
                        save_frames_from_data(
                            &all_soup,
                            num_sims,
                            grid_width,
                            grid_height,
                            frames_dir,
                            epoch,
                            frame_format,
                            thumbnail_scale,
                            mega_mode,
                            parallel_layout,
                        );
                    }
                }
            } else {
                if save_raw {
                    if let Err(e) = save_raw_data_sync(
                        &all_soup,
                        epoch,
                        raw_dir,
                        grid_width,
                        grid_height,
                        num_sims,
                        parallel_layout,
                    ) {
                        eprintln!("Warning: Could not save raw data {}: {}", epoch, e);
                    }
                }
                if save_frames {
                    save_frames_from_data(
                        &all_soup,
                        num_sims,
                        grid_width,
                        grid_height,
                        frames_dir,
                        epoch,
                        frame_format,
                        thumbnail_scale,
                        mega_mode,
                        parallel_layout,
                    );
                }
            }
        }
        
        // Progress update every second
        if last_print.elapsed().as_secs() >= 1 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let epochs_per_sec = (epoch + 1 - start_epoch) as f64 / elapsed;
            let ops_per_sec = total_ops as f64 / elapsed;
            let remaining = (max_epochs - epoch - 1) as f64 / epochs_per_sec;
            
            print!("\rEpoch {}/{} | {:.1} epochs/s | {:.2}B ops/s | ETA: {:.0}s    ",
                epoch + 1, max_epochs, epochs_per_sec, ops_per_sec / 1e9, remaining);
            std::io::Write::flush(&mut std::io::stdout()).ok();
            last_print = Instant::now();
        }
    }

    // Save final checkpoint
    if checkpoint_enabled {
        let soup = cuda_sim.get_all_soup();
        let energy_states = cuda_sim.get_all_energy_states();
        save_checkpoint_from_data(
            max_epochs,
            grid_width,
            grid_height,
            num_sims,
            parallel_layout,
            mega_mode,
            seed,
            soup,
            energy_states,
            checkpoint_path,
        );
    }

    // Shutdown async writer (wait for pending saves)
    if let Some(mut writer) = async_writer {
        println!("Waiting for async saves to complete...");
        writer.shutdown();
    }
    
    // Print metrics summary if enabled
    if let Some(tracker) = metrics_tracker {
        tracker.print_summary();
    }
    
    println!("\n\nCUDA simulation complete!");
    println!("  Total epochs: {}", max_epochs);
    println!("  Total ops: {:.2}B", total_ops as f64 / 1e9);
    println!("  Elapsed: {:.1}s", start_time.elapsed().as_secs_f64());
}

#[cfg(feature = "wgpu-compute")]
fn run_multi_gpu_simulation(
    multi_sim: &mut gpu::wgpu_sim::MultiWgpuSimulation,
    mut energy_config: Option<energy::EnergyConfig>,
    num_programs: usize,
    grid_width: usize,
    grid_height: usize,
    max_epochs: usize,
    frames_dir: &str,
    save_frames: bool,
    frame_interval: usize,
    frame_format: &str,
    thumbnail_scale: usize,
    neighbor_range: usize,
    parallel_layout: [usize; 2],
    border_interaction: bool,
    checkpoint_enabled: bool,
    checkpoint_interval: usize,
    checkpoint_path: &str,
    checkpoint_resume_from: &str,
    seed: u64,
    save_raw: bool,
    raw_dir: &str,
    async_save: bool,
    metrics_config: metrics::MetricsConfig,
) {
    use std::time::Instant;
    
    let num_sims = multi_sim.num_sims();
    let [layout_cols, layout_rows] = parallel_layout;
    
    // Validate layout
    if border_interaction && layout_cols * layout_rows != num_sims {
        eprintln!("Warning: parallel_layout {:?} does not match parallel_sims {}",
            parallel_layout, num_sims);
        eprintln!("         Border interaction disabled.");
    }
    
    let effective_border_interaction = border_interaction 
        && layout_cols * layout_rows == num_sims 
        && (layout_cols > 1 || layout_rows > 1);
    
    // Check for checkpoint resume
    let mut start_epoch = 0usize;
    if !checkpoint_resume_from.is_empty() {
        match checkpoint::Checkpoint::load(checkpoint_resume_from) {
            Ok(ckpt) => {
                // Validate checkpoint matches config
                if let Err(e) = ckpt.validate(grid_width, grid_height, num_sims, parallel_layout) {
                    eprintln!("Checkpoint validation failed: {}", e);
                    eprintln!("Starting fresh simulation instead.");
                } else {
                    println!("Resuming from checkpoint: {}", checkpoint_resume_from);
                    println!("  - Epoch: {}", ckpt.header.epoch);
                    println!("  - Saved at: {}", 
                        chrono_time_str(ckpt.header.timestamp));
                    
                    // Restore state
                    multi_sim.set_all_soup(&ckpt.soup);
                    multi_sim.set_all_energy_states(&ckpt.energy_states);
                    multi_sim.set_epoch(ckpt.header.epoch as u64);
                    start_epoch = ckpt.header.epoch;
                }
            }
            Err(e) => {
                eprintln!("Failed to load checkpoint '{}': {}", checkpoint_resume_from, e);
                eprintln!("Starting fresh simulation instead.");
            }
        }
    }
    
    // Initialize if not resuming
    if start_epoch == 0 {
        multi_sim.init_random_all();
    }
    
    // Generate pairs based on mode
    if effective_border_interaction {
        // Mega mode: generate all pairs with absolute indices (including cross-border)
        let mega_pairs = generate_mega_pairs(
            num_programs, grid_width, grid_height, neighbor_range,
            num_sims, layout_cols, layout_rows
        );
        multi_sim.set_mega_mode(true);
        multi_sim.set_pairs_mega(&mega_pairs);
        
        println!("Mega-simulation mode: {}x{} grid ({} sub-sims)", 
            layout_cols, layout_rows, num_sims);
        println!("  Total grid: {}x{} programs", 
            grid_width * layout_cols, grid_height * layout_rows);
        println!("  Total pairs per epoch: {} (including {} cross-border)", 
            mega_pairs.len(), 
            mega_pairs.iter().filter(|(a, b)| a / num_programs as u32 != b / num_programs as u32).count());
    } else {
        // Normal mode: pairs are local, shader adds sim offset
        let pairs = generate_2d_pairs(num_programs, grid_width, grid_height, neighbor_range);
        multi_sim.set_pairs_all(&pairs);
    }
    
    println!("Running {} epochs x {} simulations...\n", 
        max_epochs - start_epoch, num_sims);
    
    // Show save mode info
    if save_raw {
        if async_save {
            println!("Raw data saving: ENABLED (async, non-blocking)");
        } else {
            println!("Raw data saving: ENABLED (sync)");
        }
        println!("  Directory: {}", raw_dir);
        if let Err(e) = fs::create_dir_all(raw_dir) {
            eprintln!("Warning: Could not create raw data directory: {}", e);
        }
    }
    if save_frames {
        if async_save {
            println!("Frame rendering: ENABLED (async, format: {})", frame_format);
        } else {
            println!("Frame rendering: ENABLED (format: {})", frame_format);
        }
        let _ = fs::create_dir_all(frames_dir);
        for sim_idx in 0..num_sims {
            let _ = fs::create_dir_all(format!("{}/sim_{}", frames_dir, sim_idx));
        }
    } else if save_raw {
        println!("Frame rendering: DISABLED (will render later with --render-raw)");
    }
    println!();
    
    // Create async writer if needed
    let async_writer = if (save_raw || save_frames) && async_save {
        Some(AsyncWriter::new())
    } else {
        None
    };
    
    // Create metrics tracker if enabled
    let mut metrics_tracker = if metrics_config.enabled {
        match metrics::MetricsTracker::new(metrics_config.clone()) {
            Ok(tracker) => {
                println!("Metrics tracking: ENABLED (interval: {} epochs)", metrics_config.interval);
                if !metrics_config.output_path.as_ref().map_or(true, |p| p.is_empty()) {
                    println!("  Output file: {}", metrics_config.output_path.as_ref().unwrap());
                }
                Some(tracker)
            }
            Err(e) => {
                eprintln!("Warning: Could not create metrics tracker: {}", e);
                None
            }
        }
    } else {
        None
    };
    
    let mut total_ops = 0u64;
    let start_time = Instant::now();
    let mut last_report = Instant::now();
    let mut last_checkpoint = start_epoch;
    
    // Progress bar width
    const BAR_WIDTH: usize = 30;
    
    for epoch in start_epoch..max_epochs {
        // Regenerate pairs for this epoch (dynamic pairing for more variety)
        if effective_border_interaction {
            let pairs = generate_mega_pairs_seeded(
                num_programs, grid_width, grid_height, neighbor_range,
                num_sims, layout_cols, layout_rows, seed, epoch
            );
            multi_sim.set_pairs_mega(&pairs);
        } else {
            let pairs = generate_2d_pairs_seeded(
                num_programs, grid_width, grid_height, neighbor_range, seed, epoch
            );
            multi_sim.set_pairs_all(&pairs);
        }
        
        // Update dynamic energy sources if enabled
        if let Some(ref mut config) = energy_config {
            if config.is_dynamic() && config.update_sources(epoch) {
                multi_sim.update_energy_config_all(config);
            }
        }
        
        // Check if NEXT epoch needs saving - if so, start async readback after this epoch
        let next_epoch_needs_save = frame_interval > 0 
            && (save_raw || save_frames) 
            && (epoch + 1) % frame_interval == 0 
            && epoch + 1 < max_epochs;
        
        // Check if THIS epoch needs saving
        let this_epoch_needs_save = frame_interval > 0 && epoch % frame_interval == 0;
        
        // Run epoch (always blocking for stable throughput)
        let ops = multi_sim.run_epoch_all();
        total_ops += ops;
        
        // Start async readback for next epoch's save (copy happens while we do CPU work)
        if next_epoch_needs_save && !multi_sim.has_pending_readback() {
            multi_sim.begin_async_readback();
        }
        
        // Collect metrics if enabled
        if let Some(ref mut tracker) = metrics_tracker {
            if tracker.should_collect(epoch) {
                let all_soup = multi_sim.get_all_soup();
                let m = tracker.collect(epoch, &all_soup);
                // Print brief status
                print!("\r[Metrics] Epoch {:>8} | Compression: {:.2}x | Commands: {:.1}% | Zeros: {:.1}%    ",
                    epoch, m.compression_ratio, m.command_fraction * 100.0, m.zero_byte_fraction * 100.0);
                std::io::stdout().flush().ok();
            }
        }
        
        // Save checkpoint
        let will_checkpoint = checkpoint_enabled && checkpoint_interval > 0 
            && epoch > 0 && (epoch - last_checkpoint) >= checkpoint_interval;
        if will_checkpoint {
            save_checkpoint(
                multi_sim, epoch + 1, grid_width, grid_height, num_sims,
                parallel_layout, effective_border_interaction, seed, checkpoint_path
            );
            last_checkpoint = epoch + 1;
        }
        
        // Save raw data and/or frames at intervals
        if this_epoch_needs_save {
            // Get soup data - prefer async if available, otherwise sync
            let all_soup = if multi_sim.has_pending_readback() {
                // Use previously started async readback (should be ready now)
                multi_sim.finish_async_readback().unwrap_or_else(|| multi_sim.get_all_soup())
            } else {
                // No async pending (first save), use sync
                multi_sim.get_all_soup()
            };
            
            if async_save && (save_raw || save_frames) {
                if let Some(ref writer) = async_writer {
                    writer.save_bundle(
                        all_soup,
                        epoch,
                        if save_raw { Some(raw_dir) } else { None },
                        if save_frames { Some(frames_dir) } else { None },
                        grid_width,
                        grid_height,
                        num_sims,
                        parallel_layout,
                        frame_format,
                        thumbnail_scale,
                        effective_border_interaction,
                    );
                } else {
                    if save_raw {
                        if let Err(e) = save_raw_data_sync(
                            &all_soup,
                            epoch,
                            raw_dir,
                            grid_width,
                            grid_height,
                            num_sims,
                            parallel_layout,
                        ) {
                            eprintln!("Warning: Could not save raw data for epoch {}: {}", epoch, e);
                        }
                    }
                    if save_frames {
                        save_frames_from_data(
                            &all_soup,
                            num_sims,
                            grid_width,
                            grid_height,
                            frames_dir,
                            epoch,
                            frame_format,
                            thumbnail_scale,
                            effective_border_interaction,
                            parallel_layout,
                        );
                    }
                }
            } else {
                if save_raw {
                    if let Err(e) = save_raw_data_sync(
                        &all_soup,
                        epoch,
                        raw_dir,
                        grid_width,
                        grid_height,
                        num_sims,
                        parallel_layout,
                    ) {
                        eprintln!("Warning: Could not save raw data for epoch {}: {}", epoch, e);
                    }
                }
                if save_frames {
                    save_frames_from_data(
                        &all_soup,
                        num_sims,
                        grid_width,
                        grid_height,
                        frames_dir,
                        epoch,
                        frame_format,
                        thumbnail_scale,
                        effective_border_interaction,
                        parallel_layout,
                    );
                }
            }
        }
        
        // Report progress every second
        if last_report.elapsed().as_secs() >= 1 || epoch == max_epochs - 1 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let mops = total_ops as f64 / elapsed / 1_000_000.0;
            
            // Calculate progress
            let progress = (epoch + 1 - start_epoch) as f64 / (max_epochs - start_epoch) as f64;
            let filled = (progress * BAR_WIDTH as f64) as usize;
            let empty = BAR_WIDTH - filled;
            let bar: String = "█".repeat(filled) + &"░".repeat(empty);
            
            // Calculate ETA
            let eta_secs = if progress > 0.0 {
                (elapsed / progress - elapsed).max(0.0)
            } else {
                0.0
            };
            let eta_min = (eta_secs / 60.0).floor() as u64;
            let eta_sec = (eta_secs % 60.0).floor() as u64;
            
            let mode_str = if effective_border_interaction { "mega" } else { "multi" };
            print!("\r[{}] {:>3.0}% | {} {} | {:.1}B ops/s | ETA {:02}:{:02}  ",
                bar,
                progress * 100.0,
                num_sims,
                mode_str,
                mops / 1000.0,
                eta_min,
                eta_sec
            );
            use std::io::Write;
            std::io::stdout().flush().ok();
            
            last_report = Instant::now();
        }
    }
    println!(); // Newline after progress bar
    
    // Save final checkpoint
    if checkpoint_enabled {
        save_checkpoint(
            multi_sim, max_epochs, grid_width, grid_height, num_sims,
            parallel_layout, effective_border_interaction, seed, checkpoint_path
        );
    }
    
    // Save final raw data and/or frames
    if frame_interval > 0 {
        // Save final raw data
        if save_raw {
            let all_soup = multi_sim.get_all_soup();
            if let Some(ref writer) = async_writer {
                writer.save_bundle(
                    all_soup,
                    max_epochs,
                    Some(raw_dir),
                    None,
                    grid_width,
                    grid_height,
                    num_sims,
                    parallel_layout,
                    frame_format,
                    thumbnail_scale,
                    effective_border_interaction,
                );
            } else if let Err(e) = save_raw_data_sync(
                &all_soup,
                max_epochs,
                raw_dir,
                grid_width,
                grid_height,
                num_sims,
                parallel_layout,
            ) {
                eprintln!("Warning: Could not save final raw data: {}", e);
            }
        }
        
        // Save final rendered frames
        if save_frames {
            if effective_border_interaction {
                // Final mega frame - use same scale as regular frames
                // (auto-scaling in save_mega_frame will further reduce if needed)
                let _ = save_mega_frame(
                    multi_sim, layout_cols, layout_rows, grid_width, grid_height,
                    frames_dir, max_epochs, thumbnail_scale
                );
            }
            
            for sim_idx in 0..num_sims {
                let soup = multi_sim.get_soup(sim_idx);
                let sim_frames_dir = format!("{}/sim_{}", frames_dir, sim_idx);
                let _ = fs::create_dir_all(&sim_frames_dir);
                // Final frame at full resolution
                if let Err(e) = save_frame(&soup, grid_width, grid_height, &sim_frames_dir, max_epochs, frame_format, 1) {
                    eprintln!("Warning: Could not save final frame for sim {}: {}", sim_idx, e);
                }
            }
        }
    }
    
    // Shutdown async writer (wait for pending saves)
    if let Some(mut writer) = async_writer {
        println!("Waiting for async saves to complete...");
        writer.shutdown();
    }
    
    // Print metrics summary if enabled
    if let Some(tracker) = metrics_tracker {
        tracker.print_summary();
    }
    
    let elapsed = start_time.elapsed().as_secs_f64();
    let throughput = total_ops as f64 / elapsed / 1e9;
    let per_sim = throughput / num_sims as f64;
    
    println!();
    let mode_str = if effective_border_interaction { "Mega-Simulation" } else { "Simulations" };
    println!("┌─────────────────────────────────────────────┐");
    println!("│     ✓ {} Complete                   │", mode_str);
    println!("├─────────────────────────────────────────────┤");
    println!("│  Epochs/sim:    {:>12}               │", max_epochs);
    println!("│  Total epochs:  {:>12}  ({} × {})    │", max_epochs * num_sims, max_epochs, num_sims);
    println!("│  Time:          {:>12.2}s              │", elapsed);
    println!("│  Operations:    {:>12.2}G             │", total_ops as f64 / 1e9);
    println!("│  Throughput:    {:>12.1}B ops/sec     │", throughput);
    println!("│  Per-sim:       {:>12.1}B ops/sec     │", per_sim);
    println!("└─────────────────────────────────────────────┘");
}

#[cfg(feature = "wgpu-compute")]
fn run_gpu_simulation(
    gpu_sim: &mut gpu::wgpu_sim::WgpuSimulation,
    mut energy_config: Option<energy::EnergyConfig>,
    num_programs: usize,
    grid_width: usize,
    grid_height: usize,
    max_epochs: usize,
    frames_dir: &str,
    save_frames: bool,
    frame_interval: usize,
    neighbor_range: usize,
    auto_terminate_dead_epochs: usize,
    seed: u64,
) -> bool {
    // Returns true if completed normally, false if terminated early (all dead)
    use std::time::Instant;
    
    // Initialize with random data
    gpu_sim.init_random();
    
    println!("Running {} epochs...\n", max_epochs);
    
    let mut total_ops = 0u64;
    let start_time = Instant::now();
    let mut last_report = Instant::now();
    let mut consecutive_dead_epochs = 0usize;
    let mut final_epoch = max_epochs;
    // Only check every 100 epochs to minimize GPU read overhead
    let check_interval = if auto_terminate_dead_epochs > 0 { 100 } else { usize::MAX };
    
    for epoch in 0..max_epochs {
        // Regenerate pairs for this epoch (dynamic pairing for more variety)
        let pairs = generate_2d_pairs_seeded(
            num_programs, grid_width, grid_height, neighbor_range, seed, epoch
        );
        gpu_sim.set_pairs(&pairs);
        
        // Update dynamic energy sources if enabled
        if let Some(ref mut config) = energy_config {
            if config.is_dynamic() && config.update_sources(epoch) {
                gpu_sim.update_energy_config(config);
            }
        }
        
        let ops = gpu_sim.run_epoch();
        total_ops += ops;
        
        // Check for early termination (all dead) - only check periodically to avoid GPU overhead
        if auto_terminate_dead_epochs > 0 && energy_config.is_some() && epoch > 0 && epoch % check_interval == 0 {
            if gpu_sim.is_all_dead() {
                consecutive_dead_epochs += check_interval;
                if consecutive_dead_epochs >= auto_terminate_dead_epochs {
                    println!("\n*** All programs dead for {} epochs - terminating early ***", consecutive_dead_epochs);
                    final_epoch = epoch + 1;
                    break;
                }
            } else {
                consecutive_dead_epochs = 0;
            }
        }
        
        // Save frames (requires reading soup back from GPU)
        if save_frames && frame_interval > 0 && epoch % frame_interval == 0 {
            let soup = gpu_sim.get_soup();
            if let Err(e) = save_ppm_frame(&soup, grid_width, grid_height, frames_dir, epoch) {
                eprintln!("Warning: Could not save frame {}: {}", epoch, e);
            }
        }
        
        // Report progress every second
        if last_report.elapsed().as_secs() >= 1 || epoch == max_epochs - 1 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let mops = total_ops as f64 / elapsed / 1_000_000.0;
            println!(
                "Epoch {}/{} | {:.2}M ops/sec | Total: {:.2}G ops",
                epoch + 1,
                max_epochs,
                mops,
                total_ops as f64 / 1e9
            );
            last_report = Instant::now();
        }
    }
    
    // Save final frame
    if save_frames && frame_interval > 0 {
        let soup = gpu_sim.get_soup();
        if let Err(e) = save_ppm_frame(&soup, grid_width, grid_height, frames_dir, final_epoch) {
            eprintln!("Warning: Could not save final frame: {}", e);
        }
    }
    
    let elapsed = start_time.elapsed().as_secs_f64();
    let completed_normally = final_epoch == max_epochs;
    
    if completed_normally {
        println!("\nSimulation complete!");
    } else {
        println!("\nSimulation terminated early at epoch {}!", final_epoch);
    }
    println!("  Total time: {:.2}s", elapsed);
    println!("  Total ops: {:.2}G", total_ops as f64 / 1e9);
    println!("  Throughput: {:.2}M ops/sec", total_ops as f64 / elapsed / 1e6);
    
    completed_normally
}

/// Generate pairs for 2D grid topology (non-seeded version for backwards compatibility)
#[allow(dead_code)]
fn generate_2d_pairs(
    num_programs: usize,
    width: usize,
    height: usize,
    neighbor_range: usize,
) -> Vec<(u32, u32)> {
    generate_2d_pairs_seeded(num_programs, width, height, neighbor_range, 0, 0)
}

/// Generate pairs for 2D grid topology with deterministic seeding
/// 
/// Each epoch gets different pairings based on (base_seed, epoch).
/// The algorithm:
/// 1. Shuffle the order in which programs are processed (seeded by epoch)
/// 2. For each program, pick a random available neighbor (seeded)
/// 3. This ensures different pairings each epoch while being reproducible
#[allow(dead_code)]
fn generate_2d_pairs_seeded(
    num_programs: usize,
    width: usize,
    height: usize,
    neighbor_range: usize,
    base_seed: u64,
    epoch: usize,
) -> Vec<(u32, u32)> {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    
    // Create seeded RNG for reproducibility
    let seed = base_seed.wrapping_add(epoch as u64).wrapping_mul(0x9E3779B97F4A7C15);
    let mut rng = StdRng::seed_from_u64(seed);
    
    let mut pairs = Vec::with_capacity(num_programs / 2);
    let mut available = vec![true; num_programs];
    
    // Shuffle the order in which we process programs
    // This is key to getting different pairings each epoch
    let mut order: Vec<usize> = (0..num_programs).collect();
    for i in (1..order.len()).rev() {
        let j = rng.random_range(0..=i);
        order.swap(i, j);
    }
    
    for &i in &order {
        if !available[i] {
            continue;
        }
        
        let x = i % width;
        let y = i / width;
        
        // Find available neighbors
        let mut neighbors = Vec::new();
        for dx in -(neighbor_range as i32)..=(neighbor_range as i32) {
            for dy in -(neighbor_range as i32)..=(neighbor_range as i32) {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = (x as i32 + dx).rem_euclid(width as i32) as usize;
                let ny = (y as i32 + dy).rem_euclid(height as i32) as usize;
                let neighbor_idx = ny * width + nx;
                if available[neighbor_idx] {
                    neighbors.push(neighbor_idx);
                }
            }
        }
        
        if !neighbors.is_empty() {
            let partner = neighbors[rng.random_range(0..neighbors.len())];
            pairs.push((i as u32, partner as u32));
            available[i] = false;
            available[partner] = false;
        }
    }
    
    pairs
}

/// Generate all pairs for mega-simulation mode with absolute indices (non-seeded, for backwards compatibility).
#[cfg(any(feature = "wgpu-compute", feature = "cuda"))]
#[allow(dead_code)]
fn generate_mega_pairs(
    num_programs: usize,
    grid_width: usize,
    grid_height: usize,
    neighbor_range: usize,
    num_sims: usize,
    layout_cols: usize,
    layout_rows: usize,
) -> Vec<(u32, u32)> {
    generate_mega_pairs_seeded(
        num_programs, grid_width, grid_height, neighbor_range,
        num_sims, layout_cols, layout_rows, 0, 0
    )
}

/// Generate all pairs for mega-simulation mode with absolute indices and deterministic seeding.
/// Includes both internal pairs (within each sim) and cross-border pairs (between adjacent sims).
/// 
/// Each epoch gets different pairings based on (base_seed, epoch).
/// Cross-border pairs are randomized to ensure different edge programs interact each epoch.
#[cfg(any(feature = "wgpu-compute", feature = "cuda"))]
fn generate_mega_pairs_seeded(
    num_programs: usize,
    grid_width: usize,
    grid_height: usize,
    neighbor_range: usize,
    num_sims: usize,
    layout_cols: usize,
    layout_rows: usize,
    base_seed: u64,
    epoch: usize,
) -> Vec<(u32, u32)> {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    
    // Create seeded RNG for reproducibility
    let seed = base_seed.wrapping_add(epoch as u64).wrapping_mul(0x9E3779B97F4A7C15);
    let mut rng = StdRng::seed_from_u64(seed);
    
    let mut all_pairs = Vec::new();
    
    // Track which programs are already paired (globally)
    let total_programs = num_programs * num_sims;
    let mut available = vec![true; total_programs];
    
    // Collect all potential cross-border edges, then shuffle them
    let mut cross_border_candidates: Vec<(usize, usize)> = Vec::new();
    
    for sim_row in 0..layout_rows {
        for sim_col in 0..layout_cols {
            let sim_idx = sim_row * layout_cols + sim_col;
            let sim_offset = sim_idx * num_programs;
            
            // Right neighbor
            if sim_col + 1 < layout_cols {
                let neighbor_sim = sim_row * layout_cols + (sim_col + 1);
                let neighbor_offset = neighbor_sim * num_programs;
                
                // Collect all right-left edge pairs
                for y in 0..grid_height {
                    let local_p1 = y * grid_width + (grid_width - 1);
                    let global_p1 = sim_offset + local_p1;
                    let local_p2 = y * grid_width + 0;
                    let global_p2 = neighbor_offset + local_p2;
                    cross_border_candidates.push((global_p1, global_p2));
                }
            }
            
            // Bottom neighbor  
            if sim_row + 1 < layout_rows {
                let neighbor_sim = (sim_row + 1) * layout_cols + sim_col;
                let neighbor_offset = neighbor_sim * num_programs;
                
                // Collect all bottom-top edge pairs
                for x in 0..grid_width {
                    let local_p1 = (grid_height - 1) * grid_width + x;
                    let global_p1 = sim_offset + local_p1;
                    let local_p2 = 0 * grid_width + x;
                    let global_p2 = neighbor_offset + local_p2;
                    cross_border_candidates.push((global_p1, global_p2));
                }
            }
        }
    }
    
    // Shuffle cross-border candidates so different edges get paired each epoch
    for i in (1..cross_border_candidates.len()).rev() {
        let j = rng.random_range(0..=i);
        cross_border_candidates.swap(i, j);
    }
    
    // FIRST: Process shuffled cross-border pairs
    for (global_p1, global_p2) in cross_border_candidates {
        if available[global_p1] && available[global_p2] {
            all_pairs.push((global_p1 as u32, global_p2 as u32));
            available[global_p1] = false;
            available[global_p2] = false;
        }
    }
    
    // SECOND: Generate internal pairs for each simulation with shuffled order
    for sim_idx in 0..num_sims {
        let sim_offset = sim_idx * num_programs;
        
        // Shuffle the order in which we process programs within this sim
        let mut order: Vec<usize> = (0..num_programs).collect();
        for i in (1..order.len()).rev() {
            let j = rng.random_range(0..=i);
            order.swap(i, j);
        }
        
        for &local_i in &order {
            let global_i = sim_offset + local_i;
            if !available[global_i] {
                continue;
            }
            
            let x = local_i % grid_width;
            let y = local_i / grid_width;
            
            // Find available neighbors within this simulation
            // NOTE: No toroidal wrapping in mega mode - edge cells connect via cross-border pairs only
            let mut neighbors = Vec::new();
            for dx in -(neighbor_range as i32)..=(neighbor_range as i32) {
                for dy in -(neighbor_range as i32)..=(neighbor_range as i32) {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    // Skip if outside simulation bounds (no toroidal wrap)
                    if nx < 0 || nx >= grid_width as i32 || ny < 0 || ny >= grid_height as i32 {
                        continue;
                    }
                    let local_neighbor = (ny as usize) * grid_width + (nx as usize);
                    let global_neighbor = sim_offset + local_neighbor;
                    if available[global_neighbor] {
                        neighbors.push(global_neighbor);
                    }
                }
            }
            
            if !neighbors.is_empty() {
                let partner = neighbors[rng.random_range(0..neighbors.len())];
                all_pairs.push((global_i as u32, partner as u32));
                available[global_i] = false;
                available[partner] = false;
            }
        }
    }
    
    all_pairs
}

/// Save a checkpoint
#[cfg(feature = "wgpu-compute")]
fn save_checkpoint(
    multi_sim: &gpu::wgpu_sim::MultiWgpuSimulation,
    epoch: usize,
    grid_width: usize,
    grid_height: usize,
    num_sims: usize,
    parallel_layout: [usize; 2],
    border_interaction: bool,
    seed: u64,
    checkpoint_path: &str,
) {
    let soup = multi_sim.get_all_soup();
    let energy_states = multi_sim.get_all_energy_states();
    save_checkpoint_from_data(
        epoch,
        grid_width,
        grid_height,
        num_sims,
        parallel_layout,
        border_interaction,
        seed,
        soup,
        energy_states,
        checkpoint_path,
    );
}

#[cfg(any(feature = "wgpu-compute", feature = "cuda"))]
fn save_checkpoint_from_data(
    epoch: usize,
    grid_width: usize,
    grid_height: usize,
    num_sims: usize,
    parallel_layout: [usize; 2],
    border_interaction: bool,
    seed: u64,
    soup: Vec<u8>,
    energy_states: Vec<u32>,
    checkpoint_path: &str,
) {
    let ckpt = checkpoint::Checkpoint::new(
        epoch,
        grid_width,
        grid_height,
        num_sims,
        parallel_layout,
        border_interaction,
        seed,
        soup,
        energy_states,
    );
    
    let filename = checkpoint::checkpoint_filename(checkpoint_path, epoch, num_sims);
    match ckpt.save(&filename) {
        Ok(_) => println!("\n  ✓ Checkpoint saved: {}", filename),
        Err(e) => eprintln!("\n  ✗ Checkpoint failed: {}", e),
    }
}

/// Save a combined mega-frame showing all simulations in their layout
/// With scaling support to reduce file size for large grids
#[cfg(feature = "wgpu-compute")]
fn save_mega_frame(
    multi_sim: &gpu::wgpu_sim::MultiWgpuSimulation,
    layout_cols: usize,
    layout_rows: usize,
    grid_width: usize,
    grid_height: usize,
    frames_dir: &str,
    epoch: usize,
    scale: usize,
) -> std::io::Result<()> {
    let _ = fs::create_dir_all(frames_dir);
    
    let scale = scale.max(1);
    let sub_img_width = grid_width * 8;
    let sub_img_height = grid_height * 8;
    let full_width = sub_img_width * layout_cols;
    let full_height = sub_img_height * layout_rows;
    
    // Output dimensions after scaling
    let out_width = full_width / scale;
    let out_height = full_height / scale;
    
    // For very large images, use even more aggressive scaling
    let effective_scale = if out_width * out_height > 16_000_000 {
        // More than 16 megapixels - double the scale
        scale * 2
    } else {
        scale
    };
    let out_width = full_width / effective_scale;
    let out_height = full_height / effective_scale;
    
    let mut mega_img = vec![0u8; out_width * out_height * 3];
    let byte_colors = init_byte_colors();
    
    // Pre-calculate scaled sub-image dimensions
    let scaled_sub_width = sub_img_width / effective_scale;
    let scaled_sub_height = sub_img_height / effective_scale;
    
    for sim_row in 0..layout_rows {
        for sim_col in 0..layout_cols {
            let sim_idx = sim_row * layout_cols + sim_col;
            let soup = multi_sim.get_soup(sim_idx);
            
            let offset_x = sim_col * scaled_sub_width;
            let offset_y = sim_row * scaled_sub_height;
            
            // Render this simulation with scaling
            for out_y in 0..scaled_sub_height {
                for out_x in 0..scaled_sub_width {
                    // Sample from center of scale block
                    let src_x = out_x * effective_scale + effective_scale / 2;
                    let src_y = out_y * effective_scale + effective_scale / 2;
                    
                    // Find which program and byte this corresponds to
                    let prog_x = src_x / 8;
                    let prog_y = src_y / 8;
                    let byte_x = src_x % 8;
                    let byte_y = src_y % 8;
                    
                    if prog_x < grid_width && prog_y < grid_height {
                        let prog_idx = prog_y * grid_width + prog_x;
                        let byte_idx = byte_y * 8 + byte_x;
                        let byte_val = soup[prog_idx * 64 + byte_idx];
                        let color = byte_colors[byte_val as usize];
                        
                        let pixel_x = offset_x + out_x;
                        let pixel_y = offset_y + out_y;
                        if pixel_x < out_width && pixel_y < out_height {
                            let img_idx = (pixel_y * out_width + pixel_x) * 3;
                            mega_img[img_idx] = color[0];
                            mega_img[img_idx + 1] = color[1];
                            mega_img[img_idx + 2] = color[2];
                        }
                    }
                }
            }
        }
    }
    
    // Save as PNG with maximum compression for large images
    use std::io::BufWriter;
    let filename = format!("{}/mega_epoch_{:08}.png", frames_dir, epoch);
    let file = File::create(&filename)?;
    let w = BufWriter::new(file);
    
    let mut encoder = png::Encoder::new(w, out_width as u32, out_height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    // Use best compression for large images
    encoder.set_compression(png::Compression::Best);
    encoder.set_filter(png::FilterType::Avg);
    
    let mut writer = encoder.write_header()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    writer.write_image_data(&mega_img)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    
    Ok(())
}

/// Save a combined mega-frame from pre-fetched soup data
/// Used by async readback path to avoid re-fetching from GPU
fn save_mega_frame_from_data(
    all_soup: &[u8],
    layout_cols: usize,
    layout_rows: usize,
    grid_width: usize,
    grid_height: usize,
    frames_dir: &str,
    epoch: usize,
    scale: usize,
    num_sims: usize,
) -> std::io::Result<()> {
    let _ = fs::create_dir_all(frames_dir);
    
    let scale = scale.max(1);
    let sub_img_width = grid_width * 8;
    let sub_img_height = grid_height * 8;
    let full_width = sub_img_width * layout_cols;
    let full_height = sub_img_height * layout_rows;
    let sim_size = grid_width * grid_height * SINGLE_TAPE_SIZE;
    
    // Output dimensions after scaling
    let out_width = full_width / scale;
    let out_height = full_height / scale;
    
    // For very large images, use even more aggressive scaling
    let effective_scale = if out_width * out_height > 16_000_000 {
        scale * 2
    } else {
        scale
    };
    let out_width = full_width / effective_scale;
    let out_height = full_height / effective_scale;
    
    let mut mega_img = vec![0u8; out_width * out_height * 3];
    let byte_colors = init_byte_colors();
    
    let scaled_sub_width = sub_img_width / effective_scale;
    let scaled_sub_height = sub_img_height / effective_scale;
    
    for sim_row in 0..layout_rows {
        for sim_col in 0..layout_cols {
            let sim_idx = sim_row * layout_cols + sim_col;
            if sim_idx >= num_sims {
                continue;
            }
            
            // Get this sim's soup from the combined data
            let soup_start = sim_idx * sim_size;
            let soup_end = soup_start + sim_size;
            let soup = &all_soup[soup_start..soup_end];
            
            let offset_x = sim_col * scaled_sub_width;
            let offset_y = sim_row * scaled_sub_height;
            
            for out_y in 0..scaled_sub_height {
                for out_x in 0..scaled_sub_width {
                    let src_x = out_x * effective_scale + effective_scale / 2;
                    let src_y = out_y * effective_scale + effective_scale / 2;
                    
                    let prog_x = src_x / 8;
                    let prog_y = src_y / 8;
                    let byte_x = src_x % 8;
                    let byte_y = src_y % 8;
                    
                    if prog_x < grid_width && prog_y < grid_height {
                        let prog_idx = prog_y * grid_width + prog_x;
                        let byte_idx = byte_y * 8 + byte_x;
                        let byte_val = soup[prog_idx * 64 + byte_idx];
                        let color = byte_colors[byte_val as usize];
                        
                        let pixel_x = offset_x + out_x;
                        let pixel_y = offset_y + out_y;
                        if pixel_x < out_width && pixel_y < out_height {
                            let img_idx = (pixel_y * out_width + pixel_x) * 3;
                            mega_img[img_idx] = color[0];
                            mega_img[img_idx + 1] = color[1];
                            mega_img[img_idx + 2] = color[2];
                        }
                    }
                }
            }
        }
    }
    
    use std::io::BufWriter;
    let filename = format!("{}/mega_epoch_{:08}.png", frames_dir, epoch);
    let file = File::create(&filename)?;
    let w = BufWriter::new(file);
    
    let mut encoder = png::Encoder::new(w, out_width as u32, out_height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Best);
    encoder.set_filter(png::FilterType::Avg);
    
    let mut writer = encoder.write_header()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    writer.write_image_data(&mega_img)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    
    Ok(())
}

/// Save per-sim frames (and optional mega frame) from combined soup data
fn save_frames_from_data(
    all_soup: &[u8],
    num_sims: usize,
    grid_width: usize,
    grid_height: usize,
    frames_dir: &str,
    epoch: usize,
    frame_format: &str,
    thumbnail_scale: usize,
    mega_mode: bool,
    parallel_layout: [usize; 2],
) {
    let _ = fs::create_dir_all(frames_dir);
    if mega_mode {
        let [layout_cols, layout_rows] = parallel_layout;
        if let Err(e) = save_mega_frame_from_data(
            all_soup,
            layout_cols,
            layout_rows,
            grid_width,
            grid_height,
            frames_dir,
            epoch,
            thumbnail_scale,
            num_sims,
        ) {
            eprintln!("Warning: Could not save mega frame {}: {}", epoch, e);
        }
    }

    let sim_size = grid_width * grid_height * 64;
    for sim_idx in 0..num_sims {
        let start = sim_idx * sim_size;
        let end = start + sim_size;
        let soup = &all_soup[start..end];
        let sim_frames_dir = format!("{}/sim_{}", frames_dir, sim_idx);
        let _ = fs::create_dir_all(&sim_frames_dir);
        if let Err(e) = save_frame(
            soup,
            grid_width,
            grid_height,
            &sim_frames_dir,
            epoch,
            frame_format,
            thumbnail_scale,
        ) {
            eprintln!("Warning: Could not save frame {} for sim {}: {}", epoch, sim_idx, e);
        }
    }
}

/// Format a Unix timestamp as a human-readable string
#[allow(dead_code)]
fn chrono_time_str(timestamp: u64) -> String {
    use std::time::{Duration, UNIX_EPOCH};
    let datetime = UNIX_EPOCH + Duration::from_secs(timestamp);
    // Simple formatting without external crate
    format!("{:?}", datetime)
}

/// Save a frame from soup data (supports PNG and PPM, with optional downscaling)
fn save_frame(
    soup: &[u8],
    grid_width: usize,
    grid_height: usize,
    frames_dir: &str,
    epoch: usize,
    format: &str,
    scale: usize,
) -> std::io::Result<()> {
    let num_programs = grid_width * grid_height;
    let full_width = grid_width * 8;
    let full_height = grid_height * 8;
    
    // Generate byte colors (BFF command highlighting)
    let byte_colors = init_byte_colors();
    
    // Determine output dimensions based on scale
    let scale = scale.max(1);
    let out_width = full_width / scale;
    let out_height = full_height / scale;
    
    let mut img_data = vec![0u8; out_width * out_height * 3];
    
    if scale == 1 {
        // Full resolution - direct render
        for i in 0..num_programs {
            let grid_x = i % grid_width;
            let grid_y = i / grid_width;
            let program_start = i * SINGLE_TAPE_SIZE;
            
            for j in 0..SINGLE_TAPE_SIZE {
                let pixel_x = grid_x * 8 + (j % 8);
                let pixel_y = grid_y * 8 + (j / 8);
                let img_idx = (pixel_y * full_width + pixel_x) * 3;
                
                let byte_val = soup[program_start + j];
                let color = byte_colors[byte_val as usize];
                
                img_data[img_idx] = color[0];
                img_data[img_idx + 1] = color[1];
                img_data[img_idx + 2] = color[2];
            }
        }
    } else {
        // Downscaled - sample or average
        for out_y in 0..out_height {
            for out_x in 0..out_width {
                // Sample center of the scale block
                let src_x = out_x * scale + scale / 2;
                let src_y = out_y * scale + scale / 2;
                
                // Find which program and byte this corresponds to
                let prog_x = src_x / 8;
                let prog_y = src_y / 8;
                let byte_x = src_x % 8;
                let byte_y = src_y % 8;
                
                if prog_x < grid_width && prog_y < grid_height {
                    let prog_idx = prog_y * grid_width + prog_x;
                    let byte_idx = byte_y * 8 + byte_x;
                    let byte_val = soup[prog_idx * SINGLE_TAPE_SIZE + byte_idx];
                    let color = byte_colors[byte_val as usize];
                    
                    let out_idx = (out_y * out_width + out_x) * 3;
                    img_data[out_idx] = color[0];
                    img_data[out_idx + 1] = color[1];
                    img_data[out_idx + 2] = color[2];
                }
            }
        }
    }
    
    // Save in requested format
    if format == "png" {
        save_png(&img_data, out_width, out_height, frames_dir, epoch)
    } else {
        save_ppm(&img_data, out_width, out_height, frames_dir, epoch)
    }
}

/// Save image data as PNG (compressed)
fn save_png(
    img_data: &[u8],
    width: usize,
    height: usize,
    frames_dir: &str,
    epoch: usize,
) -> std::io::Result<()> {
    use std::io::BufWriter;
    
    let path = Path::new(frames_dir).join(format!("{:08}.png", epoch));
    let file = File::create(&path)?;
    let w = BufWriter::new(file);
    
    let mut encoder = png::Encoder::new(w, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Fast); // Fast compression, still good ratio
    
    let mut writer = encoder.write_header()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    writer.write_image_data(img_data)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    
    Ok(())
}

/// Save image data as PPM (uncompressed)
fn save_ppm(
    img_data: &[u8],
    width: usize,
    height: usize,
    frames_dir: &str,
    epoch: usize,
) -> std::io::Result<()> {
    let path = Path::new(frames_dir).join(format!("{:08}.ppm", epoch));
    let mut file = File::create(&path)?;
    writeln!(file, "P6")?;
    writeln!(file, "{} {}", width, height)?;
    writeln!(file, "255")?;
    file.write_all(img_data)?;
    Ok(())
}

/// Legacy function for backwards compatibility
#[allow(dead_code)]
fn save_ppm_frame(
    soup: &[u8],
    grid_width: usize,
    grid_height: usize,
    frames_dir: &str,
    epoch: usize,
) -> std::io::Result<()> {
    save_frame(soup, grid_width, grid_height, frames_dir, epoch, "ppm", 1)
}

/// Initialize byte colors for visualization
fn init_byte_colors() -> [[u8; 3]; 256] {
    let mut colors = [[0u8; 3]; 256];
    
    for i in 0..256 {
        // Default: grayscale based on byte value
        let gray = i as u8;
        colors[i] = [gray, gray, gray];
    }
    
    // Null byte (dead/empty) - bright red for visibility
    colors[0] = [255, 0, 0];
    
    // BFF commands get distinct colors
    colors[b'+' as usize] = [255, 100, 100]; // Red
    colors[b'-' as usize] = [100, 100, 255]; // Blue
    colors[b'>' as usize] = [100, 255, 100]; // Green
    colors[b'<' as usize] = [255, 255, 100]; // Yellow
    colors[b'{' as usize] = [100, 255, 255]; // Cyan
    colors[b'}' as usize] = [255, 100, 255]; // Magenta
    colors[b'[' as usize] = [255, 200, 100]; // Orange
    colors[b']' as usize] = [200, 100, 255]; // Purple
    colors[b'.' as usize] = [255, 255, 255]; // White
    colors[b',' as usize] = [200, 200, 200]; // Light gray
    
    colors
}

// ============================================================================
// ASYNC WRITER
// ============================================================================

/// Bundle of outputs to save asynchronously
#[allow(dead_code)]
struct SaveBundle {
    data: Vec<u8>,
    epoch: usize,
    raw_dir: Option<String>,
    frames_dir: Option<String>,
    grid_width: usize,
    grid_height: usize,
    num_sims: usize,
    layout: [usize; 2],
    frame_format: String,
    thumbnail_scale: usize,
    mega_mode: bool,
}

/// Message type for async save operations
#[allow(dead_code)]
enum SaveMessage {
    /// Save raw and/or rendered frames from soup data
    Bundle(SaveBundle),
    /// Shutdown the writer thread
    Shutdown,
}

/// Async writer handle - sends save operations to a background thread
#[allow(dead_code)]
struct AsyncWriter {
    sender: Sender<SaveMessage>,
    handle: Option<JoinHandle<()>>,
}

#[allow(dead_code)]
impl AsyncWriter {
    /// Create a new async writer with a background thread
    fn new() -> Self {
        let (sender, receiver) = mpsc::channel::<SaveMessage>();
        
        let handle = thread::spawn(move || {
            while let Ok(msg) = receiver.recv() {
                match msg {
                    SaveMessage::Bundle(bundle) => {
                        if let Some(path) = bundle.raw_dir.as_deref() {
                            if let Err(e) = save_raw_data_sync(
                                &bundle.data,
                                bundle.epoch,
                                path,
                                bundle.grid_width,
                                bundle.grid_height,
                                bundle.num_sims,
                                bundle.layout,
                            ) {
                                eprintln!("Async save error (epoch {}): {}", bundle.epoch, e);
                            }
                        }
                        if let Some(frames_dir) = bundle.frames_dir.as_deref() {
                            save_frames_from_data(
                                &bundle.data,
                                bundle.num_sims,
                                bundle.grid_width,
                                bundle.grid_height,
                                frames_dir,
                                bundle.epoch,
                                &bundle.frame_format,
                                bundle.thumbnail_scale,
                                bundle.mega_mode,
                                bundle.layout,
                            );
                        }
                    }
                    SaveMessage::Shutdown => break,
                }
            }
        });
        
        Self {
            sender,
            handle: Some(handle),
        }
    }
    
    /// Queue raw and/or frame saves (non-blocking)
    fn save_bundle(
        &self,
        data: Vec<u8>,
        epoch: usize,
        raw_dir: Option<&str>,
        frames_dir: Option<&str>,
        grid_width: usize,
        grid_height: usize,
        num_sims: usize,
        layout: [usize; 2],
        frame_format: &str,
        thumbnail_scale: usize,
        mega_mode: bool,
    ) {
        let msg = SaveMessage::Bundle(SaveBundle {
            data,
            epoch,
            raw_dir: raw_dir.map(|s| s.to_string()),
            frames_dir: frames_dir.map(|s| s.to_string()),
            grid_width,
            grid_height,
            num_sims,
            layout,
            frame_format: frame_format.to_string(),
            thumbnail_scale,
            mega_mode,
        });
        if self.sender.send(msg).is_err() {
            eprintln!("Warning: async save queue full or closed");
        }
    }
    
    /// Wait for all pending saves to complete and shutdown
    fn shutdown(&mut self) {
        let _ = self.sender.send(SaveMessage::Shutdown);
        if let Some(handle) = self.handle.take() {
            if let Err(e) = handle.join() {
                eprintln!("Warning: AsyncWriter thread panicked: {:?}", e);
            }
        }
    }
    
    /// Get the number of pending saves in the queue (approximate)
    /// Note: mpsc doesn't expose queue length, so this just checks if sender is connected
    fn is_connected(&self) -> bool {
        // Try sending a dummy check - if it fails, the receiver is gone
        // Actually, we can't do this without consuming a message, so just return true
        // if the handle exists
        self.handle.is_some()
    }
}

/// Ensure AsyncWriter flushes pending saves when dropped
impl Drop for AsyncWriter {
    fn drop(&mut self) {
        // Send shutdown signal and wait for thread to complete
        // This ensures all queued saves are processed before program exit
        if self.handle.is_some() {
            let _ = self.sender.send(SaveMessage::Shutdown);
            if let Some(handle) = self.handle.take() {
                // Give the thread a chance to complete, but don't block forever
                // In practice, join() will complete once all queued items are processed
                if let Err(e) = handle.join() {
                    eprintln!("Warning: AsyncWriter thread panicked during drop: {:?}", e);
                }
            }
        }
    }
}

/// Save raw soup data to disk (synchronous version) with Zstd compression
#[allow(dead_code)]
fn save_raw_data_sync(
    data: &[u8],
    epoch: usize,
    raw_dir: &str,
    grid_width: usize,
    grid_height: usize,
    num_sims: usize,
    layout: [usize; 2],
) -> std::io::Result<()> {
    fs::create_dir_all(raw_dir)?;
    
    // File format: raw_epoch_NNNNNNNN.bin
    // Header: 36 bytes (magic + metadata + compressed_size), then zstd compressed soup
    let path = Path::new(raw_dir).join(format!("raw_epoch_{:08}.bin", epoch));
    let file = File::create(&path)?;
    let mut writer = BufWriter::new(file);
    
    // Magic number: "BFF2" (BFF Raw v2 - compressed)
    writer.write_all(b"BFF2")?;
    
    // Version (u32)
    writer.write_all(&2u32.to_le_bytes())?;
    
    // Metadata
    writer.write_all(&(epoch as u32).to_le_bytes())?;
    writer.write_all(&(grid_width as u32).to_le_bytes())?;
    writer.write_all(&(grid_height as u32).to_le_bytes())?;
    writer.write_all(&(num_sims as u32).to_le_bytes())?;
    writer.write_all(&(layout[0] as u32).to_le_bytes())?;
    writer.write_all(&(layout[1] as u32).to_le_bytes())?;
    
    // Compress soup data with zstd level 1 (fast)
    let compressed = zstd::encode_all(data, 1)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    
    // Write compressed size and data
    writer.write_all(&(compressed.len() as u32).to_le_bytes())?;
    writer.write_all(&compressed)?;
    
    Ok(())
}

/// Header for raw data files
#[derive(Debug)]
struct RawDataHeader {
    epoch: usize,
    grid_width: usize,
    grid_height: usize,
    num_sims: usize,
    layout: [usize; 2],
}

/// Load raw data file (header + soup data) - supports both v1 BFFR and v2 BFF2 formats
fn load_raw_data(path: &Path) -> std::io::Result<(RawDataHeader, Vec<u8>)> {
    use std::io::Read;
    let mut file = File::open(path)?;
    
    // Read header (32 bytes)
    let mut header_buf = [0u8; 32];
    file.read_exact(&mut header_buf)?;
    
    // Parse magic and determine format
    let magic = &header_buf[0..4];
    let is_compressed = magic == b"BFF2";
    
    if magic != b"BFFR" && magic != b"BFF2" {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid raw data magic"));
    }
    
    let epoch = u32::from_le_bytes(header_buf[8..12].try_into().unwrap()) as usize;
    let grid_width = u32::from_le_bytes(header_buf[12..16].try_into().unwrap()) as usize;
    let grid_height = u32::from_le_bytes(header_buf[16..20].try_into().unwrap()) as usize;
    let num_sims = u32::from_le_bytes(header_buf[20..24].try_into().unwrap()) as usize;
    let layout_cols = u32::from_le_bytes(header_buf[24..28].try_into().unwrap()) as usize;
    let layout_rows = u32::from_le_bytes(header_buf[28..32].try_into().unwrap()) as usize;
    
    let header = RawDataHeader {
        epoch,
        grid_width,
        grid_height,
        num_sims,
        layout: [layout_cols, layout_rows],
    };
    
    // Read soup data based on format
    let soup_data = if is_compressed {
        // BFF2: Read compressed size, then decompress
        let mut size_buf = [0u8; 4];
        file.read_exact(&mut size_buf)?;
        let compressed_size = u32::from_le_bytes(size_buf) as usize;
        
        let mut compressed = vec![0u8; compressed_size];
        file.read_exact(&mut compressed)?;
        
        zstd::decode_all(&compressed[..])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?
    } else {
        // BFFR: Read uncompressed data
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        data
    };
    
    Ok((header, soup_data))
}

/// Render frames from raw data directory
fn render_raw_directory(
    raw_dir: &str,
    output_dir: &str,
    frame_format: &str,
    thumbnail_scale: usize,
) -> std::io::Result<()> {
    use std::time::Instant;
    
    println!("Rendering frames from raw data: {}", raw_dir);
    println!("Output directory: {}", output_dir);
    println!("Format: {}, Scale: 1/{}", frame_format, thumbnail_scale);
    
    fs::create_dir_all(output_dir)?;
    
    // Find all raw data files
    let mut files: Vec<_> = fs::read_dir(raw_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "bin"))
        .filter(|e| e.file_name().to_string_lossy().starts_with("raw_epoch_"))
        .collect();
    
    files.sort_by_key(|e| e.file_name());
    
    println!("Found {} raw data files", files.len());
    
    let start = Instant::now();
    let mut rendered = 0;
    
    for entry in &files {
        let path = entry.path();
        match load_raw_data(&path) {
            Ok((header, soup_data)) => {
                // For mega-sims, render combined frame
                if header.num_sims > 1 && header.layout[0] > 1 && header.layout[1] > 1 {
                    // Render mega frame
                    let colors = init_byte_colors();
                    let mega_width = header.grid_width * header.layout[0];
                    let mega_height = header.grid_height * header.layout[1];
                    let programs_per_sim = header.grid_width * header.grid_height;
                    
                    // Build combined image
                    let mut mega_img = vec![0u8; mega_width * mega_height * 3];
                    
                    for sim_row in 0..header.layout[1] {
                        for sim_col in 0..header.layout[0] {
                            let sim_idx = sim_row * header.layout[0] + sim_col;
                            let sim_offset = sim_idx * programs_per_sim * SINGLE_TAPE_SIZE;
                            
                            for y in 0..header.grid_height {
                                for x in 0..header.grid_width {
                                    let local_idx = y * header.grid_width + x;
                                    let byte = soup_data.get(sim_offset + local_idx * SINGLE_TAPE_SIZE).copied().unwrap_or(0);
                                    let color = colors[byte as usize];
                                    
                                    let mega_x = sim_col * header.grid_width + x;
                                    let mega_y = sim_row * header.grid_height + y;
                                    let mega_idx = (mega_y * mega_width + mega_x) * 3;
                                    
                                    mega_img[mega_idx] = color[0];
                                    mega_img[mega_idx + 1] = color[1];
                                    mega_img[mega_idx + 2] = color[2];
                                }
                            }
                        }
                    }
                    
                    // Apply scaling
                    let (final_img, final_width, final_height) = if thumbnail_scale > 1 {
                        let new_w = mega_width / thumbnail_scale;
                        let new_h = mega_height / thumbnail_scale;
                        let mut scaled = vec![0u8; new_w * new_h * 3];
                        
                        for y in 0..new_h {
                            for x in 0..new_w {
                                let src_x = x * thumbnail_scale;
                                let src_y = y * thumbnail_scale;
                                let src_idx = (src_y * mega_width + src_x) * 3;
                                let dst_idx = (y * new_w + x) * 3;
                                scaled[dst_idx..dst_idx+3].copy_from_slice(&mega_img[src_idx..src_idx+3]);
                            }
                        }
                        (scaled, new_w, new_h)
                    } else {
                        (mega_img, mega_width, mega_height)
                    };
                    
                    // Save frame
                    match frame_format {
                        "png" => save_png(&final_img, final_width, final_height, output_dir, header.epoch)?,
                        _ => save_ppm(&final_img, final_width, final_height, output_dir, header.epoch)?,
                    }
                } else {
                    // Single sim - use standard save_frame
                    save_frame(&soup_data, header.grid_width, header.grid_height, 
                        output_dir, header.epoch, frame_format, thumbnail_scale)?;
                }
                rendered += 1;
                
                if rendered % 10 == 0 {
                    print!("\rRendered {}/{} frames...", rendered, files.len());
                    std::io::stdout().flush()?;
                }
            }
            Err(e) => {
                eprintln!("Error loading {}: {}", path.display(), e);
            }
        }
    }
    
    let elapsed = start.elapsed();
    println!("\rRendered {} frames in {:.2}s ({:.1} fps)", 
        rendered, elapsed.as_secs_f64(), rendered as f64 / elapsed.as_secs_f64());
    
    Ok(())
}
