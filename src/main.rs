mod bff;
mod checkpoint;
mod energy;
mod fitness;
mod gpu;
mod islands;
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
    pub reserve_epochs: u8,
    pub death_epochs: u8,
    /// Spontaneous generation rate (1 in N chance per dead tape in energy zone per epoch, 0 = disabled)
    pub spontaneous_rate: u32,
    /// Shape of energy zones: "circle", "strip_h", "strip_v", "half_circle", "ellipse", "random"
    pub shape: String,
    /// Dynamic energy options
    pub dynamic: DynamicEnergySettings,
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
        }
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
    
    /// Generate a template config file
    pub fn write_template(path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let config = Config::default();
        config.to_yaml(path)
    }
    
    /// Convert mutation_rate (1/N) to internal mutation_prob
    pub fn mutation_prob(&self) -> u32 {
        ((1u64 << 30) / self.simulation.mutation_rate as u64) as u32
    }
}

/// Command-line arguments (internal, maps from Config)
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
    energy_reserve: u8,
    energy_death: u8,
    energy_spontaneous_rate: u32,
    energy_shape: String,
    // Dynamic energy options
    energy_random: bool,
    energy_max_sources: usize,
    energy_source_lifetime: usize,
    energy_spawn_rate: usize,
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
                    i -= 1; // didn't consume an arg
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
    
    // Try GPU first, fall back to CPU
    #[cfg(feature = "wgpu-compute")]
    {
        // Build energy config for GPU
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
        
        if args.parallel_sims > 1 {
            // Multi-simulation mode
            if let Some(mut multi_sim) = gpu::wgpu_sim::MultiWgpuSimulation::new(
                args.parallel_sims,
                num_programs,
                args.grid_width,
                args.grid_height,
                args.seed,
                args.mutation_prob,
                args.steps_per_run as u32,
                gpu_energy_config.as_ref(),
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
    energy_reserve: u8,
    energy_death: u8,
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
        println!("Frame rendering: ENABLED (format: {})", frame_format);
    } else if save_raw {
        println!("Frame rendering: DISABLED (will render later with --render-raw)");
    }
    println!();
    
    // Create async writer if needed
    let async_writer = if save_raw && async_save {
        Some(AsyncWriter::new())
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
            
            // Save raw data (async or sync I/O)
            if save_raw {
                if let Some(ref writer) = async_writer {
                    // Async save - clone data and send to background thread
                    writer.save_raw(
                        all_soup.clone(), epoch, raw_dir, 
                        grid_width, grid_height, num_sims, parallel_layout
                    );
                } else {
                    // Sync save
                    if let Err(e) = save_raw_data_sync(
                        &all_soup, epoch, raw_dir,
                        grid_width, grid_height, num_sims, parallel_layout
                    ) {
                        eprintln!("Warning: Could not save raw data for epoch {}: {}", epoch, e);
                    }
                }
            }
            
            // Save rendered frames
            if save_frames {
                // In mega-simulation mode, also save a combined frame
                if effective_border_interaction {
                    let _ = save_mega_frame_from_data(
                        &all_soup, layout_cols, layout_rows, grid_width, grid_height,
                        frames_dir, epoch, thumbnail_scale, num_sims
                    );
                }
                
                // Extract per-sim data from combined soup
                let sim_size = grid_width * grid_height * 64;
                for sim_idx in 0..num_sims {
                    let start = sim_idx * sim_size;
                    let end = start + sim_size;
                    let soup = &all_soup[start..end];
                    let sim_frames_dir = format!("{}/sim_{}", frames_dir, sim_idx);
                    let _ = fs::create_dir_all(&sim_frames_dir);
                    if let Err(e) = save_frame(soup, grid_width, grid_height, &sim_frames_dir, epoch, frame_format, thumbnail_scale) {
                        eprintln!("Warning: Could not save frame {} for sim {}: {}", epoch, sim_idx, e);
                    }
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
                writer.save_raw(
                    all_soup, max_epochs, raw_dir,
                    grid_width, grid_height, num_sims, parallel_layout
                );
            } else {
                if let Err(e) = save_raw_data_sync(
                    &all_soup, max_epochs, raw_dir,
                    grid_width, grid_height, num_sims, parallel_layout
                ) {
                    eprintln!("Warning: Could not save final raw data: {}", e);
                }
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
    if let Some(writer) = async_writer {
        println!("Waiting for async saves to complete...");
        writer.shutdown();
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
) -> bool {
    // Returns true if completed normally, false if terminated early (all dead)
    use std::time::Instant;
    
    // Initialize with random data
    gpu_sim.init_random();
    
    // Generate pair indices for 2D grid topology
    let pairs = generate_2d_pairs(num_programs, grid_width, grid_height, neighbor_range);
    gpu_sim.set_pairs(&pairs);
    
    println!("Running {} epochs...\n", max_epochs);
    
    let mut total_ops = 0u64;
    let start_time = Instant::now();
    let mut last_report = Instant::now();
    let mut consecutive_dead_epochs = 0usize;
    let mut final_epoch = max_epochs;
    // Only check every 100 epochs to minimize GPU read overhead
    let check_interval = if auto_terminate_dead_epochs > 0 { 100 } else { usize::MAX };
    
    for epoch in 0..max_epochs {
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

/// Generate pairs for 2D grid topology
fn generate_2d_pairs(
    num_programs: usize,
    width: usize,
    height: usize,
    neighbor_range: usize,
) -> Vec<(u32, u32)> {
    use rand::Rng;
    let mut rng = rand::rng();
    let mut pairs = Vec::with_capacity(num_programs / 2);
    let mut available = vec![true; num_programs];
    
    for i in 0..num_programs {
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

/// Generate all pairs for mega-simulation mode with absolute indices.
/// Includes both internal pairs (within each sim) and cross-border pairs (between adjacent sims).
/// Cross-border pairs are generated FIRST to ensure edge programs interact with neighbors.
#[cfg(feature = "wgpu-compute")]
fn generate_mega_pairs(
    num_programs: usize,
    grid_width: usize,
    grid_height: usize,
    neighbor_range: usize,
    num_sims: usize,
    layout_cols: usize,
    layout_rows: usize,
) -> Vec<(u32, u32)> {
    use rand::Rng;
    let mut rng = rand::rng();
    let mut all_pairs = Vec::new();
    
    // Track which programs are already paired (globally)
    let total_programs = num_programs * num_sims;
    let mut available = vec![true; total_programs];
    
    // FIRST: Generate cross-border pairs between adjacent simulations
    // This ensures edge programs get a chance to interact across simulation boundaries
    for sim_row in 0..layout_rows {
        for sim_col in 0..layout_cols {
            let sim_idx = sim_row * layout_cols + sim_col;
            let sim_offset = sim_idx * num_programs;
            
            // Right neighbor
            if sim_col + 1 < layout_cols {
                let neighbor_sim = sim_row * layout_cols + (sim_col + 1);
                let neighbor_offset = neighbor_sim * num_programs;
                
                // Pair right edge with left edge
                for y in 0..grid_height {
                    // Right edge of current sim
                    let local_p1 = y * grid_width + (grid_width - 1);
                    let global_p1 = sim_offset + local_p1;
                    
                    // Left edge of neighbor sim
                    let local_p2 = y * grid_width + 0;
                    let global_p2 = neighbor_offset + local_p2;
                    
                    // Pair these edge programs
                    if available[global_p1] && available[global_p2] {
                        all_pairs.push((global_p1 as u32, global_p2 as u32));
                        available[global_p1] = false;
                        available[global_p2] = false;
                    }
                }
            }
            
            // Bottom neighbor  
            if sim_row + 1 < layout_rows {
                let neighbor_sim = (sim_row + 1) * layout_cols + sim_col;
                let neighbor_offset = neighbor_sim * num_programs;
                
                // Pair bottom edge with top edge
                for x in 0..grid_width {
                    // Bottom edge of current sim
                    let local_p1 = (grid_height - 1) * grid_width + x;
                    let global_p1 = sim_offset + local_p1;
                    
                    // Top edge of neighbor sim
                    let local_p2 = 0 * grid_width + x;
                    let global_p2 = neighbor_offset + local_p2;
                    
                    if available[global_p1] && available[global_p2] {
                        all_pairs.push((global_p1 as u32, global_p2 as u32));
                        available[global_p1] = false;
                        available[global_p2] = false;
                    }
                }
            }
        }
    }
    
    let cross_border_count = all_pairs.len();
    
    // SECOND: Generate internal pairs for each simulation (remaining programs)
    for sim_idx in 0..num_sims {
        let sim_offset = sim_idx * num_programs;
        
        for local_i in 0..num_programs {
            let global_i = sim_offset + local_i;
            if !available[global_i] {
                continue;
            }
            
            let x = local_i % grid_width;
            let y = local_i / grid_width;
            
            // Find available neighbors within this simulation
            let mut neighbors = Vec::new();
            for dx in -(neighbor_range as i32)..=(neighbor_range as i32) {
                for dy in -(neighbor_range as i32)..=(neighbor_range as i32) {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = (x as i32 + dx).rem_euclid(grid_width as i32) as usize;
                    let ny = (y as i32 + dy).rem_euclid(grid_height as i32) as usize;
                    let local_neighbor = ny * grid_width + nx;
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
    
    // Log cross-border stats for debugging
    eprintln!("  Cross-border pairs generated first: {}", cross_border_count);
    
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
#[cfg(feature = "wgpu-compute")]
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
    let sim_size = grid_width * grid_height * 64;
    
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

/// Format a Unix timestamp as a human-readable string
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
// RAW DATA SAVING & ASYNC WRITER
// ============================================================================

/// Message type for async save operations
enum SaveMessage {
    /// Save raw soup data
    RawData {
        data: Vec<u8>,
        epoch: usize,
        path: String,
        grid_width: usize,
        grid_height: usize,
        num_sims: usize,
        layout: [usize; 2],
    },
    /// Shutdown the writer thread
    Shutdown,
}

/// Async writer handle - sends save operations to a background thread
struct AsyncWriter {
    sender: Sender<SaveMessage>,
    handle: Option<JoinHandle<()>>,
}

impl AsyncWriter {
    /// Create a new async writer with a background thread
    fn new() -> Self {
        let (sender, receiver) = mpsc::channel::<SaveMessage>();
        
        let handle = thread::spawn(move || {
            while let Ok(msg) = receiver.recv() {
                match msg {
                    SaveMessage::RawData { data, epoch, path, grid_width, grid_height, num_sims, layout } => {
                        if let Err(e) = save_raw_data_sync(&data, epoch, &path, grid_width, grid_height, num_sims, layout) {
                            eprintln!("Async save error (epoch {}): {}", epoch, e);
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
    
    /// Queue a raw data save operation (non-blocking)
    fn save_raw(&self, data: Vec<u8>, epoch: usize, path: &str, grid_width: usize, grid_height: usize, num_sims: usize, layout: [usize; 2]) {
        let msg = SaveMessage::RawData {
            data,
            epoch,
            path: path.to_string(),
            grid_width,
            grid_height,
            num_sims,
            layout,
        };
        if self.sender.send(msg).is_err() {
            eprintln!("Warning: async save queue full or closed");
        }
    }
    
    /// Wait for all pending saves to complete and shutdown
    fn shutdown(mut self) {
        let _ = self.sender.send(SaveMessage::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Save raw soup data to disk (synchronous version) with Zstd compression
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

/// Load raw data file header (supports both v1 BFFR and v2 BFF2 formats)
fn load_raw_header(path: &Path) -> std::io::Result<RawDataHeader> {
    use std::io::Read;
    let mut file = File::open(path)?;
    
    // Read magic
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != b"BFFR" && &magic != b"BFF2" {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid raw data magic"));
    }
    
    // Read version
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    let _version = u32::from_le_bytes(buf);
    
    // Read metadata
    file.read_exact(&mut buf)?;
    let epoch = u32::from_le_bytes(buf) as usize;
    file.read_exact(&mut buf)?;
    let grid_width = u32::from_le_bytes(buf) as usize;
    file.read_exact(&mut buf)?;
    let grid_height = u32::from_le_bytes(buf) as usize;
    file.read_exact(&mut buf)?;
    let num_sims = u32::from_le_bytes(buf) as usize;
    file.read_exact(&mut buf)?;
    let layout_cols = u32::from_le_bytes(buf) as usize;
    file.read_exact(&mut buf)?;
    let layout_rows = u32::from_le_bytes(buf) as usize;
    
    Ok(RawDataHeader {
        epoch,
        grid_width,
        grid_height,
        num_sims,
        layout: [layout_cols, layout_rows],
    })
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

