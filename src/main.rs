mod bff;
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
use std::io::Write;
use std::path::Path;

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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    pub frame_interval: usize,
    pub frames_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EnergySettings {
    pub enabled: bool,
    pub sources: usize,
    pub radius: usize,
    pub reserve_epochs: u8,
    pub death_epochs: u8,
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
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            frame_interval: 64,
            frames_dir: "frames".to_string(),
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
    auto_terminate_dead_epochs: usize,
    parallel_sims: usize,
    // Energy system options
    energy_enabled: bool,
    energy_sources: usize,
    energy_radius: usize,
    energy_reserve: u8,
    energy_death: u8,
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
            auto_terminate_dead_epochs: 0,
            parallel_sims: 1,
            // Energy defaults
            energy_enabled: false,
            energy_sources: 4,
            energy_radius: 64,
            energy_reserve: 5,
            energy_death: 10,
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
            auto_terminate_dead_epochs: c.simulation.auto_terminate_dead_epochs,
            parallel_sims: c.simulation.parallel_sims,
            energy_enabled: c.energy.enabled,
            energy_sources: c.energy.sources,
            energy_radius: c.energy.radius,
            energy_reserve: c.energy.reserve_epochs,
            energy_death: c.energy.death_epochs,
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
    println!("    --help                    Print this help message");
}

fn main() {
    let args = parse_args();
    
    let num_programs = args.grid_width * args.grid_height;
    let save_frames = args.frame_interval > 0;
    
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
            Some(energy::EnergyConfig::full(
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
                println!("\n  Backend: GPU (wgpu/Vulkan) - {} parallel simulations\n", args.parallel_sims);
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
                    args.neighbor_range,
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
    energy_random: bool,
    energy_max_sources: usize,
    energy_source_lifetime: usize,
    energy_spawn_rate: usize,
) {
    println!("  Backend: CPU ({} threads)\n", rayon::current_num_threads());
    
    // Build energy config if enabled
    let energy_config = if energy_enabled {
        Some(energy::EnergyConfig::full(
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
    neighbor_range: usize,
) {
    use std::time::Instant;
    
    let num_sims = multi_sim.num_sims();
    
    // Initialize all simulations
    multi_sim.init_random_all();
    
    // Generate pair indices for 2D grid topology
    let pairs = generate_2d_pairs(num_programs, grid_width, grid_height, neighbor_range);
    multi_sim.set_pairs_all(&pairs);
    
    println!("Running {} epochs x {} simulations...\n", max_epochs, num_sims);
    
    let mut total_ops = 0u64;
    let start_time = Instant::now();
    let mut last_report = Instant::now();
    
    // Progress bar width
    const BAR_WIDTH: usize = 30;
    
    for epoch in 0..max_epochs {
        // Update dynamic energy sources if enabled
        if let Some(ref mut config) = energy_config {
            if config.is_dynamic() && config.update_sources(epoch) {
                multi_sim.update_energy_config_all(config);
            }
        }
        
        let ops = multi_sim.run_epoch_all();
        total_ops += ops;
        
        // Save frames from ALL simulations
        if save_frames && frame_interval > 0 && epoch % frame_interval == 0 {
            for sim_idx in 0..num_sims {
                let soup = multi_sim.get_soup(sim_idx);
                let sim_frames_dir = format!("{}/sim_{}", frames_dir, sim_idx);
                let _ = fs::create_dir_all(&sim_frames_dir);
                if let Err(e) = save_ppm_frame(&soup, grid_width, grid_height, &sim_frames_dir, epoch) {
                    eprintln!("Warning: Could not save frame {} for sim {}: {}", epoch, sim_idx, e);
                }
            }
        }
        
        // Report progress every second
        if last_report.elapsed().as_secs() >= 1 || epoch == max_epochs - 1 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let mops = total_ops as f64 / elapsed / 1_000_000.0;
            
            // Calculate progress
            let progress = (epoch + 1) as f64 / max_epochs as f64;
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
            
            print!("\r[{}] {:>3.0}% | {} sims | {:.1}B ops/s | ETA {:02}:{:02}  ",
                bar,
                progress * 100.0,
                num_sims,
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
    
    // Save final frames from all simulations
    if save_frames && frame_interval > 0 {
        for sim_idx in 0..num_sims {
            let soup = multi_sim.get_soup(sim_idx);
            let sim_frames_dir = format!("{}/sim_{}", frames_dir, sim_idx);
            let _ = fs::create_dir_all(&sim_frames_dir);
            if let Err(e) = save_ppm_frame(&soup, grid_width, grid_height, &sim_frames_dir, max_epochs) {
                eprintln!("Warning: Could not save final frame for sim {}: {}", sim_idx, e);
            }
        }
    }
    
    let elapsed = start_time.elapsed().as_secs_f64();
    let throughput = total_ops as f64 / elapsed / 1e9;
    let per_sim = throughput / num_sims as f64;
    
    println!();
    println!("┌─────────────────────────────────────────────┐");
    println!("│     ✓ All {} Simulations Complete           │", num_sims);
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

/// Save a PPM frame from soup data
fn save_ppm_frame(
    soup: &[u8],
    grid_width: usize,
    grid_height: usize,
    frames_dir: &str,
    epoch: usize,
) -> std::io::Result<()> {
    let num_programs = grid_width * grid_height;
    let img_width = grid_width * 8;
    let img_height = grid_height * 8;
    let mut img_data = vec![0u8; img_width * img_height * 3];
    
    // Generate byte colors (BFF command highlighting)
    let byte_colors = init_byte_colors();
    
    for i in 0..num_programs {
        let grid_x = i % grid_width;
        let grid_y = i / grid_width;
        let program_start = i * SINGLE_TAPE_SIZE;
        
        for j in 0..SINGLE_TAPE_SIZE {
            let pixel_x = grid_x * 8 + (j % 8);
            let pixel_y = grid_y * 8 + (j / 8);
            let img_idx = (pixel_y * img_width + pixel_x) * 3;
            
            let byte_val = soup[program_start + j];
            let color = byte_colors[byte_val as usize];
            
            img_data[img_idx] = color[0];
            img_data[img_idx + 1] = color[1];
            img_data[img_idx + 2] = color[2];
        }
    }
    
    let path = Path::new(frames_dir).join(format!("{:08}.ppm", epoch));
    let mut file = File::create(&path)?;
    writeln!(file, "P6")?;
    writeln!(file, "{} {}", img_width, img_height)?;
    writeln!(file, "255")?;
    file.write_all(&img_data)?;
    
    Ok(())
}

/// Initialize byte colors for visualization
fn init_byte_colors() -> [[u8; 3]; 256] {
    let mut colors = [[0u8; 3]; 256];
    
    for i in 0..256 {
        // Default: grayscale based on byte value
        let gray = i as u8;
        colors[i] = [gray, gray, gray];
    }
    
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
