/// Simulation framework matching cubff semantics
/// Runs an evolutionary "primordial soup" of BFF programs

use crate::bff::{self, SINGLE_TAPE_SIZE, FULL_TAPE_SIZE, CrossProgramCopy};
use crate::energy::{EnergyConfig, EnergySystem, CopyEvent};
use rand::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// SplitMix64 PRNG - matching cubff's implementation for reproducibility
#[inline]
pub fn split_mix_64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Color palette for bytes - BFF commands get special colors
/// Returns (R, G, B) tuple
pub fn byte_to_color(byte: u8) -> (u8, u8, u8) {
    match byte {
        // Null byte - red (often indicates empty/dead)
        0 => (255, 0, 0),
        // BFF commands - distinctive colors
        b'[' | b']' => (0, 200, 0),     // Loops - green
        b'+' | b'-' => (200, 0, 200),   // Increment/decrement - magenta
        b'.' | b',' => (255, 165, 0),   // Copy operations - orange
        b'<' | b'>' => (0, 128, 255),   // Head0 movement - blue
        b'{' | b'}' => (0, 200, 200),   // Head1 movement - cyan
        // All other bytes - grayscale based on value
        _ => {
            let v = 192 + (byte / 4);
            (v, v, v)
        }
    }
}

/// Topology for program interactions
#[derive(Clone, Debug)]
pub enum Topology {
    /// Programs can interact with any other program (shuffled each epoch)
    Random,
    /// Programs arranged on a 2D grid, interact only with neighbors
    Grid2D {
        width: usize,
        height: usize,
        /// How far neighbors can reach (e.g., 2 means ±2 in each direction)
        neighbor_range: usize,
    },
}

impl Default for Topology {
    fn default() -> Self {
        Topology::Random
    }
}

/// Configuration for the simulation
#[derive(Clone)]
pub struct SimulationParams {
    /// Number of programs in the soup
    pub num_programs: usize,
    /// Random seed
    pub seed: u64,
    /// Mutation probability (out of 2^30)
    pub mutation_prob: u32,
    /// How many epochs between callbacks
    pub callback_interval: usize,
    /// Max steps per program execution
    pub steps_per_run: usize,
    /// Whether to zero-initialize (vs random)
    pub zero_init: bool,
    /// Whether to permute/shuffle program pairings each epoch
    pub permute_programs: bool,
    /// Topology for program interactions
    pub topology: Topology,
    /// Energy system configuration (None = disabled)
    pub energy_config: Option<EnergyConfig>,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            num_programs: 128 * 1024,  // 128K programs
            seed: 0,
            mutation_prob: 1 << 18,     // ~1/4096 mutation rate
            callback_interval: 64,
            steps_per_run: 8 * 1024,
            zero_init: false,
            permute_programs: true,
            topology: Topology::Random,
            energy_config: None,        // Energy system disabled by default
        }
    }
}

/// Statistics from the simulation
#[derive(Clone)]
pub struct SimulationState {
    pub epoch: usize,
    pub total_ops: u64,
    pub ops_this_interval: u64,
    pub elapsed_secs: f64,
    /// Approximate entropy (using simple byte frequency)
    pub h0: f64,
    /// Most common bytes
    pub byte_counts: [usize; 256],
}

impl Default for SimulationState {
    fn default() -> Self {
        Self {
            epoch: 0,
            total_ops: 0,
            ops_this_interval: 0,
            elapsed_secs: 0.0,
            h0: 0.0,
            byte_counts: [0; 256],
        }
    }
}

/// The main simulation structure
pub struct Simulation {
    /// The "soup" - all program tapes concatenated
    /// Each program is SINGLE_TAPE_SIZE bytes
    pub soup: Vec<u8>,
    /// Shuffle indices for pairing
    shuffle_idx: Vec<u32>,
    /// Parameters
    params: SimulationParams,
    /// Current epoch
    epoch: usize,
    /// Total operations executed
    total_ops: u64,
    /// RNG
    rng: StdRng,
    /// Allowed interactions for spatial topology (program_idx -> list of neighbor indices)
    allowed_interactions: Vec<Vec<u32>>,
    /// Energy system (optional)
    energy_system: Option<EnergySystem>,
}

impl Simulation {
    /// Create a new simulation
    pub fn new(params: SimulationParams) -> Self {
        let num_programs = params.num_programs;
        assert!(num_programs % 2 == 0, "num_programs must be even");
        
        let rng = StdRng::seed_from_u64(params.seed);
        
        // Initialize soup
        let mut soup = vec![0u8; num_programs * SINGLE_TAPE_SIZE];
        
        if !params.zero_init {
            // Random initialization using SplitMix64 for reproducibility
            for i in 0..soup.len() {
                let seed_val = split_mix_64(
                    (SINGLE_TAPE_SIZE as u64) * (num_programs as u64) * params.seed
                    + (i as u64)
                );
                soup[i] = (seed_val % 256) as u8;
            }
        }
        
        // Initialize shuffle indices
        let shuffle_idx: Vec<u32> = (0..num_programs as u32).collect();
        
        // Build allowed_interactions for spatial topology
        let allowed_interactions = Self::build_allowed_interactions(&params);
        
        // Initialize energy system if configured
        let energy_system = params.energy_config.as_ref().map(|config| {
            let (grid_width, grid_height) = match &params.topology {
                Topology::Grid2D { width, height, .. } => (*width, *height),
                Topology::Random => {
                    // For random topology, assume a square-ish grid
                    let side = (num_programs as f64).sqrt() as usize;
                    (side, num_programs / side)
                }
            };
            EnergySystem::new(config.clone(), grid_width, grid_height)
        });
        
        Self {
            soup,
            shuffle_idx,
            params,
            epoch: 0,
            total_ops: 0,
            rng,
            allowed_interactions,
            energy_system,
        }
    }
    
    /// Build the allowed interactions list based on topology
    fn build_allowed_interactions(params: &SimulationParams) -> Vec<Vec<u32>> {
        match &params.topology {
            Topology::Random => {
                // Empty - will use shuffle-based pairing
                Vec::new()
            }
            Topology::Grid2D { width, height, neighbor_range } => {
                let range = *neighbor_range as i32;
                let w = *width;
                let h = *height;
                
                assert_eq!(w * h, params.num_programs, 
                    "Grid dimensions {}x{} = {} don't match num_programs {}", 
                    w, h, w * h, params.num_programs);
                
                let mut interactions = vec![Vec::new(); params.num_programs];
                
                for y in 0..h {
                    for x in 0..w {
                        let idx = y * w + x;
                        
                        // Add all neighbors within range
                        for dy in -range..=range {
                            for dx in -range..=range {
                                if dx == 0 && dy == 0 {
                                    continue; // Skip self
                                }
                                
                                let nx = x as i32 + dx;
                                let ny = y as i32 + dy;
                                
                                // Check bounds
                                if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                                    let neighbor_idx = (ny as usize) * w + (nx as usize);
                                    interactions[idx].push(neighbor_idx as u32);
                                }
                            }
                        }
                    }
                }
                
                interactions
            }
        }
    }
    
    /// Get a program's tape slice
    pub fn get_program(&self, idx: usize) -> &[u8] {
        let start = idx * SINGLE_TAPE_SIZE;
        &self.soup[start..start + SINGLE_TAPE_SIZE]
    }
    
    /// Get a mutable program's tape slice
    pub fn get_program_mut(&mut self, idx: usize) -> &mut [u8] {
        let start = idx * SINGLE_TAPE_SIZE;
        &mut self.soup[start..start + SINGLE_TAPE_SIZE]
    }
    
    /// Shuffle the program indices for pairing (Random topology only)
    fn shuffle_indices(&mut self) {
        if self.params.permute_programs {
            // Fisher-Yates shuffle using SplitMix64
            let len = self.shuffle_idx.len();
            for i in (1..len).rev() {
                let j = (split_mix_64(
                    self.params.seed ^ split_mix_64(self.epoch as u64 * len as u64 + i as u64)
                ) as usize) % (i + 1);
                self.shuffle_idx.swap(i, j);
            }
        } else {
            // Alternating neighbor pairing
            if self.epoch % 2 == 0 {
                for i in 0..self.shuffle_idx.len() {
                    self.shuffle_idx[i] = if i == 0 { 
                        self.params.num_programs as u32 - 1 
                    } else { 
                        i as u32 - 1 
                    };
                }
            } else {
                for i in 0..self.shuffle_idx.len() {
                    self.shuffle_idx[i] = i as u32;
                }
            }
        }
    }
    
    /// Build pairs for spatial topology (2D grid)
    fn build_spatial_pairs(&self) -> Vec<(usize, usize, usize)> {
        let num_programs = self.params.num_programs;
        let mut used = vec![false; num_programs];
        let mut pairs = Vec::with_capacity(num_programs / 2);
        
        // Shuffle order of programs to try
        let mut order: Vec<usize> = (0..num_programs).collect();
        for i in (1..order.len()).rev() {
            let j = (split_mix_64(
                self.params.seed ^ split_mix_64(self.epoch as u64 * num_programs as u64 + i as u64)
            ) as usize) % (i + 1);
            order.swap(i, j);
        }
        
        let mut pair_idx = 0;
        for &i in &order {
            if used[i] || self.allowed_interactions[i].is_empty() {
                continue;
            }
            
            // Pick a random neighbor
            let neighbors = &self.allowed_interactions[i];
            let seed = split_mix_64(
                split_mix_64(self.params.seed) ^ split_mix_64(self.epoch as u64) ^ split_mix_64(i as u64)
            );
            let neighbor_idx = (seed as usize) % neighbors.len();
            let neighbor = neighbors[neighbor_idx] as usize;
            
            if used[neighbor] {
                continue;
            }
            
            used[i] = true;
            used[neighbor] = true;
            pairs.push((pair_idx, i, neighbor));
            pair_idx += 1;
        }
        
        pairs
    }
    
    /// Run one epoch of the simulation (parallelized with Rayon)
    pub fn run_epoch(&mut self) -> u64 {
        // Pre-epoch: Update energy states and handle deaths
        if let Some(ref mut energy) = self.energy_system {
            let newly_dead = energy.update_epoch();
            // Zero out tapes for programs that just died
            for idx in newly_dead {
                let start = idx * SINGLE_TAPE_SIZE;
                for i in start..start + SINGLE_TAPE_SIZE {
                    self.soup[i] = 0;
                }
            }
        }
        
        // Build pairs based on topology
        let pairs: Vec<(usize, usize, usize)> = if self.allowed_interactions.is_empty() {
            // Random topology - use shuffle
            self.shuffle_indices();
            let num_pairs = self.params.num_programs / 2;
            (0..num_pairs)
                .map(|pair_idx| {
                    let p1_idx = self.shuffle_idx[2 * pair_idx] as usize;
                    let p2_idx = self.shuffle_idx[2 * pair_idx + 1] as usize;
                    (pair_idx, p1_idx, p2_idx)
                })
                .collect()
        } else {
            // Spatial topology - use neighbor-based pairing
            self.build_spatial_pairs()
        };
        
        // Get energy mutation permissions if energy system is enabled
        let p1_can_mutate: Vec<bool>;
        let p2_can_mutate: Vec<bool>;
        
        if let Some(ref energy) = self.energy_system {
            p1_can_mutate = pairs.iter().map(|&(_, p1, _)| energy.can_mutate(p1)).collect();
            p2_can_mutate = pairs.iter().map(|&(_, _, p2)| energy.can_mutate(p2)).collect();
        } else {
            // No energy system - all can mutate
            p1_can_mutate = vec![true; pairs.len()];
            p2_can_mutate = vec![true; pairs.len()];
        }
        
        // Process pairs in parallel
        // Each parallel task returns (p1_idx, p2_idx, result_tape, ops, cross_copies)
        let results: Vec<(usize, usize, [u8; FULL_TAPE_SIZE], usize, Vec<CrossProgramCopy>)> = pairs
            .par_iter()
            .enumerate()
            .map(|(i, &(pair_idx, p1_idx, p2_idx))| {
                // Create combined tape (128 bytes)
                let mut tape = [0u8; FULL_TAPE_SIZE];
                
                // Copy program 1 to first half
                let p1_start = p1_idx * SINGLE_TAPE_SIZE;
                tape[..SINGLE_TAPE_SIZE].copy_from_slice(
                    &self.soup[p1_start..p1_start + SINGLE_TAPE_SIZE]
                );
                
                // Copy program 2 to second half
                let p2_start = p2_idx * SINGLE_TAPE_SIZE;
                tape[SINGLE_TAPE_SIZE..].copy_from_slice(
                    &self.soup[p2_start..p2_start + SINGLE_TAPE_SIZE]
                );
                
                // Apply mutations only to programs that can mutate
                let mutation_seed = split_mix_64(
                    (self.params.num_programs as u64 * self.params.seed + self.epoch as u64) 
                    * FULL_TAPE_SIZE as u64 + pair_idx as u64
                );
                
                // Mutate first half (p1) if allowed
                if p1_can_mutate[i] {
                    for j in 0..SINGLE_TAPE_SIZE {
                        let rng_val = split_mix_64(mutation_seed.wrapping_add(j as u64));
                        let replacement = (rng_val & 0xFF) as u8;
                        let prob_rng = ((rng_val >> 8) & ((1u64 << 30) - 1)) as u32;
                        
                        if prob_rng < self.params.mutation_prob {
                            tape[j] = replacement;
                        }
                    }
                }
                
                // Mutate second half (p2) if allowed
                if p2_can_mutate[i] {
                    for j in SINGLE_TAPE_SIZE..FULL_TAPE_SIZE {
                        let rng_val = split_mix_64(mutation_seed.wrapping_add(j as u64));
                        let replacement = (rng_val & 0xFF) as u8;
                        let prob_rng = ((rng_val >> 8) & ((1u64 << 30) - 1)) as u32;
                        
                        if prob_rng < self.params.mutation_prob {
                            tape[j] = replacement;
                        }
                    }
                }
                
                // Run the program with copy tracking
                let eval_result = bff::evaluate_with_copy_tracking(&mut tape, self.params.steps_per_run);
                
                (p1_idx, p2_idx, tape, eval_result.ops, eval_result.cross_copies)
            })
            .collect();
        
        // Write results back to soup and collect copy events
        let mut ops_this_epoch = 0u64;
        let mut all_copy_events: Vec<CopyEvent> = Vec::new();
        
        for (p1_idx, p2_idx, tape, ops, cross_copies) in results {
            let p1_start = p1_idx * SINGLE_TAPE_SIZE;
            let p2_start = p2_idx * SINGLE_TAPE_SIZE;
            
            self.soup[p1_start..p1_start + SINGLE_TAPE_SIZE]
                .copy_from_slice(&tape[..SINGLE_TAPE_SIZE]);
            self.soup[p2_start..p2_start + SINGLE_TAPE_SIZE]
                .copy_from_slice(&tape[SINGLE_TAPE_SIZE..]);
            
            ops_this_epoch += ops as u64;
            
            // Convert cross-program copies to CopyEvents for energy system
            for copy in cross_copies {
                let (source_prog, dest_prog) = if copy.source_half == 0 {
                    (p1_idx, p2_idx)
                } else {
                    (p2_idx, p1_idx)
                };
                all_copy_events.push(CopyEvent::new(source_prog, dest_prog));
            }
        }
        
        // Post-epoch: Process copy events for energy system
        if let Some(ref mut energy) = self.energy_system {
            energy.process_copy_events(&all_copy_events);
        }
        
        self.total_ops += ops_this_epoch;
        self.epoch += 1;
        
        ops_this_epoch
    }
    
    /// Compute current statistics
    pub fn get_state(&self) -> SimulationState {
        let mut byte_counts = [0usize; 256];
        for &b in &self.soup {
            byte_counts[b as usize] += 1;
        }
        
        // Compute H0 (zero-order entropy)
        let total = self.soup.len() as f64;
        let mut h0 = 0.0;
        for &count in &byte_counts {
            if count > 0 {
                let p = count as f64 / total;
                h0 -= p * p.log2();
            }
        }
        
        SimulationState {
            epoch: self.epoch,
            total_ops: self.total_ops,
            ops_this_interval: 0,
            elapsed_secs: 0.0,
            h0,
            byte_counts,
        }
    }
    
    /// Run the simulation with a callback
    pub fn run<F>(&mut self, mut callback: F)
    where
        F: FnMut(&Self, &SimulationState) -> bool,
    {
        use std::time::Instant;
        
        let start = Instant::now();
        let mut interval_start = start;
        let mut interval_ops = 0u64;
        
        loop {
            let ops = self.run_epoch();
            interval_ops += ops;
            
            if self.epoch % self.params.callback_interval == 0 {
                let now = Instant::now();
                let mut state = self.get_state();
                state.elapsed_secs = (now - start).as_secs_f64();
                state.ops_this_interval = interval_ops;
                
                let interval_secs = (now - interval_start).as_secs_f64();
                let mops_s = interval_ops as f64 / interval_secs / 1_000_000.0;
                
                // Print status
                println!(
                    "Epoch {:8} | Ops: {:12} | {:.2} MOps/s | H0: {:.4} bits",
                    state.epoch, state.total_ops, mops_s, state.h0
                );
                
                // Print most common bytes
                let mut sorted: Vec<(usize, u8)> = state.byte_counts
                    .iter()
                    .enumerate()
                    .map(|(i, &c)| (c, i as u8))
                    .collect();
                sorted.sort_by(|a, b| b.0.cmp(&a.0));
                
                print!("Top bytes: ");
                for (count, byte) in sorted.iter().take(10) {
                    let c = if byte.is_ascii_graphic() || *byte == b' ' {
                        *byte as char
                    } else {
                        '·'
                    };
                    print!("'{}'{:02X}:{:.1}% ", c, byte, 
                           *count as f64 / self.soup.len() as f64 * 100.0);
                }
                println!();
                
                if callback(self, &state) {
                    break;
                }
                
                interval_start = now;
                interval_ops = 0;
            }
        }
    }
    
    /// Print a program
    pub fn print_program(&self, idx: usize) {
        let prog = self.get_program(idx);
        print!("{:4}: ", idx);
        for &b in prog {
            let c = if b.is_ascii_graphic() || b == b' ' {
                b as char
            } else if b == 0 {
                '␀'
            } else {
                ' '
            };
            if bff::is_command(b) {
                print!("\x1b[37;1m{}\x1b[;m", c);
            } else {
                print!("{}", c);
            }
        }
        println!();
    }
    
    /// Get grid dimensions if using 2D topology
    pub fn get_grid_dims(&self) -> Option<(usize, usize)> {
        match &self.params.topology {
            Topology::Grid2D { width, height, .. } => Some((*width, *height)),
            Topology::Random => None,
        }
    }
    
    /// Render the soup as a 2D image
    /// Each program becomes an 8x8 tile (since SINGLE_TAPE_SIZE = 64)
    /// Returns (width, height, RGB data)
    pub fn render_grid_image(&self) -> Option<(usize, usize, Vec<u8>)> {
        let (grid_w, grid_h) = self.get_grid_dims()?;
        
        // Each program is 64 bytes, rendered as 8x8 pixels
        let tile_size = 8;
        let img_w = grid_w * tile_size;
        let img_h = grid_h * tile_size;
        
        let mut pixels = vec![0u8; img_w * img_h * 3];
        
        for prog_y in 0..grid_h {
            for prog_x in 0..grid_w {
                let prog_idx = prog_y * grid_w + prog_x;
                let prog = self.get_program(prog_idx);
                
                // Render this program's 64 bytes as 8x8 tile
                for byte_idx in 0..SINGLE_TAPE_SIZE {
                    let byte = prog[byte_idx];
                    let (r, g, b) = byte_to_color(byte);
                    
                    let tile_x = byte_idx % tile_size;
                    let tile_y = byte_idx / tile_size;
                    
                    let px = prog_x * tile_size + tile_x;
                    let py = prog_y * tile_size + tile_y;
                    let pixel_idx = (py * img_w + px) * 3;
                    
                    // Add subtle grid lines at tile boundaries
                    let (r, g, b) = if tile_x == 0 || tile_y == 0 {
                        (r.saturating_sub(32), g.saturating_sub(32), b.saturating_sub(32))
                    } else {
                        (r, g, b)
                    };
                    
                    pixels[pixel_idx] = r;
                    pixels[pixel_idx + 1] = g;
                    pixels[pixel_idx + 2] = b;
                }
            }
        }
        
        Some((img_w, img_h, pixels))
    }
    
    /// Save the current state as a PPM image
    pub fn save_ppm<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let (width, height, pixels) = self.render_grid_image()
            .ok_or_else(|| std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Cannot render image without 2D topology"
            ))?;
        
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        // PPM header
        writeln!(writer, "P6")?;
        writeln!(writer, "{} {}", width, height)?;
        writeln!(writer, "255")?;
        
        // Pixel data
        writer.write_all(&pixels)?;
        
        Ok(())
    }
    
    /// Save frame with epoch number in filename
    pub fn save_frame<P: AsRef<Path>>(&self, dir: P, epoch: usize) -> std::io::Result<()> {
        let path = dir.as_ref().join(format!("{:012}.ppm", epoch));
        self.save_ppm(path)
    }
}


