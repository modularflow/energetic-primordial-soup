//! Island Model Simulation
//!
//! Run multiple populations ("islands") in parallel, each with potentially
//! different parameters and fitness functions. Periodically migrate programs
//! between islands based on fitness.
//!
//! NOTE: This module is not currently used by the main simulation but is kept
//! for potential future use or experimentation.

#![allow(dead_code)]

use crate::bff::SINGLE_TAPE_SIZE;
use crate::fitness::{self, FitnessFn};
use crate::simulation::{Simulation, SimulationParams, Topology};
use rand::prelude::*;

/// Configuration for a single island
#[derive(Clone)]
pub struct IslandConfig {
    pub name: String,
    pub grid_width: usize,
    pub grid_height: usize,
    pub seed: u64,
    pub mutation_rate: u32,  // 1/N
    pub steps_per_run: usize,
    pub neighbor_range: usize,
    pub fitness_fn_name: String,
}

impl Default for IslandConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            grid_width: 256,
            grid_height: 128,
            seed: 42,
            mutation_rate: 4096,
            steps_per_run: 8192,
            neighbor_range: 2,
            fitness_fn_name: "neutral".to_string(),
        }
    }
}

/// Configuration for the island model
pub struct IslandModelConfig {
    pub islands: Vec<IslandConfig>,
    pub migration_interval: usize,  // Epochs between migrations
    pub migration_rate: f64,        // Fraction of population to migrate (0.0-1.0)
    pub migration_topology: MigrationTopology,
    pub total_epochs: usize,
}

/// How islands are connected for migration
#[derive(Clone, Copy)]
pub enum MigrationTopology {
    Ring,       // Each island connects to next (circular)
    AllToAll,   // Every island can send to every other
    Star,       // All islands connect to island 0 (hub)
}

/// A single island's state
pub struct Island {
    pub config: IslandConfig,
    pub simulation: Simulation,
    pub fitness_fn: FitnessFn,
    pub epoch: usize,
}

impl Island {
    pub fn new(config: IslandConfig) -> Self {
        let num_programs = config.grid_width * config.grid_height;
        let mutation_prob = (1u64 << 30) / config.mutation_rate as u64;
        
        let params = SimulationParams {
            num_programs,
            seed: config.seed,
            mutation_prob: mutation_prob as u32,
            callback_interval: 64,
            steps_per_run: config.steps_per_run,
            zero_init: false,
            permute_programs: true,
            topology: Topology::Grid2D {
                width: config.grid_width,
                height: config.grid_height,
                neighbor_range: config.neighbor_range,
            },
            energy_config: None,
        };
        
        let simulation = Simulation::new(params);
        let fitness_fn = fitness::get_fitness_fn(&config.fitness_fn_name)
            .unwrap_or(fitness::fitness_neutral);
        
        Self {
            config,
            simulation,
            fitness_fn,
            epoch: 0,
        }
    }
    
    /// Run one epoch
    pub fn step(&mut self) -> u64 {
        let ops = self.simulation.run_epoch();
        self.epoch += 1;
        ops
    }
    
    /// Get the top N programs by fitness
    pub fn get_top_programs(&self, n: usize) -> Vec<(usize, f64, Vec<u8>)> {
        let num_programs = self.config.grid_width * self.config.grid_height;
        let mut scored: Vec<(usize, f64)> = (0..num_programs)
            .map(|i| {
                let program = self.simulation.get_program(i);
                let score = (self.fitness_fn)(program);
                (i, score)
            })
            .collect();
        
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        scored.into_iter()
            .take(n)
            .map(|(idx, score)| {
                let program = self.simulation.get_program(idx).to_vec();
                (idx, score, program)
            })
            .collect()
    }
    
    /// Replace random programs with immigrants
    pub fn receive_migrants(&mut self, migrants: &[Vec<u8>]) {
        let num_programs = self.config.grid_width * self.config.grid_height;
        let mut rng = rand::rng();
        
        for migrant in migrants {
            // Pick a random location to place the migrant
            let target_idx = rng.random_range(0..num_programs);
            let target_start = target_idx * SINGLE_TAPE_SIZE;
            
            // Copy migrant into soup
            self.simulation.soup[target_start..target_start + SINGLE_TAPE_SIZE]
                .copy_from_slice(&migrant[..SINGLE_TAPE_SIZE.min(migrant.len())]);
        }
    }
    
    /// Get average fitness of the population
    pub fn average_fitness(&self) -> f64 {
        let num_programs = self.config.grid_width * self.config.grid_height;
        let total: f64 = (0..num_programs)
            .map(|i| (self.fitness_fn)(self.simulation.get_program(i)))
            .sum();
        total / num_programs as f64
    }
}

/// The island model controller
pub struct IslandModel {
    pub islands: Vec<Island>,
    pub config: IslandModelConfig,
    pub epoch: usize,
}

impl IslandModel {
    pub fn new(config: IslandModelConfig) -> Self {
        let islands: Vec<Island> = config.islands.iter()
            .map(|ic| Island::new(ic.clone()))
            .collect();
        
        Self {
            islands,
            config,
            epoch: 0,
        }
    }
    
    /// Run one epoch on all islands (can be parallelized)
    pub fn step(&mut self) -> Vec<u64> {
        // Run epochs in parallel using threads
        let ops: Vec<u64> = self.islands.iter_mut()
            .map(|island| island.step())
            .collect();
        
        self.epoch += 1;
        
        // Check if it's time for migration
        if self.config.migration_interval > 0 
            && self.epoch % self.config.migration_interval == 0 
        {
            self.migrate();
        }
        
        ops
    }
    
    /// Migrate programs between islands based on topology
    fn migrate(&mut self) {
        let n_islands = self.islands.len();
        if n_islands < 2 {
            return;
        }
        
        // Calculate number of migrants per island
        let migrants_per_island = (self.islands[0].config.grid_width 
            * self.islands[0].config.grid_height) as f64 
            * self.config.migration_rate;
        let n_migrants = (migrants_per_island as usize).max(1);
        
        // Collect emigrants from each island (top fitness programs)
        let emigrants: Vec<Vec<Vec<u8>>> = self.islands.iter()
            .map(|island| {
                island.get_top_programs(n_migrants)
                    .into_iter()
                    .map(|(_, _, prog)| prog)
                    .collect()
            })
            .collect();
        
        // Determine migration pairs based on topology
        let pairs = match self.config.migration_topology {
            MigrationTopology::Ring => {
                // Each island sends to the next one
                (0..n_islands).map(|i| (i, (i + 1) % n_islands)).collect::<Vec<_>>()
            }
            MigrationTopology::AllToAll => {
                // Every island sends to every other
                let mut pairs = Vec::new();
                for i in 0..n_islands {
                    for j in 0..n_islands {
                        if i != j {
                            pairs.push((i, j));
                        }
                    }
                }
                pairs
            }
            MigrationTopology::Star => {
                // All send to island 0, island 0 sends to all
                let mut pairs = Vec::new();
                for i in 1..n_islands {
                    pairs.push((i, 0));  // All to center
                    pairs.push((0, i));  // Center to all
                }
                pairs
            }
        };
        
        // Perform migrations
        for (from, to) in pairs {
            if from != to {
                self.islands[to].receive_migrants(&emigrants[from]);
            }
        }
        
        println!("  [Migration] Epoch {}: Migrated {} programs between {} islands", 
            self.epoch, n_migrants, n_islands);
    }
    
    /// Print status of all islands
    pub fn print_status(&self) {
        println!("\n=== Island Status (Epoch {}) ===", self.epoch);
        for (i, island) in self.islands.iter().enumerate() {
            let avg_fitness = island.average_fitness();
            println!("  Island {} '{}': avg_fitness={:.4}, fitness_fn={}",
                i, island.config.name, avg_fitness, island.config.fitness_fn_name);
        }
        println!();
    }
    
    /// Get combined soup from all islands (for visualization)
    pub fn get_combined_soup(&self) -> Vec<u8> {
        let mut combined = Vec::new();
        for island in &self.islands {
            combined.extend_from_slice(&island.simulation.soup);
        }
        combined
    }
}

/// Run island model with progress callback
pub fn run_island_model<F>(
    config: IslandModelConfig,
    mut callback: F,
) where
    F: FnMut(&IslandModel, usize) -> bool,  // Returns true to stop
{
    let mut model = IslandModel::new(config);
    
    println!("Starting Island Model with {} islands", model.islands.len());
    for (i, island) in model.islands.iter().enumerate() {
        println!("  Island {}: '{}' - {}x{} grid, fitness={}",
            i, island.config.name,
            island.config.grid_width, island.config.grid_height,
            island.config.fitness_fn_name);
    }
    println!();
    
    loop {
        let _ops = model.step();
        
        if callback(&model, model.epoch) {
            break;
        }
    }
}


