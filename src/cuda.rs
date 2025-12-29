//! CUDA-accelerated simulation backend
//!
//! This module provides GPU acceleration using NVIDIA CUDA.
//! Unlike the wgpu backend, CUDA has no 4GB buffer limit - you can use
//! your full GPU memory (e.g., 24GB on RTX 4090).
//!
//! Build with: cargo build --release --features cuda
//!
//! Note: Requires NVIDIA GPU and CUDA toolkit installed.

#![allow(dead_code)]

#[cfg(feature = "cuda")]
use cudarc::driver::*;
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Generate simple pairs for simulation (adjacent programs)
#[cfg(feature = "cuda")]
fn generate_pairs(num_programs: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::with_capacity(num_programs / 2);
    for i in (0..num_programs).step_by(2) {
        if i + 1 < num_programs {
            pairs.push((i, i + 1));
        }
    }
    pairs
}

/// CUDA kernel source for batched multi-simulation BFF evaluation
/// This kernel supports:
/// - Multiple simulations in parallel (batched)
/// - Energy system with per-sim death_timer and reserve_duration
/// - Full 64-bit addressing (no 4GB limit)
#[cfg(feature = "cuda")]
const BFF_CUDA_KERNEL: &str = r#"
extern "C" __global__ void bff_batched_evaluate(
    unsigned char* soup,              // All programs across all sims: [sim0_prog0, sim0_prog1, ..., sim1_prog0, ...]
    const unsigned int* pair_indices, // Pairs per sim: [p1, p2, p1, p2, ...]
    unsigned int* energy_state,       // Packed energy state per program: reserve(8) | timer(8) | dead(8) | unused(8)
    const unsigned int* sim_configs,  // Per-sim configs: [death_timer, reserve_duration] pairs
    const unsigned int* energy_map,   // Bitmask: 1 bit per program indicating if in energy zone
    unsigned long long* ops_count,    // Atomic counter for total ops
    // Packed parameters (to fit cudarc's 12-param limit)
    unsigned long long params_packed1, // num_pairs(hi) | num_programs(lo)
    unsigned long long params_packed2, // num_sims(hi) | steps_per_run(lo)
    unsigned long long params_packed3, // mutation_prob(hi) | energy_enabled(lo)
    unsigned long long seed,
    unsigned long long epoch
) {
    // Unpack parameters
    unsigned int num_pairs = (unsigned int)(params_packed1 >> 32);
    unsigned int num_programs = (unsigned int)(params_packed1 & 0xFFFFFFFF);
    unsigned int num_sims = (unsigned int)(params_packed2 >> 32);
    unsigned int steps_per_run = (unsigned int)(params_packed2 & 0xFFFFFFFF);
    unsigned int mutation_prob = (unsigned int)(params_packed3 >> 32);
    unsigned int energy_enabled = (unsigned int)(params_packed3 & 0xFFFFFFFF);
    const int SINGLE_TAPE_SIZE = 64;
    const int FULL_TAPE_SIZE = 128;
    
    // Global pair index across all sims
    unsigned long long global_idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    
    
    // Calculate which sim and which pair within that sim
    unsigned int sim_idx = global_idx / num_pairs;
    unsigned int pair_idx = global_idx % num_pairs;
    
    if (sim_idx >= num_sims || pair_idx >= num_pairs) return;
    
    // Get program indices (local to sim)
    unsigned int p1_local = pair_indices[pair_idx * 2];
    unsigned int p2_local = pair_indices[pair_idx * 2 + 1];
    
    // Convert to absolute indices
    unsigned long long sim_offset = (unsigned long long)sim_idx * num_programs;
    unsigned long long p1_abs = sim_offset + p1_local;
    unsigned long long p2_abs = sim_offset + p2_local;
    
    // Get per-sim energy config
    unsigned int death_timer = sim_configs[sim_idx * 2];
    unsigned int reserve_duration = sim_configs[sim_idx * 2 + 1];
    
    // Check energy zone membership (bitmask lookup)
    auto in_energy_zone = [&](unsigned long long prog_idx) -> bool {
        unsigned int word_idx = prog_idx / 32;
        unsigned int bit_idx = prog_idx % 32;
        return (energy_map[word_idx] & (1u << bit_idx)) != 0;
    };
    
    // Energy state helpers - packed as: reserve(16 bits) | timer(8 bits) | dead(8 bits)
    auto get_reserve = [](unsigned int state) -> unsigned int { return state & 0xFFFF; };
    auto get_timer = [](unsigned int state) -> unsigned int { return (state >> 16) & 0xFF; };
    auto is_dead = [](unsigned int state) -> bool { return ((state >> 24) & 0xFF) != 0; };
    auto pack_state = [](unsigned int reserve, unsigned int timer, bool dead) -> unsigned int {
        return (reserve & 0xFFFF) | ((timer & 0xFF) << 16) | ((dead ? 1u : 0u) << 24);
    };
    
    // Load energy states
    unsigned int p1_state = energy_state[p1_abs];
    unsigned int p2_state = energy_state[p2_abs];
    bool p1_in_zone = energy_enabled && in_energy_zone(p1_abs);
    bool p2_in_zone = energy_enabled && in_energy_zone(p2_abs);
    bool p1_was_dead = energy_enabled && is_dead(p1_state);
    bool p2_was_dead = energy_enabled && is_dead(p2_state);
    
    // Skip if both dead and not in energy zones (can't be revived)
    if (energy_enabled && p1_was_dead && p2_was_dead && !p1_in_zone && !p2_in_zone) {
        return;
    }
    
    // Can mutate check
    auto can_mutate = [&](unsigned long long prog_idx, unsigned int state, bool in_zone) -> bool {
        if (!energy_enabled) return true;
        if (is_dead(state)) return false;
        return in_zone || get_reserve(state) > 0;
    };
    
    bool p1_can_mutate = can_mutate(p1_abs, p1_state, p1_in_zone);
    bool p2_can_mutate = can_mutate(p2_abs, p2_state, p2_in_zone);
    
    // Local tape (128 bytes)
    unsigned char tape[FULL_TAPE_SIZE];
    
    // Copy programs to local tape (use 64-bit offsets)
    unsigned long long p1_byte_offset = p1_abs * SINGLE_TAPE_SIZE;
    unsigned long long p2_byte_offset = p2_abs * SINGLE_TAPE_SIZE;
    
    for (int i = 0; i < SINGLE_TAPE_SIZE; i++) {
        tape[i] = soup[p1_byte_offset + i];
        tape[SINGLE_TAPE_SIZE + i] = soup[p2_byte_offset + i];
    }
    
    // LCG for fast mutations
    auto lcg = [](unsigned int s) -> unsigned int {
        return s * 1664525u + 1013904223u;
    };
    
    // Apply mutations with geometric skip (sparse mutation optimization)
    unsigned int rng = (unsigned int)(seed ^ epoch ^ (global_idx * 0x9E3779B9ULL));
    
    if (p1_can_mutate) {
        for (int i = 0; i < SINGLE_TAPE_SIZE; i++) {
            rng = lcg(rng);
            if ((rng & 0x3FFFFFFF) < mutation_prob) {
                rng = lcg(rng);
                tape[i] = (unsigned char)(rng & 0xFF);
            }
        }
    }
    
    if (p2_can_mutate) {
        for (int i = SINGLE_TAPE_SIZE; i < FULL_TAPE_SIZE; i++) {
            rng = lcg(rng);
            if ((rng & 0x3FFFFFFF) < mutation_prob) {
                rng = lcg(rng);
                tape[i] = (unsigned char)(rng & 0xFF);
            }
        }
    }
    
    // Track if copies occurred (for energy inheritance)
    bool p1_received_copy = false;
    bool p2_received_copy = false;
    
    // BFF Evaluation
    int pos = 2;
    int head0 = tape[0] & (FULL_TAPE_SIZE - 1);
    int head1 = tape[1] & (FULL_TAPE_SIZE - 1);
    unsigned int ops = 0;
    
    
    for (unsigned int step = 0; step < steps_per_run; step++) {
        head0 = head0 & (FULL_TAPE_SIZE - 1);
        head1 = head1 & (FULL_TAPE_SIZE - 1);
        
        unsigned char cmd = tape[pos];
        
        // BFF commands as byte values: < > { } + - . , [ ]
        // 0x3C=60, 0x3E=62, 0x7B=123, 0x7D=125, 0x2B=43, 0x2D=45, 0x2E=46, 0x2C=44, 0x5B=91, 0x5D=93
        switch (cmd) {
            case 0x3C: head0--; ops++; break;  // '<'
            case 0x3E: head0++; ops++; break;  // '>'
            case 0x7B: head1--; ops++; break;  // '{'
            case 0x7D: head1++; ops++; break;  // '}'
            case 0x2B: tape[head0 & (FULL_TAPE_SIZE-1)]++; ops++; break;  // '+'
            case 0x2D: tape[head0 & (FULL_TAPE_SIZE-1)]--; ops++; break;  // '-'
            case 0x2E:  // '.'
                tape[head1 & (FULL_TAPE_SIZE-1)] = tape[head0 & (FULL_TAPE_SIZE-1)];
                // Track copy direction for energy
                if ((head0 & (FULL_TAPE_SIZE-1)) < SINGLE_TAPE_SIZE && 
                    (head1 & (FULL_TAPE_SIZE-1)) >= SINGLE_TAPE_SIZE) {
                    p2_received_copy = true;
                } else if ((head0 & (FULL_TAPE_SIZE-1)) >= SINGLE_TAPE_SIZE && 
                           (head1 & (FULL_TAPE_SIZE-1)) < SINGLE_TAPE_SIZE) {
                    p1_received_copy = true;
                }
                ops++;
                break;
            case 0x2C:  // ','
                tape[head0 & (FULL_TAPE_SIZE-1)] = tape[head1 & (FULL_TAPE_SIZE-1)];
                // Track copy direction for energy
                if ((head1 & (FULL_TAPE_SIZE-1)) < SINGLE_TAPE_SIZE && 
                    (head0 & (FULL_TAPE_SIZE-1)) >= SINGLE_TAPE_SIZE) {
                    p2_received_copy = true;
                } else if ((head1 & (FULL_TAPE_SIZE-1)) >= SINGLE_TAPE_SIZE && 
                           (head0 & (FULL_TAPE_SIZE-1)) < SINGLE_TAPE_SIZE) {
                    p1_received_copy = true;
                }
                ops++;
                break;
            case 0x5B:  // '['
                if (tape[head0 & (FULL_TAPE_SIZE-1)] == 0) {
                    int depth = 1;
                    pos++;
                    while (pos < FULL_TAPE_SIZE && depth > 0) {
                        if (tape[pos] == 0x5D) depth--;  // ']'
                        if (tape[pos] == 0x5B) depth++;  // '['
                        pos++;
                    }
                    pos--;
                    if (depth != 0) pos = FULL_TAPE_SIZE;
                }
                ops++;
                break;
            case 0x5D:  // ']'
                if (tape[head0 & (FULL_TAPE_SIZE-1)] != 0) {
                    int depth = 1;
                    pos--;
                    while (pos >= 0 && depth > 0) {
                        if (tape[pos] == 0x5D) depth++;  // ']'
                        if (tape[pos] == 0x5B) depth--;  // '['
                        pos--;
                    }
                    pos++;
                    if (depth != 0) pos = -1;
                }
                ops++;
                break;
        }
        
        if (pos < 0) break;
        pos++;
        if (pos >= FULL_TAPE_SIZE) break;
    }
    
    // Update energy states
    bool p1_stays_dead = false;
    bool p2_stays_dead = false;
    
    if (energy_enabled) {
        // P1 energy update
        unsigned int p1_reserve = get_reserve(p1_state);
        unsigned int p1_timer = get_timer(p1_state);
        bool p1_dead = p1_was_dead;
        
        if (p1_in_zone) {
            p1_reserve = reserve_duration;
            p1_timer = 0;
        } else if (p1_received_copy) {
            p1_reserve = p2_in_zone ? reserve_duration : get_reserve(p2_state);
            p1_timer = 0;
            p1_dead = false;
        } else {
            if (p1_reserve > 0) p1_reserve--;
            if (!p1_dead) p1_timer++;
            // death_timer = 0 means infinite (never dies)
            if (death_timer > 0 && p1_timer > death_timer && !p1_dead) {
                p1_dead = true;
            }
        }
        p1_stays_dead = p1_was_dead && p1_dead;
        energy_state[p1_abs] = pack_state(p1_reserve, p1_timer, p1_dead);
        
        // P2 energy update
        unsigned int p2_reserve = get_reserve(p2_state);
        unsigned int p2_timer = get_timer(p2_state);
        bool p2_dead = p2_was_dead;
        
        if (p2_in_zone) {
            p2_reserve = reserve_duration;
            p2_timer = 0;
        } else if (p2_received_copy) {
            p2_reserve = p1_in_zone ? reserve_duration : get_reserve(p1_state);
            p2_timer = 0;
            p2_dead = false;
        } else {
            if (p2_reserve > 0) p2_reserve--;
            if (!p2_dead) p2_timer++;
            if (death_timer > 0 && p2_timer > death_timer && !p2_dead) {
                p2_dead = true;
            }
        }
        p2_stays_dead = p2_was_dead && p2_dead;
        energy_state[p2_abs] = pack_state(p2_reserve, p2_timer, p2_dead);
    }
    
    // Write back soup
    if (p1_stays_dead) {
        for (int i = 0; i < SINGLE_TAPE_SIZE; i++) {
            soup[p1_byte_offset + i] = 0;
        }
    } else {
        for (int i = 0; i < SINGLE_TAPE_SIZE; i++) {
            soup[p1_byte_offset + i] = tape[i];
        }
    }
    
    if (p2_stays_dead) {
        for (int i = 0; i < SINGLE_TAPE_SIZE; i++) {
            soup[p2_byte_offset + i] = 0;
        }
    } else {
        for (int i = 0; i < SINGLE_TAPE_SIZE; i++) {
            soup[p2_byte_offset + i] = tape[SINGLE_TAPE_SIZE + i];
        }
    }
    
    // Atomic add to total ops
    atomicAdd(ops_count, (unsigned long long)ops);
}
"#;

/// CUDA-based multi-simulation
#[cfg(feature = "cuda")]
pub struct CudaMultiSimulation {
    device: Arc<CudaDevice>,
    soup_gpu: CudaSlice<u8>,
    pairs_gpu: CudaSlice<u32>,
    energy_state_gpu: CudaSlice<u32>,
    sim_configs_gpu: CudaSlice<u32>,
    energy_map_gpu: CudaSlice<u32>,
    ops_count_gpu: CudaSlice<u64>,
    kernel: CudaFunction,
    // Config
    num_sims: usize,
    num_programs: usize,
    num_pairs: usize,
    grid_width: usize,
    grid_height: usize,
    steps_per_run: u32,
    mutation_prob: u32,
    seed: u64,
    epoch: u64,
    energy_enabled: bool,
}

#[cfg(feature = "cuda")]
impl CudaMultiSimulation {
    /// Create a new CUDA multi-simulation
    /// 
    /// Unlike wgpu, CUDA has no 4GB buffer limit - you can use your full GPU memory.
    pub fn new(
        num_sims: usize,
        num_programs: usize,
        grid_width: usize,
        grid_height: usize,
        seed: u64,
        mutation_prob: u32,
        steps_per_run: u32,
        energy_config: Option<&crate::energy::EnergyConfig>,
        per_sim_configs: Option<Vec<(u32, u32)>>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize CUDA
        let device = CudaDevice::new(0)?;
        
        // Note: cudarc 0.12 doesn't expose device properties directly
        // We'll just print a simple message
        println!("CUDA Device: Initialized successfully");
        
        // Calculate memory requirements
        let total_programs = num_sims * num_programs;
        let soup_size = total_programs * 64;
        let energy_size = total_programs * 4;
        let pairs_size = (num_programs / 2) * 2 * 4;
        let sim_configs_size = num_sims * 2 * 4;
        let energy_map_size = ((total_programs + 31) / 32) * 4;
        
        let total_required = soup_size + energy_size + pairs_size + sim_configs_size + energy_map_size;
        println!("  Memory required: {:.2} GB", total_required as f64 / 1e9);
        println!("  Total programs: {} ({} sims × {} programs/sim)", total_programs, num_sims, num_programs);
        
        // Compile kernel using nvrtc
        let ptx = cudarc::nvrtc::compile_ptx(BFF_CUDA_KERNEL)?;
        device.load_ptx(ptx, "bff", &["bff_batched_evaluate"])?;
        let kernel = device.get_func("bff", "bff_batched_evaluate").unwrap();
        
        // Initialize data on CPU first
        let num_pairs = num_programs / 2;
        let energy_enabled = energy_config.map(|c| c.enabled).unwrap_or(false);
        let default_death = energy_config.map(|c| c.interaction_death).unwrap_or(10);
        let default_reserve = energy_config.map(|c| c.reserve_duration).unwrap_or(5);
        
        // Pairs (same for all sims - local indices)
        let pairs: Vec<u32> = generate_pairs(num_programs)
            .into_iter()
            .flat_map(|(a, b)| [a as u32, b as u32])
            .collect();
        
        // Per-sim configs
        let sim_configs: Vec<u32> = match per_sim_configs {
            Some(configs) if !configs.is_empty() => {
                (0..num_sims)
                    .flat_map(|i| {
                        let (death, reserve) = configs[i % configs.len()];
                        [death, reserve]
                    })
                    .collect()
            }
            _ => {
                (0..num_sims)
                    .flat_map(|_| [default_death, default_reserve])
                    .collect()
            }
        };
        
        // Energy map (all zeros for now)
        let energy_map = vec![0u32; (total_programs + 31) / 32];
        
        // Soup with random data
        use rand::Rng;
        let mut rng = rand::rng();
        let soup: Vec<u8> = (0..soup_size).map(|_| rng.random()).collect();
        
        
        // Energy states (all alive with full reserve)
        let packed_initial_state = (default_reserve.min(65535) & 0xFFFF) as u32;
        let energy_states: Vec<u32> = vec![packed_initial_state; total_programs];
        
        // Allocate AND initialize GPU buffers using htod_sync_copy (copies data in one step)
        let soup_gpu = device.htod_sync_copy(&soup)?;
        let pairs_gpu = device.htod_sync_copy(&pairs)?;
        let energy_state_gpu = device.htod_sync_copy(&energy_states)?;
        let sim_configs_gpu = device.htod_sync_copy(&sim_configs)?;
        let energy_map_gpu = device.htod_sync_copy(&energy_map)?;
        let ops_count_gpu = device.alloc_zeros::<u64>(1)?;
        
        Ok(Self {
            device,
            soup_gpu,
            pairs_gpu,
            energy_state_gpu,
            sim_configs_gpu,
            energy_map_gpu,
            ops_count_gpu,
            kernel,
            num_sims,
            num_programs,
            num_pairs,
            grid_width,
            grid_height,
            steps_per_run,
            mutation_prob,
            seed,
            epoch: 0,
            energy_enabled,
        })
    }
    
    /// Run one epoch across all simulations
    pub fn step(&mut self) -> u64 {
        // Calculate grid dimensions
        let total_pairs = self.num_pairs * self.num_sims;
        let block_size = 256u32;
        let grid_size = ((total_pairs as u32) + block_size - 1) / block_size;
        
        
        // Launch kernel with 11 params (cudarc limit is 12)
        // Pack some u32 params together into u64 values for the kernel
        let params_packed1 = ((self.num_pairs as u64) << 32) | (self.num_programs as u64);
        let params_packed2 = ((self.num_sims as u64) << 32) | (self.steps_per_run as u64);
        let params_packed3 = ((self.mutation_prob as u64) << 32) | (if self.energy_enabled { 1u64 } else { 0u64 });
        
        let cfg = LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (grid_size, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            self.kernel.clone().launch(cfg, (
                &self.soup_gpu,
                &self.pairs_gpu,
                &self.energy_state_gpu,
                &self.sim_configs_gpu,
                &self.energy_map_gpu,
                &self.ops_count_gpu,
                params_packed1,
                params_packed2,
                params_packed3,
                self.seed,
                self.epoch,
            )).expect("Kernel launch failed");
        }
        
        // Sync and check for errors (kernel errors are caught here)
        match self.device.synchronize() {
            Ok(()) => {}
            Err(e) => {
                eprintln!("CUDA sync error at epoch {}: {:?}", self.epoch, e);
                return 0;
            }
        }
        
        self.epoch += 1;
        
        // Read ops count
        let mut ops = [0u64];
        self.device.dtoh_sync_copy_into(&self.ops_count_gpu, &mut ops).unwrap();
        
        
        // Reset counter for next epoch using memset
        if let Err(e) = self.device.memset_zeros(&mut self.ops_count_gpu) {
            eprintln!("Failed to reset ops counter: {:?}", e);
        }
        
        ops[0]
    }
    
    /// Get soup data for a specific simulation
    pub fn get_sim_soup(&self, sim_idx: usize) -> Vec<u8> {
        let offset = sim_idx * self.num_programs * 64;
        let size = self.num_programs * 64;
        
        let mut data = vec![0u8; size];
        // Note: cudarc requires slicing for partial reads
        self.device.dtoh_sync_copy_into(
            &self.soup_gpu.slice(offset..offset + size),
            &mut data
        ).unwrap();
        data
    }
    
    /// Get all soup data
    pub fn get_all_soup(&self) -> Vec<u8> {
        let size = self.num_sims * self.num_programs * 64;
        let mut data = vec![0u8; size];
        self.device.dtoh_sync_copy_into(&self.soup_gpu, &mut data).unwrap();
        data
    }
    
    pub fn num_sims(&self) -> usize { self.num_sims }
    pub fn num_programs(&self) -> usize { self.num_programs }
    pub fn grid_width(&self) -> usize { self.grid_width }
    pub fn grid_height(&self) -> usize { self.grid_height }
    pub fn epoch(&self) -> u64 { self.epoch }
}

/// Check if CUDA is available
#[cfg(feature = "cuda")]
pub fn cuda_available() -> bool {
    CudaDevice::new(0).is_ok()
}

#[cfg(not(feature = "cuda"))]
pub fn cuda_available() -> bool {
    false
}

/// Print CUDA device info
#[cfg(feature = "cuda")]
pub fn print_cuda_info() {
    match CudaDevice::new(0) {
        Ok(_device) => {
            println!("CUDA Device: Available and initialized");
            // Note: cudarc 0.12 has limited device property access
        }
        Err(e) => {
            println!("CUDA not available: {}", e);
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub fn print_cuda_info() {
    println!("CUDA: Not compiled with CUDA support");
    println!("  To enable: cargo build --release --features cuda");
}

