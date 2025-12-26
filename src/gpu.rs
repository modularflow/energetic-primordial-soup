/// GPU-accelerated simulation using CUDA
/// 
/// This module provides GPU acceleration for the BFF simulation using cudarc.
/// The simulation runs entirely on the GPU, with only periodic reads back to CPU
/// for statistics and visualization.

#[cfg(feature = "cuda")]
pub mod cuda {
    use cudarc::driver::*;
    use cudarc::nvrtc::Ptx;
    use std::sync::Arc;

    /// CUDA kernel source for BFF evaluation
    const BFF_KERNEL: &str = r#"
extern "C" __global__ void bff_evaluate(
    unsigned char* soup,           // All programs concatenated
    const unsigned int* pair_indices, // Pairs: [p1_0, p2_0, p1_1, p2_1, ...]
    unsigned long long* ops_count, // Atomic counter for total ops
    unsigned int num_pairs,
    unsigned int steps_per_run,
    unsigned int mutation_prob,
    unsigned long long seed,
    unsigned long long epoch
) {
    const int SINGLE_TAPE_SIZE = 64;
    const int FULL_TAPE_SIZE = 128;
    
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= num_pairs) return;
    
    // Get program indices for this pair
    unsigned int p1_idx = pair_indices[pair_idx * 2];
    unsigned int p2_idx = pair_indices[pair_idx * 2 + 1];
    
    // Local tape (128 bytes)
    unsigned char tape[FULL_TAPE_SIZE];
    
    // Copy programs to local tape
    for (int i = 0; i < SINGLE_TAPE_SIZE; i++) {
        tape[i] = soup[p1_idx * SINGLE_TAPE_SIZE + i];
        tape[SINGLE_TAPE_SIZE + i] = soup[p2_idx * SINGLE_TAPE_SIZE + i];
    }
    
    // SplitMix64 for deterministic mutations
    auto splitmix = [](unsigned long long z) -> unsigned long long {
        z += 0x9e3779b97f4a7c15ULL;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        return z ^ (z >> 31);
    };
    
    // Apply mutations
    unsigned long long mut_seed = splitmix(seed + epoch * num_pairs + pair_idx);
    for (int i = 0; i < FULL_TAPE_SIZE; i++) {
        unsigned long long rng = splitmix(mut_seed + i);
        unsigned char replacement = rng & 0xFF;
        unsigned int prob = (rng >> 8) & ((1U << 30) - 1);
        if (prob < mutation_prob) {
            tape[i] = replacement;
        }
    }
    
    // BFF Evaluation
    int pos = 2;
    int head0 = tape[0] % FULL_TAPE_SIZE;
    int head1 = tape[1] % FULL_TAPE_SIZE;
    int nskip = 0;
    
    for (unsigned int step = 0; step < steps_per_run; step++) {
        head0 = head0 & (FULL_TAPE_SIZE - 1);
        head1 = head1 & (FULL_TAPE_SIZE - 1);
        
        unsigned char cmd = tape[pos];
        
        switch (cmd) {
            case '<': head0--; break;
            case '>': head0++; break;
            case '{': head1--; break;
            case '}': head1++; break;
            case '+': tape[head0 & (FULL_TAPE_SIZE-1)]++; break;
            case '-': tape[head0 & (FULL_TAPE_SIZE-1)]--; break;
            case '.': tape[head1 & (FULL_TAPE_SIZE-1)] = tape[head0 & (FULL_TAPE_SIZE-1)]; break;
            case ',': tape[head0 & (FULL_TAPE_SIZE-1)] = tape[head1 & (FULL_TAPE_SIZE-1)]; break;
            case '[':
                if (tape[head0 & (FULL_TAPE_SIZE-1)] == 0) {
                    int depth = 1;
                    pos++;
                    while (pos < FULL_TAPE_SIZE && depth > 0) {
                        if (tape[pos] == ']') depth--;
                        if (tape[pos] == '[') depth++;
                        pos++;
                    }
                    pos--;
                    if (depth != 0) pos = FULL_TAPE_SIZE;
                }
                break;
            case ']':
                if (tape[head0 & (FULL_TAPE_SIZE-1)] != 0) {
                    int depth = 1;
                    pos--;
                    while (pos >= 0 && depth > 0) {
                        if (tape[pos] == ']') depth++;
                        if (tape[pos] == '[') depth--;
                        pos--;
                    }
                    pos++;
                    if (depth != 0) pos = -1;
                }
                break;
            default:
                nskip++;
        }
        
        if (pos < 0) { step++; break; }
        pos++;
        if (pos >= FULL_TAPE_SIZE) { step++; break; }
    }
    
    // Copy results back to soup
    for (int i = 0; i < SINGLE_TAPE_SIZE; i++) {
        soup[p1_idx * SINGLE_TAPE_SIZE + i] = tape[i];
        soup[p2_idx * SINGLE_TAPE_SIZE + i] = tape[SINGLE_TAPE_SIZE + i];
    }
    
    // Atomic add to total ops (approximate)
    atomicAdd(ops_count, (unsigned long long)(steps_per_run - nskip));
}
"#;

    /// GPU Simulation state
    pub struct GpuSimulation {
        device: Arc<CudaDevice>,
        soup_gpu: CudaSlice<u8>,
        pairs_gpu: CudaSlice<u32>,
        ops_count_gpu: CudaSlice<u64>,
        kernel: CudaFunction,
        num_programs: usize,
        num_pairs: usize,
        steps_per_run: u32,
        mutation_prob: u32,
        seed: u64,
        epoch: u64,
    }

    impl GpuSimulation {
        /// Create a new GPU simulation
        pub fn new(
            num_programs: usize,
            seed: u64,
            mutation_prob: u32,
            steps_per_run: u32,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            // Initialize CUDA
            let device = CudaDevice::new(0)?;
            
            // Compile kernel
            let ptx = cudarc::nvrtc::compile_ptx(BFF_KERNEL)?;
            device.load_ptx(ptx, "bff", &["bff_evaluate"])?;
            let kernel = device.get_func("bff", "bff_evaluate").unwrap();
            
            // Allocate GPU memory
            let soup_size = num_programs * 64;
            let soup_gpu = device.alloc_zeros::<u8>(soup_size)?;
            
            let num_pairs = num_programs / 2;
            let pairs_gpu = device.alloc_zeros::<u32>(num_pairs * 2)?;
            let ops_count_gpu = device.alloc_zeros::<u64>(1)?;
            
            Ok(Self {
                device,
                soup_gpu,
                pairs_gpu,
                ops_count_gpu,
                kernel,
                num_programs,
                num_pairs,
                steps_per_run,
                mutation_prob,
                seed,
                epoch: 0,
            })
        }
        
        /// Initialize soup with random data
        pub fn init_random(&mut self) -> Result<(), Box<dyn std::error::Error>> {
            let mut soup = vec![0u8; self.num_programs * 64];
            for (i, byte) in soup.iter_mut().enumerate() {
                let seed = crate::simulation::split_mix_64(self.seed + i as u64);
                *byte = (seed % 256) as u8;
            }
            self.device.htod_copy_into(soup, &mut self.soup_gpu)?;
            Ok(())
        }
        
        /// Set pair indices for this epoch
        pub fn set_pairs(&mut self, pairs: &[(u32, u32)]) -> Result<(), Box<dyn std::error::Error>> {
            let flat: Vec<u32> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
            self.device.htod_copy_into(flat, &mut self.pairs_gpu)?;
            self.num_pairs = pairs.len();
            Ok(())
        }
        
        /// Run one epoch on GPU
        pub fn run_epoch(&mut self) -> Result<u64, Box<dyn std::error::Error>> {
            // Reset ops counter
            self.device.htod_copy_into(vec![0u64], &mut self.ops_count_gpu)?;
            
            // Launch kernel
            let block_size = 256;
            let grid_size = (self.num_pairs + block_size - 1) / block_size;
            
            unsafe {
                self.kernel.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_size as u32, 1, 1),
                        block_dim: (block_size as u32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &mut self.soup_gpu,
                        &self.pairs_gpu,
                        &mut self.ops_count_gpu,
                        self.num_pairs as u32,
                        self.steps_per_run,
                        self.mutation_prob,
                        self.seed,
                        self.epoch,
                    ),
                )?;
            }
            
            self.device.synchronize()?;
            
            // Read ops count
            let ops: Vec<u64> = self.device.dtoh_sync_copy(&self.ops_count_gpu)?;
            
            self.epoch += 1;
            Ok(ops[0])
        }
        
        /// Copy soup back to CPU
        pub fn get_soup(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
            Ok(self.device.dtoh_sync_copy(&self.soup_gpu)?)
        }
        
        /// Get current epoch
        pub fn epoch(&self) -> u64 {
            self.epoch
        }
    }
}

/// Check if CUDA is available
#[cfg(feature = "cuda")]
pub fn cuda_available() -> bool {
    cudarc::driver::CudaDevice::new(0).is_ok()
}

#[cfg(not(feature = "cuda"))]
pub fn cuda_available() -> bool {
    false
}

/// Print GPU info
#[cfg(feature = "cuda")]
pub fn print_gpu_info() {
    if let Ok(_device) = cudarc::driver::CudaDevice::new(0) {
        println!("CUDA Device: Available");
        println!("  Memory: Check nvidia-smi for details");
    } else {
        println!("CUDA Device: Not available");
    }
}

#[cfg(not(feature = "cuda"))]
pub fn print_gpu_info() {
    println!("CUDA: Not compiled with CUDA support");
    println!("  To enable: cargo build --release --features cuda");
}

// ============================================================================
// WGPU (Vulkan/WebGPU) Implementation
// ============================================================================

#[cfg(feature = "wgpu-compute")]
pub mod wgpu_sim {
    use std::borrow::Cow;
    use bytemuck::{Pod, Zeroable};
    
    /// BFF compute shader in WGSL with energy system support (single sim)
    const BFF_SHADER: &str = r#"
// Constants
const SINGLE_TAPE_SIZE: u32 = 64u;
const FULL_TAPE_SIZE: u32 = 128u;

struct Params {
    num_pairs: u32,
    steps_per_run: u32,
    mutation_prob: u32,
    grid_width: u32,
    seed_lo: u32,
    seed_hi: u32,
    epoch_lo: u32,
    epoch_hi: u32,
}

struct EnergyParams {
    enabled: u32,
    num_sources: u32,
    radius: u32,
    reserve_duration: u32,
    death_timer: u32,
    _pad: u32,
    // Up to 8 source positions (x, y pairs)
    src0_x: u32, src0_y: u32,
    src1_x: u32, src1_y: u32,
    src2_x: u32, src2_y: u32,
    src3_x: u32, src3_y: u32,
    src4_x: u32, src4_y: u32,
    src5_x: u32, src5_y: u32,
    src6_x: u32, src6_y: u32,
    src7_x: u32, src7_y: u32,
}

@group(0) @binding(0) var<storage, read_write> soup: array<u32>;
@group(0) @binding(1) var<storage, read> pairs: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> ops_count: atomic<u32>;
@group(0) @binding(4) var<uniform> energy_params: EnergyParams;
@group(0) @binding(5) var<storage, read_write> energy_state: array<u32>; // packed: reserve(8) | timer(8) | dead(8) | unused(8)

// Simple 32-bit LCG for mutations (faster than 64-bit splitmix on GPU)
fn lcg(seed: u32) -> u32 {
    return seed * 1664525u + 1013904223u;
}

// Check distance squared helper
fn dist_sq(x: u32, y: u32, sx: u32, sy: u32) -> u32 {
    let dx = i32(x) - i32(sx);
    let dy = i32(y) - i32(sy);
    return u32(dx * dx + dy * dy);
}

// Check if position is within any energy zone
fn in_energy_zone(prog_idx: u32) -> bool {
    if (energy_params.enabled == 0u) {
        return true; // Energy disabled = everywhere is energized
    }
    
    let x = prog_idx % params.grid_width;
    let y = prog_idx / params.grid_width;
    let radius_sq = energy_params.radius * energy_params.radius;
    let n = energy_params.num_sources;
    
    // Check distance to each source (up to 8)
    if (n >= 1u && dist_sq(x, y, energy_params.src0_x, energy_params.src0_y) <= radius_sq) { return true; }
    if (n >= 2u && dist_sq(x, y, energy_params.src1_x, energy_params.src1_y) <= radius_sq) { return true; }
    if (n >= 3u && dist_sq(x, y, energy_params.src2_x, energy_params.src2_y) <= radius_sq) { return true; }
    if (n >= 4u && dist_sq(x, y, energy_params.src3_x, energy_params.src3_y) <= radius_sq) { return true; }
    if (n >= 5u && dist_sq(x, y, energy_params.src4_x, energy_params.src4_y) <= radius_sq) { return true; }
    if (n >= 6u && dist_sq(x, y, energy_params.src5_x, energy_params.src5_y) <= radius_sq) { return true; }
    if (n >= 7u && dist_sq(x, y, energy_params.src6_x, energy_params.src6_y) <= radius_sq) { return true; }
    if (n >= 8u && dist_sq(x, y, energy_params.src7_x, energy_params.src7_y) <= radius_sq) { return true; }
    
    return false;
}

// Get energy state components
fn get_reserve(state: u32) -> u32 { return state & 0xFFu; }
fn get_timer(state: u32) -> u32 { return (state >> 8u) & 0xFFu; }
fn is_dead(state: u32) -> bool { return ((state >> 16u) & 0xFFu) != 0u; }

// Pack energy state
fn pack_state(reserve: u32, timer: u32, dead: bool) -> u32 {
    return (reserve & 0xFFu) | ((timer & 0xFFu) << 8u) | (select(0u, 1u, dead) << 16u);
}

// Check if program can mutate
fn can_mutate(prog_idx: u32) -> bool {
    if (energy_params.enabled == 0u) {
        return true;
    }
    let state = energy_state[prog_idx];
    if (is_dead(state)) {
        return false;
    }
    return in_energy_zone(prog_idx) || get_reserve(state) > 0u;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_idx = global_id.x;
    if (pair_idx >= params.num_pairs) {
        return;
    }
    
    let p1_idx = pairs[pair_idx * 2u];
    let p2_idx = pairs[pair_idx * 2u + 1u];
    
    // Load energy states
    var p1_state = energy_state[p1_idx];
    var p2_state = energy_state[p2_idx];
    let p1_in_zone = in_energy_zone(p1_idx);
    let p2_in_zone = in_energy_zone(p2_idx);
    let p1_can_mutate = can_mutate(p1_idx);
    let p2_can_mutate = can_mutate(p2_idx);
    
    // Track if cross-program copies occur
    var p1_received_copy = false;
    var p2_received_copy = false;
    
    // Local tape storage (128 bytes as 32 u32s)
    var tape: array<u32, 32>;
    
    // Copy programs to local tape
    let p1_base = p1_idx * (SINGLE_TAPE_SIZE / 4u);
    let p2_base = p2_idx * (SINGLE_TAPE_SIZE / 4u);
    for (var i = 0u; i < 16u; i++) {
        tape[i] = soup[p1_base + i];
        tape[i + 16u] = soup[p2_base + i];
    }
    
    // Apply mutations using simple LCG - only to programs that can mutate
    var rng_state = params.seed_lo ^ params.epoch_lo ^ (pair_idx * 0x9E3779B9u);
    
    // Mutate first half (p1) if allowed
    if (p1_can_mutate) {
        for (var i = 0u; i < 16u; i++) {
            rng_state = lcg(rng_state);
            let prob = rng_state & 0x3FFFFFFFu;
            if (prob < params.mutation_prob) {
                rng_state = lcg(rng_state);
                tape[i] = rng_state;
            }
        }
    } else {
        // Still advance RNG to keep determinism
        for (var i = 0u; i < 16u; i++) {
            rng_state = lcg(rng_state);
            if ((rng_state & 0x3FFFFFFFu) < params.mutation_prob) {
                rng_state = lcg(rng_state);
            }
        }
    }
    
    // Mutate second half (p2) if allowed
    if (p2_can_mutate) {
        for (var i = 16u; i < 32u; i++) {
            rng_state = lcg(rng_state);
            let prob = rng_state & 0x3FFFFFFFu;
            if (prob < params.mutation_prob) {
                rng_state = lcg(rng_state);
                tape[i] = rng_state;
            }
        }
    } else {
        for (var i = 16u; i < 32u; i++) {
            rng_state = lcg(rng_state);
            if ((rng_state & 0x3FFFFFFFu) < params.mutation_prob) {
                rng_state = lcg(rng_state);
            }
        }
    }
    
    // BFF evaluation with inlined byte access
    var pos: i32 = 2;
    
    // Read head positions from tape bytes 0 and 1
    var head0: i32 = i32((tape[0] & 0xFFu) % FULL_TAPE_SIZE);
    var head1: i32 = i32(((tape[0] >> 8u) & 0xFFu) % FULL_TAPE_SIZE);
    var nskip: u32 = 0u;
    
    for (var step = 0u; step < params.steps_per_run; step++) {
        head0 = head0 & i32(FULL_TAPE_SIZE - 1u);
        head1 = head1 & i32(FULL_TAPE_SIZE - 1u);
        
        // Read command byte
        let cmd_word_idx = u32(pos) / 4u;
        let cmd_byte_offset = (u32(pos) % 4u) * 8u;
        let cmd = (tape[cmd_word_idx] >> cmd_byte_offset) & 0xFFu;
        
        switch (cmd) {
            case 60u: { head0 -= 1; } // '<'
            case 62u: { head0 += 1; } // '>'
            case 123u: { head1 -= 1; } // '{'
            case 125u: { head1 += 1; } // '}'
            case 43u: { // '+' 
                let idx = u32(head0) & (FULL_TAPE_SIZE - 1u);
                let word_idx = idx / 4u;
                let byte_off = (idx % 4u) * 8u;
                let val = ((tape[word_idx] >> byte_off) & 0xFFu) + 1u;
                let mask = ~(0xFFu << byte_off);
                tape[word_idx] = (tape[word_idx] & mask) | ((val & 0xFFu) << byte_off);
            }
            case 45u: { // '-'
                let idx = u32(head0) & (FULL_TAPE_SIZE - 1u);
                let word_idx = idx / 4u;
                let byte_off = (idx % 4u) * 8u;
                let val = ((tape[word_idx] >> byte_off) & 0xFFu) - 1u;
                let mask = ~(0xFFu << byte_off);
                tape[word_idx] = (tape[word_idx] & mask) | ((val & 0xFFu) << byte_off);
            }
            case 46u: { // '.' - copy from head0 to head1
                let src_idx = u32(head0) & (FULL_TAPE_SIZE - 1u);
                let dst_idx = u32(head1) & (FULL_TAPE_SIZE - 1u);
                let src_word = src_idx / 4u;
                let src_off = (src_idx % 4u) * 8u;
                let val = (tape[src_word] >> src_off) & 0xFFu;
                let dst_word = dst_idx / 4u;
                let dst_off = (dst_idx % 4u) * 8u;
                let mask = ~(0xFFu << dst_off);
                tape[dst_word] = (tape[dst_word] & mask) | (val << dst_off);
                
                // Track cross-program copies (src in one half, dst in other)
                let src_half = select(0u, 1u, src_idx >= SINGLE_TAPE_SIZE);
                let dst_half = select(0u, 1u, dst_idx >= SINGLE_TAPE_SIZE);
                if (src_half != dst_half) {
                    if (dst_half == 0u) { p1_received_copy = true; }
                    else { p2_received_copy = true; }
                }
            }
            case 44u: { // ',' - copy from head1 to head0
                let src_idx = u32(head1) & (FULL_TAPE_SIZE - 1u);
                let dst_idx = u32(head0) & (FULL_TAPE_SIZE - 1u);
                let src_word = src_idx / 4u;
                let src_off = (src_idx % 4u) * 8u;
                let val = (tape[src_word] >> src_off) & 0xFFu;
                let dst_word = dst_idx / 4u;
                let dst_off = (dst_idx % 4u) * 8u;
                let mask = ~(0xFFu << dst_off);
                tape[dst_word] = (tape[dst_word] & mask) | (val << dst_off);
                
                // Track cross-program copies
                let src_half = select(0u, 1u, src_idx >= SINGLE_TAPE_SIZE);
                let dst_half = select(0u, 1u, dst_idx >= SINGLE_TAPE_SIZE);
                if (src_half != dst_half) {
                    if (dst_half == 0u) { p1_received_copy = true; }
                    else { p2_received_copy = true; }
                }
            }
            case 91u: { // '['
                let h_idx = u32(head0) & (FULL_TAPE_SIZE - 1u);
                let h_word = h_idx / 4u;
                let h_off = (h_idx % 4u) * 8u;
                let h_val = (tape[h_word] >> h_off) & 0xFFu;
                if (h_val == 0u) {
                    var depth: i32 = 1;
                    pos += 1;
                    loop {
                        if (pos >= i32(FULL_TAPE_SIZE) || depth <= 0) { break; }
                        let c_word = u32(pos) / 4u;
                        let c_off = (u32(pos) % 4u) * 8u;
                        let c = (tape[c_word] >> c_off) & 0xFFu;
                        if (c == 93u) { depth -= 1; }
                        if (c == 91u) { depth += 1; }
                        pos += 1;
                    }
                    pos -= 1;
                    if (depth != 0) { pos = i32(FULL_TAPE_SIZE); }
                }
            }
            case 93u: { // ']'
                let h_idx = u32(head0) & (FULL_TAPE_SIZE - 1u);
                let h_word = h_idx / 4u;
                let h_off = (h_idx % 4u) * 8u;
                let h_val = (tape[h_word] >> h_off) & 0xFFu;
                if (h_val != 0u) {
                    var depth: i32 = 1;
                    pos -= 1;
                    loop {
                        if (pos < 0 || depth <= 0) { break; }
                        let c_word = u32(pos) / 4u;
                        let c_off = (u32(pos) % 4u) * 8u;
                        let c = (tape[c_word] >> c_off) & 0xFFu;
                        if (c == 93u) { depth += 1; }
                        if (c == 91u) { depth -= 1; }
                        pos -= 1;
                    }
                    pos += 1;
                    if (depth != 0) { pos = -1; }
                }
            }
            default: { nskip += 1u; }
        }
        
        if (pos < 0) { break; }
        pos += 1;
        if (pos >= i32(FULL_TAPE_SIZE)) { break; }
    }
    
    // Copy results back to global soup
    for (var i = 0u; i < 16u; i++) {
        soup[p1_base + i] = tape[i];
        soup[p2_base + i] = tape[i + 16u];
    }
    
    // Update energy states if enabled
    if (energy_params.enabled != 0u) {
        // Process p1 energy
        var p1_reserve = get_reserve(p1_state);
        var p1_timer = get_timer(p1_state);
        var p1_dead = is_dead(p1_state);
        
        if (p1_in_zone) {
            // In zone: full reserve, reset timer
            p1_reserve = energy_params.reserve_duration;
            p1_timer = 0u;
        } else {
            // Outside zone
            if (p1_received_copy) {
                // Received copy: inherit energy from partner (p2), reset timer
                let p2_reserve = select(energy_params.reserve_duration, get_reserve(p2_state), !p2_in_zone);
                p1_reserve = p2_reserve;
                p1_timer = 0u;
                p1_dead = false;
            } else {
                // No interaction: decrement reserve, increment timer
                if (p1_reserve > 0u) { p1_reserve -= 1u; }
                p1_timer += 1u;
                
                // Check death
                if (p1_timer > energy_params.death_timer && !p1_dead) {
                    p1_dead = true;
                    // Zero the tape
                    for (var i = 0u; i < 16u; i++) {
                        soup[p1_base + i] = 0u;
                    }
                }
            }
        }
        energy_state[p1_idx] = pack_state(p1_reserve, p1_timer, p1_dead);
        
        // Process p2 energy
        var p2_reserve = get_reserve(p2_state);
        var p2_timer = get_timer(p2_state);
        var p2_dead = is_dead(p2_state);
        
        if (p2_in_zone) {
            p2_reserve = energy_params.reserve_duration;
            p2_timer = 0u;
        } else {
            if (p2_received_copy) {
                let p1_reserve_for_inherit = select(energy_params.reserve_duration, get_reserve(p1_state), !p1_in_zone);
                p2_reserve = p1_reserve_for_inherit;
                p2_timer = 0u;
                p2_dead = false;
            } else {
                if (p2_reserve > 0u) { p2_reserve -= 1u; }
                p2_timer += 1u;
                
                if (p2_timer > energy_params.death_timer && !p2_dead) {
                    p2_dead = true;
                    for (var i = 0u; i < 16u; i++) {
                        soup[p2_base + i] = 0u;
                    }
                }
            }
        }
        energy_state[p2_idx] = pack_state(p2_reserve, p2_timer, p2_dead);
    }
    
    atomicAdd(&ops_count, params.steps_per_run - nskip);
}
"#;

    /// Batched multi-sim shader: runs N simulations in a single dispatch using global_id.y for sim index
    const BFF_SHADER_BATCHED: &str = r#"
// Constants
const SINGLE_TAPE_SIZE: u32 = 64u;
const FULL_TAPE_SIZE: u32 = 128u;

struct Params {
    num_pairs: u32,
    steps_per_run: u32,
    mutation_prob: u32,
    grid_width: u32,
    seed_lo: u32,
    seed_hi: u32,
    epoch_lo: u32,
    epoch_hi: u32,
    num_programs: u32,  // Programs per simulation
    num_sims: u32,      // Number of simulations (global_id.y range)
    _pad0: u32,
    _pad1: u32,
}

struct EnergyParams {
    enabled: u32,
    num_sources: u32,
    radius: u32,
    reserve_duration: u32,
    death_timer: u32,
    _pad: u32,
    src0_x: u32, src0_y: u32,
    src1_x: u32, src1_y: u32,
    src2_x: u32, src2_y: u32,
    src3_x: u32, src3_y: u32,
    src4_x: u32, src4_y: u32,
    src5_x: u32, src5_y: u32,
    src6_x: u32, src6_y: u32,
    src7_x: u32, src7_y: u32,
}

@group(0) @binding(0) var<storage, read_write> soup: array<u32>;
@group(0) @binding(1) var<storage, read> pairs: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> ops_count: atomic<u32>;
@group(0) @binding(4) var<uniform> energy_params: EnergyParams;
@group(0) @binding(5) var<storage, read_write> energy_state: array<u32>;

fn lcg(seed: u32) -> u32 {
    return seed * 1664525u + 1013904223u;
}

fn dist_sq(x: u32, y: u32, sx: u32, sy: u32) -> u32 {
    let dx = i32(x) - i32(sx);
    let dy = i32(y) - i32(sy);
    return u32(dx * dx + dy * dy);
}

fn in_energy_zone(prog_idx: u32) -> bool {
    if (energy_params.enabled == 0u) { return true; }
    let x = prog_idx % params.grid_width;
    let y = prog_idx / params.grid_width;
    let radius_sq = energy_params.radius * energy_params.radius;
    let n = energy_params.num_sources;
    if (n >= 1u && dist_sq(x, y, energy_params.src0_x, energy_params.src0_y) <= radius_sq) { return true; }
    if (n >= 2u && dist_sq(x, y, energy_params.src1_x, energy_params.src1_y) <= radius_sq) { return true; }
    if (n >= 3u && dist_sq(x, y, energy_params.src2_x, energy_params.src2_y) <= radius_sq) { return true; }
    if (n >= 4u && dist_sq(x, y, energy_params.src3_x, energy_params.src3_y) <= radius_sq) { return true; }
    if (n >= 5u && dist_sq(x, y, energy_params.src4_x, energy_params.src4_y) <= radius_sq) { return true; }
    if (n >= 6u && dist_sq(x, y, energy_params.src5_x, energy_params.src5_y) <= radius_sq) { return true; }
    if (n >= 7u && dist_sq(x, y, energy_params.src6_x, energy_params.src6_y) <= radius_sq) { return true; }
    if (n >= 8u && dist_sq(x, y, energy_params.src7_x, energy_params.src7_y) <= radius_sq) { return true; }
    return false;
}

fn get_reserve(state: u32) -> u32 { return state & 0xFFu; }
fn get_timer(state: u32) -> u32 { return (state >> 8u) & 0xFFu; }
fn is_dead(state: u32) -> bool { return ((state >> 16u) & 0xFFu) != 0u; }
fn pack_state(reserve: u32, timer: u32, dead: bool) -> u32 {
    return (reserve & 0xFFu) | ((timer & 0xFFu) << 8u) | (select(0u, 1u, dead) << 16u);
}

fn can_mutate(prog_idx: u32, sim_offset: u32) -> bool {
    if (energy_params.enabled == 0u) { return true; }
    let state = energy_state[sim_offset + prog_idx];
    if (is_dead(state)) { return false; }
    return in_energy_zone(prog_idx) || get_reserve(state) > 0u;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_idx = global_id.x;
    let sim_idx = global_id.y;
    
    if (pair_idx >= params.num_pairs || sim_idx >= params.num_sims) { return; }
    
    // Offset for this simulation's data in concatenated buffers
    let soup_offset = sim_idx * params.num_programs * (SINGLE_TAPE_SIZE / 4u);
    let energy_offset = sim_idx * params.num_programs;
    
    let p1_idx = pairs[pair_idx * 2u];
    let p2_idx = pairs[pair_idx * 2u + 1u];
    
    // Load energy states (with offset)
    var p1_state = energy_state[energy_offset + p1_idx];
    var p2_state = energy_state[energy_offset + p2_idx];
    let p1_in_zone = in_energy_zone(p1_idx);
    let p2_in_zone = in_energy_zone(p2_idx);
    let p1_can_mutate = can_mutate(p1_idx, energy_offset);
    let p2_can_mutate = can_mutate(p2_idx, energy_offset);
    
    var p1_received_copy = false;
    var p2_received_copy = false;
    var tape: array<u32, 32>;
    
    // Load soup (with offset)
    let p1_base = soup_offset + p1_idx * (SINGLE_TAPE_SIZE / 4u);
    let p2_base = soup_offset + p2_idx * (SINGLE_TAPE_SIZE / 4u);
    for (var i = 0u; i < 16u; i++) {
        tape[i] = soup[p1_base + i];
        tape[i + 16u] = soup[p2_base + i];
    }
    
    // Use sim_idx to differentiate RNG per simulation
    var rng_state = params.seed_lo ^ params.epoch_lo ^ (pair_idx * 0x9E3779B9u) ^ (sim_idx * 0x85EBCA6Bu);
    
    if (p1_can_mutate) {
        for (var i = 0u; i < 16u; i++) {
            rng_state = lcg(rng_state);
            if ((rng_state & 0x3FFFFFFFu) < params.mutation_prob) {
                rng_state = lcg(rng_state);
                tape[i] = rng_state;
            }
        }
    } else {
        for (var i = 0u; i < 16u; i++) {
            rng_state = lcg(rng_state);
            if ((rng_state & 0x3FFFFFFFu) < params.mutation_prob) { rng_state = lcg(rng_state); }
        }
    }
    
    if (p2_can_mutate) {
        for (var i = 16u; i < 32u; i++) {
            rng_state = lcg(rng_state);
            if ((rng_state & 0x3FFFFFFFu) < params.mutation_prob) {
                rng_state = lcg(rng_state);
                tape[i] = rng_state;
            }
        }
    } else {
        for (var i = 16u; i < 32u; i++) {
            rng_state = lcg(rng_state);
            if ((rng_state & 0x3FFFFFFFu) < params.mutation_prob) { rng_state = lcg(rng_state); }
        }
    }
    
    // BFF evaluation
    var pos: i32 = 2;
    var head0: i32 = i32((tape[0] & 0xFFu) % FULL_TAPE_SIZE);
    var head1: i32 = i32(((tape[0] >> 8u) & 0xFFu) % FULL_TAPE_SIZE);
    var nskip: u32 = 0u;
    
    for (var step = 0u; step < params.steps_per_run; step++) {
        head0 = head0 & i32(FULL_TAPE_SIZE - 1u);
        head1 = head1 & i32(FULL_TAPE_SIZE - 1u);
        let cmd_word_idx = u32(pos) / 4u;
        let cmd_byte_offset = (u32(pos) % 4u) * 8u;
        let cmd = (tape[cmd_word_idx] >> cmd_byte_offset) & 0xFFu;
        
        switch (cmd) {
            case 60u: { head0 -= 1; }
            case 62u: { head0 += 1; }
            case 123u: { head1 -= 1; }
            case 125u: { head1 += 1; }
            case 43u: {
                let idx = u32(head0) & (FULL_TAPE_SIZE - 1u);
                let word_idx = idx / 4u;
                let byte_off = (idx % 4u) * 8u;
                let val = ((tape[word_idx] >> byte_off) & 0xFFu) + 1u;
                let mask = ~(0xFFu << byte_off);
                tape[word_idx] = (tape[word_idx] & mask) | ((val & 0xFFu) << byte_off);
            }
            case 45u: {
                let idx = u32(head0) & (FULL_TAPE_SIZE - 1u);
                let word_idx = idx / 4u;
                let byte_off = (idx % 4u) * 8u;
                let val = ((tape[word_idx] >> byte_off) & 0xFFu) - 1u;
                let mask = ~(0xFFu << byte_off);
                tape[word_idx] = (tape[word_idx] & mask) | ((val & 0xFFu) << byte_off);
            }
            case 46u: {
                let src_idx = u32(head0) & (FULL_TAPE_SIZE - 1u);
                let dst_idx = u32(head1) & (FULL_TAPE_SIZE - 1u);
                let src_word = src_idx / 4u;
                let src_off = (src_idx % 4u) * 8u;
                let val = (tape[src_word] >> src_off) & 0xFFu;
                let dst_word = dst_idx / 4u;
                let dst_off = (dst_idx % 4u) * 8u;
                let mask = ~(0xFFu << dst_off);
                tape[dst_word] = (tape[dst_word] & mask) | (val << dst_off);
                let src_half = select(0u, 1u, src_idx >= SINGLE_TAPE_SIZE);
                let dst_half = select(0u, 1u, dst_idx >= SINGLE_TAPE_SIZE);
                if (src_half != dst_half) {
                    if (dst_half == 0u) { p1_received_copy = true; } else { p2_received_copy = true; }
                }
            }
            case 44u: {
                let src_idx = u32(head1) & (FULL_TAPE_SIZE - 1u);
                let dst_idx = u32(head0) & (FULL_TAPE_SIZE - 1u);
                let src_word = src_idx / 4u;
                let src_off = (src_idx % 4u) * 8u;
                let val = (tape[src_word] >> src_off) & 0xFFu;
                let dst_word = dst_idx / 4u;
                let dst_off = (dst_idx % 4u) * 8u;
                let mask = ~(0xFFu << dst_off);
                tape[dst_word] = (tape[dst_word] & mask) | (val << dst_off);
                let src_half = select(0u, 1u, src_idx >= SINGLE_TAPE_SIZE);
                let dst_half = select(0u, 1u, dst_idx >= SINGLE_TAPE_SIZE);
                if (src_half != dst_half) {
                    if (dst_half == 0u) { p1_received_copy = true; } else { p2_received_copy = true; }
                }
            }
            case 91u: {
                let h_idx = u32(head0) & (FULL_TAPE_SIZE - 1u);
                let h_word = h_idx / 4u;
                let h_off = (h_idx % 4u) * 8u;
                if (((tape[h_word] >> h_off) & 0xFFu) == 0u) {
                    var depth: i32 = 1;
                    pos += 1;
                    loop {
                        if (pos >= i32(FULL_TAPE_SIZE) || depth <= 0) { break; }
                        let c_word = u32(pos) / 4u;
                        let c_off = (u32(pos) % 4u) * 8u;
                        let c = (tape[c_word] >> c_off) & 0xFFu;
                        if (c == 93u) { depth -= 1; }
                        if (c == 91u) { depth += 1; }
                        pos += 1;
                    }
                    pos -= 1;
                    if (depth != 0) { pos = i32(FULL_TAPE_SIZE); }
                }
            }
            case 93u: {
                let h_idx = u32(head0) & (FULL_TAPE_SIZE - 1u);
                let h_word = h_idx / 4u;
                let h_off = (h_idx % 4u) * 8u;
                if (((tape[h_word] >> h_off) & 0xFFu) != 0u) {
                    var depth: i32 = 1;
                    pos -= 1;
                    loop {
                        if (pos < 0 || depth <= 0) { break; }
                        let c_word = u32(pos) / 4u;
                        let c_off = (u32(pos) % 4u) * 8u;
                        let c = (tape[c_word] >> c_off) & 0xFFu;
                        if (c == 93u) { depth += 1; }
                        if (c == 91u) { depth -= 1; }
                        pos -= 1;
                    }
                    pos += 1;
                    if (depth != 0) { pos = -1; }
                }
            }
            default: { nskip += 1u; }
        }
        if (pos < 0) { break; }
        pos += 1;
        if (pos >= i32(FULL_TAPE_SIZE)) { break; }
    }
    
    // Write back soup
    for (var i = 0u; i < 16u; i++) {
        soup[p1_base + i] = tape[i];
        soup[p2_base + i] = tape[i + 16u];
    }
    
    // Update energy states
    if (energy_params.enabled != 0u) {
        var p1_reserve = get_reserve(p1_state);
        var p1_timer = get_timer(p1_state);
        var p1_dead = is_dead(p1_state);
        if (p1_in_zone) {
            p1_reserve = energy_params.reserve_duration;
            p1_timer = 0u;
        } else if (p1_received_copy) {
            let p2_res = select(energy_params.reserve_duration, get_reserve(p2_state), !p2_in_zone);
            p1_reserve = p2_res;
            p1_timer = 0u;
            p1_dead = false;
        } else {
            if (p1_reserve > 0u) { p1_reserve -= 1u; }
            p1_timer += 1u;
            if (p1_timer > energy_params.death_timer && !p1_dead) {
                p1_dead = true;
                for (var i = 0u; i < 16u; i++) { soup[p1_base + i] = 0u; }
            }
        }
        energy_state[energy_offset + p1_idx] = pack_state(p1_reserve, p1_timer, p1_dead);
        
        var p2_reserve = get_reserve(p2_state);
        var p2_timer = get_timer(p2_state);
        var p2_dead = is_dead(p2_state);
        if (p2_in_zone) {
            p2_reserve = energy_params.reserve_duration;
            p2_timer = 0u;
        } else if (p2_received_copy) {
            let p1_res = select(energy_params.reserve_duration, get_reserve(p1_state), !p1_in_zone);
            p2_reserve = p1_res;
            p2_timer = 0u;
            p2_dead = false;
        } else {
            if (p2_reserve > 0u) { p2_reserve -= 1u; }
            p2_timer += 1u;
            if (p2_timer > energy_params.death_timer && !p2_dead) {
                p2_dead = true;
                for (var i = 0u; i < 16u; i++) { soup[p2_base + i] = 0u; }
            }
        }
        energy_state[energy_offset + p2_idx] = pack_state(p2_reserve, p2_timer, p2_dead);
    }
    
    atomicAdd(&ops_count, params.steps_per_run - nskip);
}
"#;

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct Params {
        num_pairs: u32,
        steps_per_run: u32,
        mutation_prob: u32,
        grid_width: u32,
        seed_lo: u32,
        seed_hi: u32,
        epoch_lo: u32,
        epoch_hi: u32,
    }
    
    /// Params for batched multi-sim shader
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct BatchedParams {
        num_pairs: u32,
        steps_per_run: u32,
        mutation_prob: u32,
        grid_width: u32,
        seed_lo: u32,
        seed_hi: u32,
        epoch_lo: u32,
        epoch_hi: u32,
        num_programs: u32,
        num_sims: u32,
        _pad0: u32,
        _pad1: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct EnergyParams {
        enabled: u32,
        num_sources: u32,
        radius: u32,
        reserve_duration: u32,
        death_timer: u32,
        _pad: u32,
        // Up to 8 sources
        src0_x: u32, src0_y: u32,
        src1_x: u32, src1_y: u32,
        src2_x: u32, src2_y: u32,
        src3_x: u32, src3_y: u32,
        src4_x: u32, src4_y: u32,
        src5_x: u32, src5_y: u32,
        src6_x: u32, src6_y: u32,
        src7_x: u32, src7_y: u32,
    }

    impl EnergyParams {
        fn disabled() -> Self {
            Self {
                enabled: 0,
                num_sources: 0,
                radius: 0,
                reserve_duration: 5,
                death_timer: 10,
                _pad: 0,
                src0_x: 0, src0_y: 0,
                src1_x: 0, src1_y: 0,
                src2_x: 0, src2_y: 0,
                src3_x: 0, src3_y: 0,
                src4_x: 0, src4_y: 0,
                src5_x: 0, src5_y: 0,
                src6_x: 0, src6_y: 0,
                src7_x: 0, src7_y: 0,
            }
        }

        fn from_config(config: &crate::energy::EnergyConfig, _grid_width: usize, _grid_height: usize) -> Self {
            if !config.enabled || config.sources.is_empty() {
                return Self::disabled();
            }

            // Get up to 8 sources
            let get_src = |i: usize| -> (u32, u32) {
                if i < config.sources.len() {
                    (config.sources[i].x as u32, config.sources[i].y as u32)
                } else {
                    (0, 0)
                }
            };

            let (src0_x, src0_y) = get_src(0);
            let (src1_x, src1_y) = get_src(1);
            let (src2_x, src2_y) = get_src(2);
            let (src3_x, src3_y) = get_src(3);
            let (src4_x, src4_y) = get_src(4);
            let (src5_x, src5_y) = get_src(5);
            let (src6_x, src6_y) = get_src(6);
            let (src7_x, src7_y) = get_src(7);

            Self {
                enabled: 1,
                num_sources: config.sources.len().min(8) as u32,
                radius: config.sources.get(0).map(|s| s.radius).unwrap_or(64) as u32,
                reserve_duration: config.reserve_duration as u32,
                death_timer: config.interaction_death as u32,
                _pad: 0,
                src0_x, src0_y,
                src1_x, src1_y,
                src2_x, src2_y,
                src3_x, src3_y,
                src4_x, src4_y,
                src5_x, src5_y,
                src6_x, src6_y,
                src7_x, src7_y,
            }
        }
    }

    pub struct WgpuSimulation {
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline: wgpu::ComputePipeline,
        soup_buffer: wgpu::Buffer,
        pairs_buffer: wgpu::Buffer,
        params_buffer: wgpu::Buffer,
        ops_buffer: wgpu::Buffer,
        energy_params_buffer: wgpu::Buffer,
        energy_state_buffer: wgpu::Buffer,
        staging_buffer: wgpu::Buffer,
        bind_group: wgpu::BindGroup,
        num_programs: usize,
        num_pairs: usize,
        grid_width: usize,
        steps_per_run: u32,
        mutation_prob: u32,
        seed: u64,
        epoch: u64,
        energy_params: EnergyParams,
    }

    impl WgpuSimulation {
        pub fn new(
            num_programs: usize,
            grid_width: usize,
            grid_height: usize,
            seed: u64,
            mutation_prob: u32,
            steps_per_run: u32,
            energy_config: Option<&crate::energy::EnergyConfig>,
        ) -> Option<Self> {
            pollster::block_on(Self::new_async(
                num_programs, grid_width, grid_height, seed, mutation_prob, steps_per_run, energy_config
            ))
        }

        async fn new_async(
            num_programs: usize,
            grid_width: usize,
            grid_height: usize,
            seed: u64,
            mutation_prob: u32,
            steps_per_run: u32,
            energy_config: Option<&crate::energy::EnergyConfig>,
        ) -> Option<Self> {
            let energy_params = energy_config
                .map(|c| EnergyParams::from_config(c, grid_width, grid_height))
                .unwrap_or_else(EnergyParams::disabled);
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
                ..Default::default()
            });

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await?;

            println!("GPU Adapter: {:?}", adapter.get_info().name);

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("BFF Simulation"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                    },
                    None,
                )
                .await
                .ok()?;

            // Create shader module
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("BFF Shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(BFF_SHADER)),
            });

            // Create buffers
            let soup_size = (num_programs * 64) as u64;
            let soup_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Soup"),
                size: soup_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let num_pairs = num_programs / 2;
            let pairs_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Pairs"),
                size: (num_pairs * 2 * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Params"),
                size: std::mem::size_of::<Params>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let ops_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Ops"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let energy_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("EnergyParams"),
                size: std::mem::size_of::<EnergyParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Energy state: one u32 per program (packed: reserve | timer | dead | unused)
            let energy_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("EnergyState"),
                size: (num_programs * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging"),
                size: soup_size.max(4),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // Create bind group layout and bind group
            let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("BFF Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("BFF Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: soup_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: pairs_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: ops_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: energy_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: energy_state_buffer.as_entire_binding(),
                    },
                ],
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("BFF Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("BFF Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: Default::default(),
            });

            Some(Self {
                device,
                queue,
                pipeline,
                soup_buffer,
                pairs_buffer,
                params_buffer,
                ops_buffer,
                energy_params_buffer,
                energy_state_buffer,
                staging_buffer,
                bind_group,
                num_programs,
                num_pairs,
                grid_width,
                steps_per_run,
                mutation_prob,
                seed,
                epoch: 0,
                energy_params,
            })
        }

        pub fn init_random(&self) {
            let mut soup = vec![0u8; self.num_programs * 64];
            for (i, byte) in soup.iter_mut().enumerate() {
                let seed = crate::simulation::split_mix_64(self.seed + i as u64);
                *byte = (seed % 256) as u8;
            }
            self.queue.write_buffer(&self.soup_buffer, 0, &soup);
            
            // Initialize energy state: all programs start alive with full reserve if in zone
            // Format: reserve(8) | timer(8) | dead(8) | unused(8)
            let energy_state: Vec<u32> = (0..self.num_programs)
                .map(|i| {
                    // Start with reserve = reserve_duration, timer = 0, dead = false
                    self.energy_params.reserve_duration & 0xFF
                })
                .collect();
            self.queue.write_buffer(&self.energy_state_buffer, 0, bytemuck::cast_slice(&energy_state));
            
            // Upload energy params
            self.queue.write_buffer(&self.energy_params_buffer, 0, bytemuck::bytes_of(&self.energy_params));
        }

        pub fn set_pairs(&self, pairs: &[(u32, u32)]) {
            let flat: Vec<u32> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
            self.queue.write_buffer(&self.pairs_buffer, 0, bytemuck::cast_slice(&flat));
        }

        pub fn run_epoch(&mut self) -> u64 {
            // Update params
            let params = Params {
                num_pairs: self.num_pairs as u32,
                steps_per_run: self.steps_per_run,
                mutation_prob: self.mutation_prob,
                grid_width: self.grid_width as u32,
                seed_lo: self.seed as u32,
                seed_hi: (self.seed >> 32) as u32,
                epoch_lo: self.epoch as u32,
                epoch_hi: (self.epoch >> 32) as u32,
            };
            self.queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
            
            // Reset ops counter
            self.queue.write_buffer(&self.ops_buffer, 0, &[0u8; 4]);
            
            // Create command encoder
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("BFF Compute"),
            });
            
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("BFF Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(((self.num_pairs + 255) / 256) as u32, 1, 1);
            }
            
            self.queue.submit(Some(encoder.finish()));
            self.device.poll(wgpu::Maintain::Wait);
            
            self.epoch += 1;
            
            // Return approximate ops (we can't easily read atomic counters back)
            (self.num_pairs * self.steps_per_run as usize) as u64
        }
        
        /// Update energy params from a potentially changed config
        /// Call this when sources have spawned/expired in dynamic mode
        pub fn update_energy_config(&mut self, config: &crate::energy::EnergyConfig) {
            self.energy_params = EnergyParams::from_config(config, self.grid_width, self.grid_width);
            self.queue.write_buffer(&self.energy_params_buffer, 0, bytemuck::bytes_of(&self.energy_params));
        }

        pub fn get_soup(&self) -> Vec<u8> {
            let size = self.num_programs * 64;
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&self.soup_buffer, 0, &self.staging_buffer, 0, size as u64);
            self.queue.submit(Some(encoder.finish()));
            
            let slice = self.staging_buffer.slice(..size as u64);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            self.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            
            let data = slice.get_mapped_range();
            let result = data.to_vec();
            drop(data);
            self.staging_buffer.unmap();
            result
        }

        pub fn epoch(&self) -> u64 {
            self.epoch
        }
        
        /// Check if all programs are dead (no mutations possible)
        /// Returns (alive_count, can_mutate_count)
        pub fn check_alive_status(&self) -> (usize, usize) {
            if self.energy_params.enabled == 0 {
                // Energy disabled, all programs are always alive
                return (self.num_programs, self.num_programs);
            }
            
            // Read energy state buffer
            let size = self.num_programs * 4; // u32 per program
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Energy State Staging"),
                size: size as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&self.energy_state_buffer, 0, &staging, 0, size as u64);
            self.queue.submit(Some(encoder.finish()));
            
            let slice = staging.slice(..size as u64);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            self.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            
            let data = slice.get_mapped_range();
            let states: &[u32] = bytemuck::cast_slice(&data);
            
            let mut alive = 0;
            let mut can_mutate = 0;
            for &state in states {
                let is_dead = ((state >> 16) & 0xFF) != 0;
                let reserve = state & 0xFF;
                if !is_dead {
                    alive += 1;
                    if reserve > 0 {
                        can_mutate += 1;
                    }
                }
            }
            
            drop(data);
            staging.unmap();
            
            (alive, can_mutate)
        }
        
        /// Quick check if simulation should terminate (all dead, no activity)
        pub fn is_all_dead(&self) -> bool {
            if self.energy_params.enabled == 0 {
                return false; // Energy disabled
            }
            let (alive, _) = self.check_alive_status();
            alive == 0
        }
    }

    /// Batched multi-simulation: runs N simulations in a SINGLE dispatch using global_id.y
    pub struct MultiWgpuSimulation {
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline: wgpu::ComputePipeline,
        // Concatenated buffers (all sims in one buffer)
        soup_buffer: wgpu::Buffer,        // size = num_programs * 64 * num_sims
        energy_state_buffer: wgpu::Buffer, // size = num_programs * 4 * num_sims
        pairs_buffer: wgpu::Buffer,
        params_buffer: wgpu::Buffer,
        ops_buffer: wgpu::Buffer,
        energy_params_buffer: wgpu::Buffer,
        staging_buffer: wgpu::Buffer,
        bind_group: wgpu::BindGroup,
        // Config
        num_sims: usize,
        num_programs: usize,
        num_pairs: usize,
        grid_width: usize,
        steps_per_run: u32,
        mutation_prob: u32,
        base_seed: u64,
        epoch: u64,
        energy_params: EnergyParams,
    }

    impl MultiWgpuSimulation {
        /// Create N simulations with a single dispatch (true SIMD parallelism)
        pub fn new(
            num_sims: usize,
            num_programs: usize,
            grid_width: usize,
            grid_height: usize,
            base_seed: u64,
            mutation_prob: u32,
            steps_per_run: u32,
            energy_config: Option<&crate::energy::EnergyConfig>,
        ) -> Option<Self> {
            pollster::block_on(Self::new_async(
                num_sims, num_programs, grid_width, grid_height, base_seed,
                mutation_prob, steps_per_run, energy_config
            ))
        }

        async fn new_async(
            num_sims: usize,
            num_programs: usize,
            grid_width: usize,
            grid_height: usize,
            base_seed: u64,
            mutation_prob: u32,
            steps_per_run: u32,
            energy_config: Option<&crate::energy::EnergyConfig>,
        ) -> Option<Self> {
            if num_sims == 0 {
                return None;
            }

            let energy_params = energy_config
                .map(|c| EnergyParams::from_config(c, grid_width, grid_height))
                .unwrap_or_else(EnergyParams::disabled);

            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
                ..Default::default()
            });

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await?;

            println!("GPU Adapter: {:?} (batched {} simulations)", adapter.get_info().name, num_sims);

            // Calculate required buffer size for the limits
            let soup_size_total = (num_programs * 64 * num_sims) as u32;
            let energy_size_total = (num_programs * 4 * num_sims) as u32;
            let max_buffer_size = soup_size_total.max(energy_size_total).max(1 << 30); // At least 1GB

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("BFF Batched Multi Simulation"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits {
                            max_storage_buffer_binding_size: max_buffer_size,
                            max_buffer_size: max_buffer_size as u64,
                            ..Default::default()
                        },
                    },
                    None,
                )
                .await
                .ok()?;

            // Use the BATCHED shader
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("BFF Batched Shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(BFF_SHADER_BATCHED)),
            });

            let num_pairs = num_programs / 2;
            let soup_size_per_sim = (num_programs * 64) as u64;
            let total_soup_size = soup_size_per_sim * num_sims as u64;
            let energy_size_per_sim = (num_programs * 4) as u64;
            let total_energy_size = energy_size_per_sim * num_sims as u64;

            // Single concatenated soup buffer for all simulations
            let soup_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Batched Soup"),
                size: total_soup_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            // Single concatenated energy state buffer
            let energy_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Batched EnergyState"),
                size: total_energy_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let pairs_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Pairs"),
                size: (num_pairs * 2 * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("BatchedParams"),
                size: std::mem::size_of::<BatchedParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let ops_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Ops"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let energy_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("EnergyParams"),
                size: std::mem::size_of::<EnergyParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging"),
                size: soup_size_per_sim,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("BFF Batched Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("BFF Batched Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("BFF Batched Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: Default::default(),
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("BFF Batched Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: soup_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: pairs_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: ops_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: energy_params_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: energy_state_buffer.as_entire_binding() },
                ],
            });

            queue.write_buffer(&energy_params_buffer, 0, bytemuck::bytes_of(&energy_params));

            Some(Self {
                device,
                queue,
                pipeline,
                soup_buffer,
                energy_state_buffer,
                pairs_buffer,
                params_buffer,
                ops_buffer,
                energy_params_buffer,
                staging_buffer,
                bind_group,
                num_sims,
                num_programs,
                num_pairs,
                grid_width,
                steps_per_run,
                mutation_prob,
                base_seed,
                epoch: 0,
                energy_params,
            })
        }

        pub fn num_sims(&self) -> usize {
            self.num_sims
        }

        /// Initialize all simulations with random data
        pub fn init_random_all(&mut self) {
            use rand::Rng;
            let mut rng = rand::rng();
            
            // Initialize all soups at once
            let total_size = self.num_programs * 64 * self.num_sims;
            let mut data = vec![0u8; total_size];
            rng.fill(&mut data[..]);
            self.queue.write_buffer(&self.soup_buffer, 0, &data);
            
            // Initialize all energy states
            let total_energy = self.num_programs * self.num_sims;
            let energy_state = vec![0u32; total_energy];
            self.queue.write_buffer(&self.energy_state_buffer, 0, bytemuck::cast_slice(&energy_state));
            
            self.epoch = 0;
        }

        /// Set pairs (shared by all simulations)
        pub fn set_pairs_all(&self, pairs: &[(u32, u32)]) {
            let flat: Vec<u32> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
            self.queue.write_buffer(&self.pairs_buffer, 0, bytemuck::cast_slice(&flat));
        }

        /// Run one epoch on ALL simulations in a SINGLE dispatch
        pub fn run_epoch_all(&mut self) -> u64 {
            // Update params once (shared by all sims, sim_idx differentiates via global_id.y)
            let params = BatchedParams {
                num_pairs: self.num_pairs as u32,
                steps_per_run: self.steps_per_run,
                mutation_prob: self.mutation_prob,
                grid_width: self.grid_width as u32,
                seed_lo: self.base_seed as u32,
                seed_hi: (self.base_seed >> 32) as u32,
                epoch_lo: self.epoch as u32,
                epoch_hi: (self.epoch >> 32) as u32,
                num_programs: self.num_programs as u32,
                num_sims: self.num_sims as u32,
                _pad0: 0,
                _pad1: 0,
            };
            self.queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
            
            // Single dispatch for ALL simulations
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("BFF Batched Compute"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("BFF Batched Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                // dispatch_workgroups(x, y, z) where:
                // x = ceil(num_pairs / 256) - handles pairs
                // y = num_sims - handles simulations
                pass.dispatch_workgroups(
                    ((self.num_pairs + 255) / 256) as u32,
                    self.num_sims as u32,
                    1
                );
            }

            self.queue.submit(Some(encoder.finish()));
            self.device.poll(wgpu::Maintain::Wait);

            self.epoch += 1;
            (self.num_pairs * self.steps_per_run as usize * self.num_sims) as u64
        }

        /// Update energy config
        pub fn update_energy_config_all(&mut self, config: &crate::energy::EnergyConfig) {
            self.energy_params = EnergyParams::from_config(config, self.grid_width, self.grid_width);
            self.queue.write_buffer(&self.energy_params_buffer, 0, bytemuck::bytes_of(&self.energy_params));
        }

        /// Get soup data from a specific simulation
        pub fn get_soup(&self, sim_idx: usize) -> Vec<u8> {
            let size = self.num_programs * 64;
            let offset = (sim_idx * size) as u64;
            
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&self.soup_buffer, offset, &self.staging_buffer, 0, size as u64);
            self.queue.submit(Some(encoder.finish()));
            
            let slice = self.staging_buffer.slice(..size as u64);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            self.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            
            let data = slice.get_mapped_range();
            let result = data.to_vec();
            drop(data);
            self.staging_buffer.unmap();
            result
        }

        /// Get epoch count
        pub fn epoch(&self) -> u64 {
            self.epoch
        }
    }
}

#[cfg(feature = "wgpu-compute")]
pub fn wgpu_available() -> bool {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
            ..Default::default()
        });
        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .is_some()
    })
}

#[cfg(not(feature = "wgpu-compute"))]
pub fn wgpu_available() -> bool {
    false
}

