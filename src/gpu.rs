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
    spontaneous_rate: u32,  // 1 in N chance for dead tape in zone to spawn (0 = disabled)
    // Up to 8 sources: x, y, shape, radius
    src0_x: u32, src0_y: u32, src0_shape: u32, src0_radius: u32,
    src1_x: u32, src1_y: u32, src1_shape: u32, src1_radius: u32,
    src2_x: u32, src2_y: u32, src2_shape: u32, src2_radius: u32,
    src3_x: u32, src3_y: u32, src3_shape: u32, src3_radius: u32,
    src4_x: u32, src4_y: u32, src4_shape: u32, src4_radius: u32,
    src5_x: u32, src5_y: u32, src5_shape: u32, src5_radius: u32,
    src6_x: u32, src6_y: u32, src6_shape: u32, src6_radius: u32,
    src7_x: u32, src7_y: u32, src7_shape: u32, src7_radius: u32,
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

// Check if point (x,y) is within a source at (sx,sy) with given shape and radius
// Shape IDs: 0=Circle, 1=StripH, 2=StripV, 3=HalfTop, 4=HalfBottom, 5=HalfLeft, 6=HalfRight, 7=EllipseH, 8=EllipseV
fn in_source(x: i32, y: i32, sx: u32, sy: u32, shape: u32, radius: u32) -> bool {
    let dx = f32(x) - f32(i32(sx));
    let dy = f32(y) - f32(i32(sy));
    let r = f32(radius);
    let r_sq = r * r;
    let dist_sq = dx * dx + dy * dy;
    
    switch (shape) {
        case 0u: { // Circle
            return dist_sq <= r_sq;
        }
        case 1u: { // Strip Horizontal (width=2r, height=r/2)
            return abs(dx) <= r && abs(dy) <= r / 4.0;
        }
        case 2u: { // Strip Vertical (width=r/2, height=2r)
            return abs(dx) <= r / 4.0 && abs(dy) <= r;
        }
        case 3u: { // Half Circle Top
            return dy <= 0.0 && dist_sq <= r_sq;
        }
        case 4u: { // Half Circle Bottom
            return dy >= 0.0 && dist_sq <= r_sq;
        }
        case 5u: { // Half Circle Left
            return dx <= 0.0 && dist_sq <= r_sq;
        }
        case 6u: { // Half Circle Right
            return dx >= 0.0 && dist_sq <= r_sq;
        }
        case 7u: { // Ellipse Horizontal (width=2r, height=r)
            let norm = (dx / r) * (dx / r) + (dy / (r / 2.0)) * (dy / (r / 2.0));
            return norm <= 1.0;
        }
        case 8u: { // Ellipse Vertical (width=r, height=2r)
            let norm = (dx / (r / 2.0)) * (dx / (r / 2.0)) + (dy / r) * (dy / r);
            return norm <= 1.0;
        }
        default: {
            return dist_sq <= r_sq; // Default to circle
        }
    }
}

// Check if position is within any energy zone
fn in_energy_zone(prog_idx: u32) -> bool {
    if (energy_params.enabled == 0u) {
        return true; // Energy disabled = everywhere is energized
    }
    
    let x = i32(prog_idx % params.grid_width);
    let y = i32(prog_idx / params.grid_width);
    let n = energy_params.num_sources;
    
    // Check each source with its shape
    if (n >= 1u && in_source(x, y, energy_params.src0_x, energy_params.src0_y, energy_params.src0_shape, energy_params.src0_radius)) { return true; }
    if (n >= 2u && in_source(x, y, energy_params.src1_x, energy_params.src1_y, energy_params.src1_shape, energy_params.src1_radius)) { return true; }
    if (n >= 3u && in_source(x, y, energy_params.src2_x, energy_params.src2_y, energy_params.src2_shape, energy_params.src2_radius)) { return true; }
    if (n >= 4u && in_source(x, y, energy_params.src3_x, energy_params.src3_y, energy_params.src3_shape, energy_params.src3_radius)) { return true; }
    if (n >= 5u && in_source(x, y, energy_params.src4_x, energy_params.src4_y, energy_params.src4_shape, energy_params.src4_radius)) { return true; }
    if (n >= 6u && in_source(x, y, energy_params.src5_x, energy_params.src5_y, energy_params.src5_shape, energy_params.src5_radius)) { return true; }
    if (n >= 7u && in_source(x, y, energy_params.src6_x, energy_params.src6_y, energy_params.src6_shape, energy_params.src6_radius)) { return true; }
    if (n >= 8u && in_source(x, y, energy_params.src7_x, energy_params.src7_y, energy_params.src7_shape, energy_params.src7_radius)) { return true; }
    
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
    } else {
        // Tape is empty/dead - skip interpreter, count all as skipped
        nskip = params.steps_per_run;
    }
    
    // Track if programs were dead at START of epoch (before any revival)
    let p1_was_dead = is_dead(p1_state);
    let p2_was_dead = is_dead(p2_state);
    
    // Update energy states if enabled (before writing soup back)
    var p1_stays_dead = false;
    var p2_stays_dead = false;
    
    if (energy_params.enabled != 0u) {
        // Process p1 energy
        var p1_reserve = get_reserve(p1_state);
        var p1_timer = get_timer(p1_state);
        var p1_dead = p1_was_dead;
        
        if (p1_in_zone) {
            // In zone: full reserve, reset timer
            p1_reserve = energy_params.reserve_duration;
            p1_timer = 0u;
            // Note: dead programs in zone stay dead until they receive a copy
        } else {
            // Outside zone
            if (p1_received_copy) {
                // Received copy: inherit energy from partner (p2), reset timer
                let p2_reserve = select(energy_params.reserve_duration, get_reserve(p2_state), !p2_in_zone);
                p1_reserve = p2_reserve;
                p1_timer = 0u;
                p1_dead = false;  // REVIVED!
            } else {
                // No interaction: decrement reserve, increment timer
                if (p1_reserve > 0u) { p1_reserve -= 1u; }
                if (!p1_dead) {
                    p1_timer += 1u;
                }
                
                // Check death
                if (p1_timer > energy_params.death_timer && !p1_dead) {
                    p1_dead = true;
                }
            }
        }
        
        // If program was dead and remains dead, keep tape zeroed
        p1_stays_dead = p1_was_dead && p1_dead;
        
        energy_state[p1_idx] = pack_state(p1_reserve, p1_timer, p1_dead);
        
        // Process p2 energy
        var p2_reserve = get_reserve(p2_state);
        var p2_timer = get_timer(p2_state);
        var p2_dead = p2_was_dead;
        
        if (p2_in_zone) {
            p2_reserve = energy_params.reserve_duration;
            p2_timer = 0u;
        } else {
            if (p2_received_copy) {
                let p1_reserve_for_inherit = select(energy_params.reserve_duration, get_reserve(p1_state), !p1_in_zone);
                p2_reserve = p1_reserve_for_inherit;
                p2_timer = 0u;
                p2_dead = false;  // REVIVED!
            } else {
                if (p2_reserve > 0u) { p2_reserve -= 1u; }
                if (!p2_dead) {
                    p2_timer += 1u;
                }
                
                if (p2_timer > energy_params.death_timer && !p2_dead) {
                    p2_dead = true;
                }
            }
        }
        
        p2_stays_dead = p2_was_dead && p2_dead;
        
        energy_state[p2_idx] = pack_state(p2_reserve, p2_timer, p2_dead);
    }
    
    // Copy results back to global soup
    // Dead tapes that weren't revived stay zeroed
    if (p1_stays_dead) {
        for (var i = 0u; i < 16u; i++) {
            soup[p1_base + i] = 0u;
        }
    } else if (is_dead(energy_state[p1_idx])) {
        // Just died this epoch - zero the tape
        for (var i = 0u; i < 16u; i++) {
            soup[p1_base + i] = 0u;
        }
    } else {
        for (var i = 0u; i < 16u; i++) {
            soup[p1_base + i] = tape[i];
        }
    }
    
    if (p2_stays_dead) {
        for (var i = 0u; i < 16u; i++) {
            soup[p2_base + i] = 0u;
        }
    } else if (is_dead(energy_state[p2_idx])) {
        // Just died this epoch - zero the tape
        for (var i = 0u; i < 16u; i++) {
            soup[p2_base + i] = 0u;
        }
    } else {
        for (var i = 0u; i < 16u; i++) {
            soup[p2_base + i] = tape[i + 16u];
        }
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
    num_sims: u32,      // Number of simulations
    mega_mode: u32,     // 0 = normal batched (pairs local), 1 = mega (pairs are absolute indices)
    _pad1: u32,
}

struct EnergyParams {
    enabled: u32,
    num_sources: u32,
    radius: u32,
    reserve_duration: u32,
    death_timer: u32,
    spontaneous_rate: u32,
    src0_x: u32, src0_y: u32, src0_shape: u32, src0_radius: u32,
    src1_x: u32, src1_y: u32, src1_shape: u32, src1_radius: u32,
    src2_x: u32, src2_y: u32, src2_shape: u32, src2_radius: u32,
    src3_x: u32, src3_y: u32, src3_shape: u32, src3_radius: u32,
    src4_x: u32, src4_y: u32, src4_shape: u32, src4_radius: u32,
    src5_x: u32, src5_y: u32, src5_shape: u32, src5_radius: u32,
    src6_x: u32, src6_y: u32, src6_shape: u32, src6_radius: u32,
    src7_x: u32, src7_y: u32, src7_shape: u32, src7_radius: u32,
}

@group(0) @binding(0) var<storage, read_write> soup: array<u32>;
@group(0) @binding(1) var<storage, read> pairs: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> ops_count: atomic<u32>;
@group(0) @binding(4) var<uniform> energy_params: EnergyParams;
@group(0) @binding(5) var<storage, read_write> energy_state: array<u32>;
@group(0) @binding(6) var<storage, read> energy_map: array<u32>;  // Pre-computed energy zone lookup (1 bit per program, packed)

fn lcg(seed: u32) -> u32 {
    return seed * 1664525u + 1013904223u;
}

// Apply mutations using geometric skip (much faster for low mutation rates)
// Instead of checking every byte, we skip directly to mutation positions
// For mutation_prob = 1/4096, this is ~64x faster (0.016 mutations per 64 bytes on average)
fn apply_mutations_sparse(
    tape: ptr<function, array<u32, 32>>,
    start_word: u32,
    end_word: u32,
    rng: ptr<function, u32>,
    mutation_prob: u32
) {
    // Calculate inverse probability for skip calculation
    // mutation_prob is out of 2^30, so inv ≈ 2^30 / mutation_prob
    // For mutation_prob = 2^18, this is 4096
    let inv_prob = (1u << 30u) / max(mutation_prob, 1u);
    
    // Start position in bytes
    var byte_pos = start_word * 4u;
    let end_byte = end_word * 4u;
    
    // Skip to first mutation position using geometric distribution
    // skip ≈ U * inv_prob where U is uniform [0,1) gives us exponential distribution
    *rng = lcg(*rng);
    var skip = ((*rng >> 8u) * inv_prob) >> 22u;
    byte_pos = byte_pos + skip;
    
    // Apply mutations at geometric intervals
    loop {
        if (byte_pos >= end_byte) { break; }
        
        // Apply mutation at this byte position
        *rng = lcg(*rng);
        let word_idx = byte_pos / 4u;
        let byte_offset = (byte_pos % 4u) * 8u;
        let new_byte = (*rng >> 8u) & 0xFFu;
        let mask = ~(0xFFu << byte_offset);
        (*tape)[word_idx] = ((*tape)[word_idx] & mask) | (new_byte << byte_offset);
        
        // Skip to next mutation position (geometric distribution)
        // Add 1 to ensure we always advance at least 1 byte
        *rng = lcg(*rng);
        skip = max(1u, ((*rng >> 8u) * inv_prob) >> 22u);
        byte_pos = byte_pos + skip;
    }
}

// Fast energy zone lookup using pre-computed bitmask
// Returns true if program is in an energy zone
fn in_energy_zone_fast(global_prog_idx: u32) -> bool {
    if (energy_params.enabled == 0u) { return true; }
    // Each u32 holds 32 program flags
    let word_idx = global_prog_idx / 32u;
    let bit_idx = global_prog_idx % 32u;
    return (energy_map[word_idx] & (1u << bit_idx)) != 0u;
}

// Check if point is within source (with shape support) - kept for fallback
fn in_source_batched(x: i32, y: i32, sx: u32, sy: u32, shape: u32, radius: u32) -> bool {
    let dx = f32(x) - f32(i32(sx));
    let dy = f32(y) - f32(i32(sy));
    let r = f32(radius);
    let r_sq = r * r;
    let dist_sq = dx * dx + dy * dy;
    
    switch (shape) {
        case 0u: { return dist_sq <= r_sq; }
        case 1u: { return abs(dx) <= r && abs(dy) <= r / 4.0; }
        case 2u: { return abs(dx) <= r / 4.0 && abs(dy) <= r; }
        case 3u: { return dy <= 0.0 && dist_sq <= r_sq; }
        case 4u: { return dy >= 0.0 && dist_sq <= r_sq; }
        case 5u: { return dx <= 0.0 && dist_sq <= r_sq; }
        case 6u: { return dx >= 0.0 && dist_sq <= r_sq; }
        case 7u: { let norm = (dx/r)*(dx/r) + (dy/(r/2.0))*(dy/(r/2.0)); return norm <= 1.0; }
        case 8u: { let norm = (dx/(r/2.0))*(dx/(r/2.0)) + (dy/r)*(dy/r); return norm <= 1.0; }
        default: { return dist_sq <= r_sq; }
    }
}

// Hash function to generate per-sim offsets for energy sources
fn sim_hash(sim_idx: u32, src_idx: u32) -> u32 {
    var h = sim_idx * 0x9E3779B9u + src_idx * 0x85EBCA6Bu;
    h = h ^ (h >> 16u);
    h = h * 0x21F0AAADu;
    h = h ^ (h >> 15u);
    return h;
}

// Get per-sim offset for a source position
fn source_offset_x(sim_idx: u32, src_idx: u32, base_x: u32) -> u32 {
    let h = sim_hash(sim_idx, src_idx * 2u);
    let offset = i32(h % params.grid_width) - i32(params.grid_width / 2u);
    let new_x = i32(base_x) + offset;
    return u32((new_x + i32(params.grid_width)) % i32(params.grid_width));
}

fn source_offset_y(sim_idx: u32, src_idx: u32, base_y: u32) -> u32 {
    let h = sim_hash(sim_idx, src_idx * 2u + 1u);
    let grid_height = params.num_programs / params.grid_width;
    let offset = i32(h % grid_height) - i32(grid_height / 2u);
    let new_y = i32(base_y) + offset;
    return u32((new_y + i32(grid_height)) % i32(grid_height));
}

// Energy zone check - original geometric calculations (faster than buffer lookup on GPU)
fn in_energy_zone_sim(prog_idx: u32, sim_idx: u32) -> bool {
    if (energy_params.enabled == 0u) { return true; }
    let x = i32(prog_idx % params.grid_width);
    let y = i32(prog_idx / params.grid_width);
    let n = energy_params.num_sources;
    
    // Each source gets a per-sim random offset
    if (n >= 1u && in_source_batched(x, y, source_offset_x(sim_idx, 0u, energy_params.src0_x), source_offset_y(sim_idx, 0u, energy_params.src0_y), energy_params.src0_shape, energy_params.src0_radius)) { return true; }
    if (n >= 2u && in_source_batched(x, y, source_offset_x(sim_idx, 1u, energy_params.src1_x), source_offset_y(sim_idx, 1u, energy_params.src1_y), energy_params.src1_shape, energy_params.src1_radius)) { return true; }
    if (n >= 3u && in_source_batched(x, y, source_offset_x(sim_idx, 2u, energy_params.src2_x), source_offset_y(sim_idx, 2u, energy_params.src2_y), energy_params.src2_shape, energy_params.src2_radius)) { return true; }
    if (n >= 4u && in_source_batched(x, y, source_offset_x(sim_idx, 3u, energy_params.src3_x), source_offset_y(sim_idx, 3u, energy_params.src3_y), energy_params.src3_shape, energy_params.src3_radius)) { return true; }
    if (n >= 5u && in_source_batched(x, y, source_offset_x(sim_idx, 4u, energy_params.src4_x), source_offset_y(sim_idx, 4u, energy_params.src4_y), energy_params.src4_shape, energy_params.src4_radius)) { return true; }
    if (n >= 6u && in_source_batched(x, y, source_offset_x(sim_idx, 5u, energy_params.src5_x), source_offset_y(sim_idx, 5u, energy_params.src5_y), energy_params.src5_shape, energy_params.src5_radius)) { return true; }
    if (n >= 7u && in_source_batched(x, y, source_offset_x(sim_idx, 6u, energy_params.src6_x), source_offset_y(sim_idx, 6u, energy_params.src6_y), energy_params.src6_shape, energy_params.src6_radius)) { return true; }
    if (n >= 8u && in_source_batched(x, y, source_offset_x(sim_idx, 7u, energy_params.src7_x), source_offset_y(sim_idx, 7u, energy_params.src7_y), energy_params.src7_shape, energy_params.src7_radius)) { return true; }
    return false;
}

// Keep original for backwards compatibility (sim_idx = 0)
fn in_energy_zone(prog_idx: u32) -> bool {
    return in_energy_zone_sim(prog_idx, 0u);
}

fn get_reserve(state: u32) -> u32 { return state & 0xFFu; }
fn get_timer(state: u32) -> u32 { return (state >> 8u) & 0xFFu; }
fn is_dead(state: u32) -> bool { return ((state >> 16u) & 0xFFu) != 0u; }
fn pack_state(reserve: u32, timer: u32, dead: bool) -> u32 {
    return (reserve & 0xFFu) | ((timer & 0xFFu) << 8u) | (select(0u, 1u, dead) << 16u);
}

fn can_mutate_sim(prog_idx: u32, sim_offset: u32, sim_idx: u32) -> bool {
    if (energy_params.enabled == 0u) { return true; }
    let state = energy_state[sim_offset + prog_idx];
    if (is_dead(state)) { return false; }
    return in_energy_zone_sim(prog_idx, sim_idx) || get_reserve(state) > 0u;
}

// Fast version using pre-computed energy map bitmask - O(1) lookup
fn can_mutate_fast(global_prog_idx: u32) -> bool {
    if (energy_params.enabled == 0u) { return true; }
    let state = energy_state[global_prog_idx];
    if (is_dead(state)) { return false; }
    return in_energy_zone_fast(global_prog_idx) || get_reserve(state) > 0u;
}

// === WORKGROUP SHARED MEMORY CACHE FOR ENERGY MAP ===
// In normal batched mode, all 256 threads in a workgroup access the same simulation.
// We pre-load energy_map words for that sim into shared memory for faster lookups.
// 256 words = 8192 programs covered (enough for grids up to ~90x90 per sim).
var<workgroup> energy_cache: array<u32, 256>;
var<workgroup> cache_base_word: u32;  // Starting word index of cached region
var<workgroup> cache_valid: u32;      // 1 if cache is populated

// === COMMAND LOOKUP TABLE ===
// Maps byte values (0-255) to operation codes (0-10) for faster dispatch.
// Operation codes: 0=noop, 1=<, 2=>, 3={, 4=}, 5=+, 6=-, 7=., 8=,, 9=[, 10=]
// Using 256 bytes packed into 64 u32s (4 bytes per word).
var<workgroup> cmd_table: array<u32, 64>;

// Lookup using shared cache (call only after cache is populated)
fn in_energy_zone_cached(global_prog_idx: u32) -> bool {
    if (energy_params.enabled == 0u) { return true; }
    let word_idx = global_prog_idx / 32u;
    let bit_idx = global_prog_idx % 32u;
    
    // Check if this word is in our cache
    let cache_offset = word_idx - cache_base_word;
    if (cache_offset < 256u) {
        return (energy_cache[cache_offset] & (1u << bit_idx)) != 0u;
    }
    // Fallback to global memory for out-of-cache accesses
    return (energy_map[word_idx] & (1u << bit_idx)) != 0u;
}

// Fast can_mutate using cached lookup
fn can_mutate_cached(global_prog_idx: u32) -> bool {
    if (energy_params.enabled == 0u) { return true; }
    let state = energy_state[global_prog_idx];
    if (is_dead(state)) { return false; }
    return in_energy_zone_cached(global_prog_idx) || get_reserve(state) > 0u;
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let pair_idx = global_id.x;
    
    // In mega mode: pairs are absolute indices, dispatch y=1
    // In normal mode: pairs are local, dispatch y=num_sims
    var sim_idx = global_id.y;
    
    // === POPULATE ENERGY MAP CACHE ===
    // In normal batched mode, all threads in a workgroup access the same simulation.
    // We cooperatively load energy_map words into shared memory for faster lookups.
    // This must happen BEFORE any early returns to ensure all threads participate in barrier.
    if (params.mega_mode == 0u && energy_params.enabled != 0u) {
        // Calculate the base word index for this simulation's energy region
        let words_per_sim = (params.num_programs + 31u) / 32u;
        let sim_base_word = sim_idx * words_per_sim;
        
        // Thread 0 sets up the cache metadata
        if (local_id.x == 0u) {
            cache_base_word = sim_base_word;
            cache_valid = 1u;
        }
        
        // Each thread loads one word into the cache (if within range)
        if (local_id.x < min(256u, words_per_sim)) {
            energy_cache[local_id.x] = energy_map[sim_base_word + local_id.x];
        }
    }
    
    // === INITIALIZE COMMAND LOOKUP TABLE ===
    // Each thread initializes 4 bytes (one word) of the 256-byte command table.
    // Maps byte values to opcodes: 0=noop, 1=<, 2=>, 3={, 4=}, 5=+, 6=-, 7=., 8=,, 9=[, 10=]
    // Only need 64 threads to fill 64 words (256 bytes).
    if (local_id.x < 64u) {
        let base_byte = local_id.x * 4u;
        var packed: u32 = 0u;
        for (var b = 0u; b < 4u; b++) {
            let byte_val = base_byte + b;
            var opcode: u32 = 0u;  // Default: noop
            if (byte_val == 60u) { opcode = 1u; }       // '<'
            else if (byte_val == 62u) { opcode = 2u; }  // '>'
            else if (byte_val == 123u) { opcode = 3u; } // '{'
            else if (byte_val == 125u) { opcode = 4u; } // '}'
            else if (byte_val == 43u) { opcode = 5u; }  // '+'
            else if (byte_val == 45u) { opcode = 6u; }  // '-'
            else if (byte_val == 46u) { opcode = 7u; }  // '.'
            else if (byte_val == 44u) { opcode = 8u; }  // ','
            else if (byte_val == 91u) { opcode = 9u; }  // '['
            else if (byte_val == 93u) { opcode = 10u; } // ']'
            packed = packed | (opcode << (b * 8u));
        }
        cmd_table[local_id.x] = packed;
    }
    
    // Synchronize - all threads must reach this point
    workgroupBarrier();
    
    // Now we can do early returns after the barrier
    if (params.mega_mode == 0u) {
        // Normal batched mode
        if (pair_idx >= params.num_pairs || sim_idx >= params.num_sims) { return; }
    } else {
        // Mega mode: all pairs in x dimension
        if (pair_idx >= params.num_pairs) { return; }
    }
    
    // Read pair indices
    let raw_p1 = pairs[pair_idx * 2u];
    let raw_p2 = pairs[pair_idx * 2u + 1u];
    
    // In mega mode, pairs are absolute indices; in normal mode, they're local
    var p1_abs: u32;
    var p2_abs: u32;
    var p1_local: u32;
    var p2_local: u32;
    var p1_sim: u32;
    var p2_sim: u32;
    
    if (params.mega_mode == 0u) {
        // Normal mode: add sim offset
        p1_local = raw_p1;
        p2_local = raw_p2;
        p1_sim = sim_idx;
        p2_sim = sim_idx;
        p1_abs = sim_idx * params.num_programs + raw_p1;
        p2_abs = sim_idx * params.num_programs + raw_p2;
    } else {
        // Mega mode: pairs are already absolute
        p1_abs = raw_p1;
        p2_abs = raw_p2;
        p1_sim = raw_p1 / params.num_programs;
        p2_sim = raw_p2 / params.num_programs;
        p1_local = raw_p1 % params.num_programs;
        p2_local = raw_p2 % params.num_programs;
        sim_idx = p1_sim; // Use p1's sim for RNG
    }
    
    // Load energy states using absolute indices
    // Use pre-computed energy map bitmask for O(1) zone lookups instead of expensive geometric math
    var p1_state = energy_state[p1_abs];
    var p2_state = energy_state[p2_abs];
    
    // Use cached lookups in normal mode (cache populated above), fall back to global in mega mode
    var p1_in_zone: bool;
    var p2_in_zone: bool;
    var p1_can_mutate: bool;
    var p2_can_mutate: bool;
    
    if (params.mega_mode == 0u) {
        // Normal mode: use shared memory cache for faster lookups
        p1_in_zone = in_energy_zone_cached(p1_abs);
        p2_in_zone = in_energy_zone_cached(p2_abs);
        p1_can_mutate = can_mutate_cached(p1_abs);
        p2_can_mutate = can_mutate_cached(p2_abs);
    } else {
        // Mega mode: programs may span sims, use global memory
        p1_in_zone = in_energy_zone_fast(p1_abs);
        p2_in_zone = in_energy_zone_fast(p2_abs);
        p1_can_mutate = can_mutate_fast(p1_abs);
        p2_can_mutate = can_mutate_fast(p2_abs);
    }
    
    // === SKIP DEAD PAIRS OPTIMIZATION ===
    // If both programs are dead and neither is in an energy zone (no revival possible),
    // skip the entire tape load, mutation, and BFF evaluation.
    // In late-stage sims with high mortality, this can skip 50-90% of pairs.
    let p1_dead = is_dead(p1_state);
    let p2_dead = is_dead(p2_state);
    if (p1_dead && p2_dead && !p1_in_zone && !p2_in_zone) {
        // Both dead, outside energy zones - nothing can happen, skip entirely
        return;
    }
    
    var p1_received_copy = false;
    var p2_received_copy = false;
    var tape: array<u32, 32>;
    
    // Load soup using absolute positions
    let p1_base = p1_abs * (SINGLE_TAPE_SIZE / 4u);
    let p2_base = p2_abs * (SINGLE_TAPE_SIZE / 4u);
    for (var i = 0u; i < 16u; i++) {
        tape[i] = soup[p1_base + i];
        tape[i + 16u] = soup[p2_base + i];
    }
    
    // Use sim_idx to differentiate RNG per simulation
    var rng_state = params.seed_lo ^ params.epoch_lo ^ (pair_idx * 0x9E3779B9u) ^ (sim_idx * 0x85EBCA6Bu);
    
    // Apply mutations using geometric skip (O(mutations) instead of O(bytes))
    // For mutation_prob = 1/4096 and 64 bytes, we expect ~0.016 mutations on average
    // This skips directly to mutation positions instead of checking every byte
    if (p1_can_mutate) {
        apply_mutations_sparse(&tape, 0u, 16u, &rng_state, params.mutation_prob);
    }
    
    if (p2_can_mutate) {
        apply_mutations_sparse(&tape, 16u, 32u, &rng_state, params.mutation_prob);
    }
    
    // Early exit optimization: check if tape is empty (dead programs)
    // Skip expensive interpreter loop for all-zero tapes
    var tape_active = false;
    for (var i = 0u; i < 32u; i++) {
        if (tape[i] != 0u) { 
            tape_active = true; 
            break; 
        }
    }
    
    // BFF evaluation
    var pos: i32 = 2;
    var head0: i32 = i32((tape[0] & 0xFFu) % FULL_TAPE_SIZE);
    var head1: i32 = i32(((tape[0] >> 8u) & 0xFFu) % FULL_TAPE_SIZE);
    var nskip: u32 = 0u;
    
    // Only run interpreter if tape has content
    if (tape_active) {
    for (var step = 0u; step < params.steps_per_run; step++) {
        head0 = head0 & i32(FULL_TAPE_SIZE - 1u);
        head1 = head1 & i32(FULL_TAPE_SIZE - 1u);
        let cmd_word_idx = u32(pos) / 4u;
        let cmd_byte_offset = (u32(pos) % 4u) * 8u;
        let cmd = (tape[cmd_word_idx] >> cmd_byte_offset) & 0xFFu;
        
        // === COMMAND LOOKUP TABLE ===
        // Look up opcode from shared memory table instead of switching on 256 byte values.
        // Reduces branch divergence by mapping to smaller opcode range (0-10).
        let table_word = cmd / 4u;
        let table_byte = (cmd % 4u) * 8u;
        let opcode = (cmd_table[table_word] >> table_byte) & 0xFFu;
        
        // Switch on opcode (0-10) instead of byte value (0-255)
        // 0=noop, 1=<, 2=>, 3={, 4=}, 5=+, 6=-, 7=., 8=,, 9=[, 10=]
        switch (opcode) {
            case 0u: { nskip += 1u; }  // no-op
            case 1u: { head0 -= 1; }   // '<'
            case 2u: { head0 += 1; }   // '>'
            case 3u: { head1 -= 1; }   // '{'
            case 4u: { head1 += 1; }   // '}'
            case 5u: {  // '+' increment
                let idx = u32(head0) & (FULL_TAPE_SIZE - 1u);
                let word_idx = idx / 4u;
                let byte_off = (idx % 4u) * 8u;
                let val = ((tape[word_idx] >> byte_off) & 0xFFu) + 1u;
                let mask = ~(0xFFu << byte_off);
                tape[word_idx] = (tape[word_idx] & mask) | ((val & 0xFFu) << byte_off);
            }
            case 6u: {  // '-' decrement
                let idx = u32(head0) & (FULL_TAPE_SIZE - 1u);
                let word_idx = idx / 4u;
                let byte_off = (idx % 4u) * 8u;
                let val = ((tape[word_idx] >> byte_off) & 0xFFu) - 1u;
                let mask = ~(0xFFu << byte_off);
                tape[word_idx] = (tape[word_idx] & mask) | ((val & 0xFFu) << byte_off);
            }
            case 7u: {  // '.' copy head0 -> head1
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
            case 8u: {  // ',' copy head1 -> head0
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
            case 9u: {  // '[' loop start
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
            case 10u: {  // ']' loop end
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
    } else {
        // Tape is empty/dead - skip interpreter, count all as skipped
        nskip = params.steps_per_run;
    }
    
    // Track if programs were dead at START of epoch
    let p1_was_dead = is_dead(p1_state);
    let p2_was_dead = is_dead(p2_state);
    
    // Update energy states (before writing soup back)
    var p1_stays_dead = false;
    var p2_stays_dead = false;
    
    if (energy_params.enabled != 0u) {
        var p1_reserve = get_reserve(p1_state);
        var p1_timer = get_timer(p1_state);
        var p1_dead = p1_was_dead;
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
            if (!p1_dead) { p1_timer += 1u; }
            if (p1_timer > energy_params.death_timer && !p1_dead) {
                p1_dead = true;
            }
        }
        p1_stays_dead = p1_was_dead && p1_dead;
        energy_state[p1_abs] = pack_state(p1_reserve, p1_timer, p1_dead);
        
        var p2_reserve = get_reserve(p2_state);
        var p2_timer = get_timer(p2_state);
        var p2_dead = p2_was_dead;
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
            if (!p2_dead) { p2_timer += 1u; }
            if (p2_timer > energy_params.death_timer && !p2_dead) {
                p2_dead = true;
            }
        }
        p2_stays_dead = p2_was_dead && p2_dead;
        energy_state[p2_abs] = pack_state(p2_reserve, p2_timer, p2_dead);
    }
    
    // Write back soup - dead tapes stay zeroed
    if (p1_stays_dead) {
        for (var i = 0u; i < 16u; i++) { soup[p1_base + i] = 0u; }
    } else if (is_dead(energy_state[p1_abs])) {
        for (var i = 0u; i < 16u; i++) { soup[p1_base + i] = 0u; }
    } else {
        for (var i = 0u; i < 16u; i++) { soup[p1_base + i] = tape[i]; }
    }
    
    if (p2_stays_dead) {
        for (var i = 0u; i < 16u; i++) { soup[p2_base + i] = 0u; }
    } else if (is_dead(energy_state[p2_abs])) {
        for (var i = 0u; i < 16u; i++) { soup[p2_base + i] = 0u; }
    } else {
        for (var i = 0u; i < 16u; i++) { soup[p2_base + i] = tape[i + 16u]; }
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
        mega_mode: u32,  // 0 = normal batched, 1 = mega mode with absolute indices
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
        spontaneous_rate: u32,  // 1 in N chance per dead tape in zone (0 = disabled)
        // Up to 8 sources: x, y, shape_id (packed), radius_override
        src0_x: u32, src0_y: u32, src0_shape: u32, src0_radius: u32,
        src1_x: u32, src1_y: u32, src1_shape: u32, src1_radius: u32,
        src2_x: u32, src2_y: u32, src2_shape: u32, src2_radius: u32,
        src3_x: u32, src3_y: u32, src3_shape: u32, src3_radius: u32,
        src4_x: u32, src4_y: u32, src4_shape: u32, src4_radius: u32,
        src5_x: u32, src5_y: u32, src5_shape: u32, src5_radius: u32,
        src6_x: u32, src6_y: u32, src6_shape: u32, src6_radius: u32,
        src7_x: u32, src7_y: u32, src7_shape: u32, src7_radius: u32,
    }

    impl EnergyParams {
        fn disabled() -> Self {
            Self {
                enabled: 0,
                num_sources: 0,
                radius: 0,
                reserve_duration: 5,
                death_timer: 10,
                spontaneous_rate: 0,
                src0_x: 0, src0_y: 0, src0_shape: 0, src0_radius: 64,
                src1_x: 0, src1_y: 0, src1_shape: 0, src1_radius: 64,
                src2_x: 0, src2_y: 0, src2_shape: 0, src2_radius: 64,
                src3_x: 0, src3_y: 0, src3_shape: 0, src3_radius: 64,
                src4_x: 0, src4_y: 0, src4_shape: 0, src4_radius: 64,
                src5_x: 0, src5_y: 0, src5_shape: 0, src5_radius: 64,
                src6_x: 0, src6_y: 0, src6_shape: 0, src6_radius: 64,
                src7_x: 0, src7_y: 0, src7_shape: 0, src7_radius: 64,
            }
        }

        fn from_config(config: &crate::energy::EnergyConfig, _grid_width: usize, _grid_height: usize) -> Self {
            if !config.enabled || config.sources.is_empty() {
                return Self::disabled();
            }

            // Get up to 8 sources with position, shape, and radius
            let get_src = |i: usize| -> (u32, u32, u32, u32) {
                if i < config.sources.len() {
                    let s = &config.sources[i];
                    (s.x as u32, s.y as u32, s.shape.to_gpu_id(), s.radius as u32)
                } else {
                    (0, 0, 0, 64)
                }
            };

            let (src0_x, src0_y, src0_shape, src0_radius) = get_src(0);
            let (src1_x, src1_y, src1_shape, src1_radius) = get_src(1);
            let (src2_x, src2_y, src2_shape, src2_radius) = get_src(2);
            let (src3_x, src3_y, src3_shape, src3_radius) = get_src(3);
            let (src4_x, src4_y, src4_shape, src4_radius) = get_src(4);
            let (src5_x, src5_y, src5_shape, src5_radius) = get_src(5);
            let (src6_x, src6_y, src6_shape, src6_radius) = get_src(6);
            let (src7_x, src7_y, src7_shape, src7_radius) = get_src(7);

            Self {
                enabled: 1,
                num_sources: config.sources.len().min(8) as u32,
                radius: config.sources.get(0).map(|s| s.radius).unwrap_or(64) as u32,
                reserve_duration: config.reserve_duration as u32,
                death_timer: config.interaction_death as u32,
                spontaneous_rate: config.spontaneous_rate,
                src0_x, src0_y, src0_shape, src0_radius,
                src1_x, src1_y, src1_shape, src1_radius,
                src2_x, src2_y, src2_shape, src2_radius,
                src3_x, src3_y, src3_shape, src3_radius,
                src4_x, src4_y, src4_shape, src4_radius,
                src5_x, src5_y, src5_shape, src5_radius,
                src6_x, src6_y, src6_shape, src6_radius,
                src7_x, src7_y, src7_shape, src7_radius,
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
        energy_map_buffer: wgpu::Buffer,  // Pre-computed energy zone bitmask (packed bits)
        pairs_buffer: wgpu::Buffer,
        params_buffer: wgpu::Buffer,
        ops_buffer: wgpu::Buffer,
        energy_params_buffer: wgpu::Buffer,
        staging_buffer: wgpu::Buffer,     // Single-sim staging (legacy)
        // Ping-pong staging for async readback
        staging_soup_a: wgpu::Buffer,     // Full soup staging buffer A
        staging_soup_b: wgpu::Buffer,     // Full soup staging buffer B
        staging_current: bool,            // true = A is being written, B is ready to read
        pending_readback: bool,           // true = there's pending data to read
        bind_group: wgpu::BindGroup,
        // Config
        num_sims: usize,
        num_programs: usize,
        num_pairs: usize,
        grid_width: usize,
        grid_height: usize,
        steps_per_run: u32,
        mutation_prob: u32,
        base_seed: u64,
        epoch: u64,
        energy_params: EnergyParams,
        // Mega-simulation mode
        mega_mode: bool,
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
            
            // In mega mode, pairs can span all sims, so allocate for worst case
            let max_pairs = (num_programs * num_sims / 2) + (grid_width + grid_height) * num_sims;

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
                size: (max_pairs * 2 * 4) as u64,
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

            // Energy map buffer: 1 bit per program across all sims, packed into u32s
            // Size = ceil(num_programs * num_sims / 32) * 4 bytes
            let total_programs = num_programs * num_sims;
            let energy_map_words = (total_programs + 31) / 32;
            let energy_map_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("EnergyMap"),
                size: (energy_map_words * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging"),
                size: soup_size_per_sim,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // Ping-pong staging buffers for async readback (full soup size)
            let staging_soup_a = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Soup A"),
                size: total_soup_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let staging_soup_b = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Soup B"),
                size: total_soup_size,
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                    wgpu::BindGroupEntry { binding: 6, resource: energy_map_buffer.as_entire_binding() },
                ],
            });

            queue.write_buffer(&energy_params_buffer, 0, bytemuck::bytes_of(&energy_params));

            // Compute and upload initial energy map
            let energy_map_data = Self::compute_energy_map_static(
                &energy_params, num_programs, num_sims, grid_width, grid_height
            );
            queue.write_buffer(&energy_map_buffer, 0, bytemuck::cast_slice(&energy_map_data));

            Some(Self {
                device,
                queue,
                pipeline,
                soup_buffer,
                energy_state_buffer,
                energy_map_buffer,
                pairs_buffer,
                params_buffer,
                ops_buffer,
                energy_params_buffer,
                staging_buffer,
                staging_soup_a,
                staging_soup_b,
                staging_current: false, // Start with writing to A
                pending_readback: false,
                bind_group,
                num_sims,
                num_programs,
                num_pairs,
                grid_width,
                grid_height,
                steps_per_run,
                mutation_prob,
                base_seed,
                epoch: 0,
                energy_params,
                mega_mode: false,
            })
        }

        pub fn num_sims(&self) -> usize {
            self.num_sims
        }

        /// Enable mega-simulation mode (pairs use absolute indices)
        pub fn set_mega_mode(&mut self, enabled: bool) {
            self.mega_mode = enabled;
        }

        /// Set pairs for mega mode (absolute indices including cross-sim pairs)
        pub fn set_pairs_mega(&mut self, pairs: &[(u32, u32)]) {
            let flat: Vec<u32> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
            self.queue.write_buffer(&self.pairs_buffer, 0, bytemuck::cast_slice(&flat));
            self.num_pairs = pairs.len();
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
            // In mega mode: all pairs in x dimension, y=1
            // In normal mode: pairs in x, simulations in y
            let (dispatch_x, dispatch_y) = if self.mega_mode {
                // Mega mode: all pairs processed via x dimension
                (((self.num_pairs + 255) / 256) as u32, 1u32)
            } else {
                // Normal batched mode
                (((self.num_pairs + 255) / 256) as u32, self.num_sims as u32)
            };
            
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
                mega_mode: if self.mega_mode { 1 } else { 0 },
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
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            self.queue.submit(Some(encoder.finish()));
            self.device.poll(wgpu::Maintain::Wait);

            self.epoch += 1;
            
            // In mega mode, we process all pairs once; in normal mode, each sim processes its pairs
            let ops_multiplier = if self.mega_mode { 1 } else { self.num_sims };
            (self.num_pairs * self.steps_per_run as usize * ops_multiplier) as u64
        }

        /// Run one epoch without waiting for completion (non-blocking)
        /// Use sync() before reading data back
        pub fn run_epoch_all_async(&mut self) -> u64 {
            let (dispatch_x, dispatch_y) = if self.mega_mode {
                (((self.num_pairs + 255) / 256) as u32, 1u32)
            } else {
                (((self.num_pairs + 255) / 256) as u32, self.num_sims as u32)
            };
            
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
                mega_mode: if self.mega_mode { 1 } else { 0 },
                _pad1: 0,
            };
            self.queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
            
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("BFF Batched Compute Async"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("BFF Batched Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            self.queue.submit(Some(encoder.finish()));
            // Non-blocking poll - just check if there's work to do
            self.device.poll(wgpu::Maintain::Poll);

            self.epoch += 1;
            
            let ops_multiplier = if self.mega_mode { 1 } else { self.num_sims };
            (self.num_pairs * self.steps_per_run as usize * ops_multiplier) as u64
        }

        /// Wait for all pending GPU work to complete
        pub fn sync(&self) {
            self.device.poll(wgpu::Maintain::Wait);
        }

        /// Compute energy zone bitmask on CPU
        /// Returns packed u32s where bit i indicates if program i is in an energy zone
        fn compute_energy_map_static(
            energy_params: &EnergyParams,
            num_programs: usize,
            num_sims: usize,
            grid_width: usize,
            grid_height: usize,
        ) -> Vec<u32> {
            let total_programs = num_programs * num_sims;
            let num_words = (total_programs + 31) / 32;
            let mut map = vec![0u32; num_words];

            // If energy not enabled, all programs are "in zone" (can always mutate)
            if energy_params.enabled == 0 {
                for word in &mut map {
                    *word = 0xFFFFFFFF;
                }
                return map;
            }

            let sources = [
                (energy_params.src0_x, energy_params.src0_y, energy_params.src0_shape, energy_params.src0_radius),
                (energy_params.src1_x, energy_params.src1_y, energy_params.src1_shape, energy_params.src1_radius),
                (energy_params.src2_x, energy_params.src2_y, energy_params.src2_shape, energy_params.src2_radius),
                (energy_params.src3_x, energy_params.src3_y, energy_params.src3_shape, energy_params.src3_radius),
                (energy_params.src4_x, energy_params.src4_y, energy_params.src4_shape, energy_params.src4_radius),
                (energy_params.src5_x, energy_params.src5_y, energy_params.src5_shape, energy_params.src5_radius),
                (energy_params.src6_x, energy_params.src6_y, energy_params.src6_shape, energy_params.src6_radius),
                (energy_params.src7_x, energy_params.src7_y, energy_params.src7_shape, energy_params.src7_radius),
            ];

            for sim_idx in 0..num_sims {
                for prog_idx in 0..num_programs {
                    let x = (prog_idx % grid_width) as i32;
                    let y = (prog_idx / grid_width) as i32;
                    
                    let mut in_zone = false;
                    for src_idx in 0..energy_params.num_sources as usize {
                        let (base_x, base_y, shape, radius) = sources[src_idx];
                        
                        // Compute per-sim offset (matches shader logic)
                        let offset_x = Self::source_offset_x_cpu(sim_idx as u32, src_idx as u32, base_x, grid_width as u32);
                        let offset_y = Self::source_offset_y_cpu(sim_idx as u32, src_idx as u32, base_y, grid_width as u32, grid_height as u32);
                        
                        if Self::in_source_cpu(x, y, offset_x, offset_y, shape, radius) {
                            in_zone = true;
                            break;
                        }
                    }
                    
                    if in_zone {
                        let global_idx = sim_idx * num_programs + prog_idx;
                        let word_idx = global_idx / 32;
                        let bit_idx = global_idx % 32;
                        map[word_idx] |= 1u32 << bit_idx;
                    }
                }
            }
            
            map
        }

        /// Hash function matching shader
        fn sim_hash_cpu(sim_idx: u32, src_idx: u32) -> u32 {
            let mut h = sim_idx.wrapping_mul(0x9E3779B9).wrapping_add(src_idx.wrapping_mul(0x85EBCA6B));
            h = h ^ (h >> 16);
            h = h.wrapping_mul(0x21F0AAAD);
            h = h ^ (h >> 15);
            h
        }

        fn source_offset_x_cpu(sim_idx: u32, src_idx: u32, base_x: u32, grid_width: u32) -> u32 {
            let h = Self::sim_hash_cpu(sim_idx, src_idx * 2);
            let offset = (h % grid_width) as i32 - (grid_width / 2) as i32;
            let new_x = base_x as i32 + offset;
            ((new_x + grid_width as i32) % grid_width as i32) as u32
        }

        fn source_offset_y_cpu(sim_idx: u32, src_idx: u32, base_y: u32, _grid_width: u32, grid_height: u32) -> u32 {
            let h = Self::sim_hash_cpu(sim_idx, src_idx * 2 + 1);
            let offset = (h % grid_height) as i32 - (grid_height / 2) as i32;
            let new_y = base_y as i32 + offset;
            ((new_y + grid_height as i32) % grid_height as i32) as u32
        }

        fn in_source_cpu(x: i32, y: i32, sx: u32, sy: u32, shape: u32, radius: u32) -> bool {
            let dx = x as f32 - sx as f32;
            let dy = y as f32 - sy as f32;
            let r = radius as f32;
            let r_sq = r * r;
            let dist_sq = dx * dx + dy * dy;

            match shape {
                0 => dist_sq <= r_sq,                           // Circle
                1 => dx.abs() <= r && dy.abs() <= r / 4.0,     // Horizontal bar
                2 => dx.abs() <= r / 4.0 && dy.abs() <= r,     // Vertical bar
                3 => dy <= 0.0 && dist_sq <= r_sq,             // Upper semicircle
                4 => dy >= 0.0 && dist_sq <= r_sq,             // Lower semicircle
                5 => dx <= 0.0 && dist_sq <= r_sq,             // Left semicircle
                6 => dx >= 0.0 && dist_sq <= r_sq,             // Right semicircle
                7 => {
                    let norm = (dx / r).powi(2) + (dy / (r / 2.0)).powi(2);
                    norm <= 1.0
                }                                               // Horizontal ellipse
                8 => {
                    let norm = (dx / (r / 2.0)).powi(2) + (dy / r).powi(2);
                    norm <= 1.0
                }                                               // Vertical ellipse
                _ => dist_sq <= r_sq,
            }
        }

        /// Update energy map on GPU
        fn update_energy_map(&self) {
            let map = Self::compute_energy_map_static(
                &self.energy_params,
                self.num_programs,
                self.num_sims,
                self.grid_width,
                self.grid_height,
            );
            self.queue.write_buffer(&self.energy_map_buffer, 0, bytemuck::cast_slice(&map));
        }

        /// Update energy config
        pub fn update_energy_config_all(&mut self, config: &crate::energy::EnergyConfig) {
            self.energy_params = EnergyParams::from_config(config, self.grid_width, self.grid_height);
            self.queue.write_buffer(&self.energy_params_buffer, 0, bytemuck::bytes_of(&self.energy_params));
            // Also update energy map bitmask
            self.update_energy_map();
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

        /// Set epoch (for resuming from checkpoint)
        pub fn set_epoch(&mut self, epoch: u64) {
            self.epoch = epoch;
        }

        /// Get all soup data from all simulations (for checkpointing)
        pub fn get_all_soup(&self) -> Vec<u8> {
            let total_size = self.num_programs * 64 * self.num_sims;
            
            // Create a larger staging buffer if needed
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Checkpoint Staging"),
                size: total_size as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&self.soup_buffer, 0, &staging, 0, total_size as u64);
            self.queue.submit(Some(encoder.finish()));
            
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            self.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            
            let data = slice.get_mapped_range();
            let result = data.to_vec();
            drop(data);
            staging.unmap();
            result
        }

        /// Begin async readback - copies soup to staging buffer, returns immediately
        /// Call finish_async_readback() to get the data
        pub fn begin_async_readback(&mut self) {
            let total_size = (self.num_programs * 64 * self.num_sims) as u64;
            
            // Choose which staging buffer to write to
            let staging = if self.staging_current {
                &self.staging_soup_b
            } else {
                &self.staging_soup_a
            };
            
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&self.soup_buffer, 0, staging, 0, total_size);
            self.queue.submit(Some(encoder.finish()));
            
            // Start the map operation (non-blocking)
            let slice = staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_result| {
                // Callback - we'll check result in finish_async_readback
            });
            
            // Swap buffers for next time
            self.staging_current = !self.staging_current;
            self.pending_readback = true;
        }

        /// Check if there's pending readback data
        pub fn has_pending_readback(&self) -> bool {
            self.pending_readback
        }

        /// Finish async readback - polls until ready and returns data
        /// Returns None if no readback was started
        pub fn finish_async_readback(&mut self) -> Option<Vec<u8>> {
            if !self.pending_readback {
                return None;
            }

            // The "other" buffer (not the one currently being written to) has our data
            let staging = if self.staging_current {
                &self.staging_soup_a  // current=true means we just wrote to B, so A has data
            } else {
                &self.staging_soup_b  // current=false means we just wrote to A, so B has data
            };

            // Poll until ready
            self.device.poll(wgpu::Maintain::Wait);

            let slice = staging.slice(..);
            let data = slice.get_mapped_range();
            let result = data.to_vec();
            drop(data);
            staging.unmap();

            self.pending_readback = false;
            Some(result)
        }

        /// Get all soup data using async pattern if readback is pending,
        /// otherwise use synchronous path. More convenient API.
        pub fn get_all_soup_async(&mut self) -> Vec<u8> {
            if self.pending_readback {
                self.finish_async_readback().unwrap_or_else(|| self.get_all_soup())
            } else {
                self.get_all_soup()
            }
        }

        /// Get all energy states from all simulations (for checkpointing)
        pub fn get_all_energy_states(&self) -> Vec<u32> {
            let total_size = self.num_programs * self.num_sims * 4; // u32 per program
            
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Energy Staging"),
                size: total_size as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&self.energy_state_buffer, 0, &staging, 0, total_size as u64);
            self.queue.submit(Some(encoder.finish()));
            
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            self.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            
            let data = slice.get_mapped_range();
            let result: Vec<u32> = data.chunks(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            drop(data);
            staging.unmap();
            result
        }

        /// Restore soup data from checkpoint
        pub fn set_all_soup(&self, soup: &[u8]) {
            self.queue.write_buffer(&self.soup_buffer, 0, soup);
        }

        /// Restore energy states from checkpoint
        pub fn set_all_energy_states(&self, energy_states: &[u32]) {
            let bytes: Vec<u8> = energy_states.iter()
                .flat_map(|&e| e.to_le_bytes())
                .collect();
            self.queue.write_buffer(&self.energy_state_buffer, 0, &bytes);
        }

        /// Get grid width
        pub fn grid_width(&self) -> usize {
            self.grid_width
        }

        /// Get number of programs per simulation
        pub fn num_programs(&self) -> usize {
            self.num_programs
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

