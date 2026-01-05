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
use cudarc::driver::result;
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

#[cfg(feature = "cuda")]
fn sim_hash_cpu(sim_idx: u32, src_idx: u32) -> u32 {
    let mut h = sim_idx.wrapping_mul(0x9E3779B9).wrapping_add(src_idx.wrapping_mul(0x85EBCA6B));
    h ^= h >> 16;
    h = h.wrapping_mul(0x21F0AAAD);
    h ^= h >> 15;
    h
}

#[cfg(feature = "cuda")]
fn source_offset_x_cpu(sim_idx: u32, src_idx: u32, base_x: u32, grid_width: u32) -> u32 {
    let h = sim_hash_cpu(sim_idx, src_idx * 2);
    let offset = (h % grid_width) as i32 - (grid_width / 2) as i32;
    let new_x = base_x as i32 + offset;
    ((new_x + grid_width as i32) % grid_width as i32) as u32
}

#[cfg(feature = "cuda")]
fn source_offset_y_cpu(sim_idx: u32, src_idx: u32, base_y: u32, grid_height: u32) -> u32 {
    let h = sim_hash_cpu(sim_idx, src_idx * 2 + 1);
    let offset = (h % grid_height) as i32 - (grid_height / 2) as i32;
    let new_y = base_y as i32 + offset;
    ((new_y + grid_height as i32) % grid_height as i32) as u32
}

#[cfg(feature = "cuda")]
fn in_source_cpu(x: i32, y: i32, sx: u32, sy: u32, shape: u32, radius: u32) -> bool {
    let dx = x as f32 - sx as f32;
    let dy = y as f32 - sy as f32;
    let r = radius as f32;
    let r_sq = r * r;
    let dist_sq = dx * dx + dy * dy;

    match shape {
        0 => dist_sq <= r_sq,
        1 => dx.abs() <= r && dy.abs() <= r / 4.0,
        2 => dx.abs() <= r / 4.0 && dy.abs() <= r,
        3 => dy <= 0.0 && dist_sq <= r_sq,
        4 => dy >= 0.0 && dist_sq <= r_sq,
        5 => dx <= 0.0 && dist_sq <= r_sq,
        6 => dx >= 0.0 && dist_sq <= r_sq,
        7 => {
            let norm = (dx / r).powi(2) + (dy / (r / 2.0)).powi(2);
            norm <= 1.0
        }
        8 => {
            let norm = (dx / (r / 2.0)).powi(2) + (dy / r).powi(2);
            norm <= 1.0
        }
        _ => dist_sq <= r_sq,
    }
}

#[cfg(feature = "cuda")]
fn compute_energy_map(
    config: Option<&crate::energy::EnergyConfig>,
    num_programs: usize,
    num_sims: usize,
    grid_width: usize,
    grid_height: usize,
    border_thickness: usize,
) -> Vec<u32> {
    let total_programs = num_programs * num_sims;
    let num_words = (total_programs + 31) / 32;
    let mut map = vec![0u32; num_words];

    let config = match config {
        Some(cfg) if cfg.enabled && !cfg.sources.is_empty() => cfg,
        _ => {
            for word in &mut map {
                *word = 0xFFFFFFFF;
            }
            return map;
        }
    };

    let sources: Vec<(u32, u32, u32, u32)> = config
        .sources
        .iter()
        .take(8)
        .map(|s| (s.x as u32, s.y as u32, s.shape.to_gpu_id(), s.radius as u32))
        .collect();

    for sim_idx in 0..num_sims {
        for prog_idx in 0..num_programs {
            let x = (prog_idx % grid_width) as i32;
            let y = (prog_idx / grid_width) as i32;

            // Check if we are in the "dead zone" border
            if border_thickness > 0 {
                if x < border_thickness as i32 || x >= (grid_width - border_thickness) as i32 ||
                   y < border_thickness as i32 || y >= (grid_height - border_thickness) as i32 {
                    // In dead zone - no energy
                    continue;
                }
            }

            let mut in_zone = false;
            for (src_idx, (base_x, base_y, shape, radius)) in sources.iter().enumerate() {
                let offset_x = source_offset_x_cpu(sim_idx as u32, src_idx as u32, *base_x, grid_width as u32);
                let offset_y = source_offset_y_cpu(sim_idx as u32, src_idx as u32, *base_y, grid_height as u32);
                if in_source_cpu(x, y, offset_x, offset_y, *shape, *radius) {
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

/// CUDA kernel source for batched multi-simulation BFF evaluation
/// This kernel supports:
/// - Multiple simulations in parallel (batched)
/// - Energy system with per-sim death_timer and reserve_duration
/// - Full 64-bit addressing (no 4GB limit)
/// - Per-block reduction for ops counter (reduces atomic contention)
#[cfg(feature = "cuda")]
const BFF_CUDA_KERNEL: &str = r#"
extern "C" __global__ void bff_batched_evaluate(
    unsigned char* soup,              // All programs across all sims: [sim0_prog0, sim0_prog1, ..., sim1_prog0, ...]
    const unsigned int* pair_indices, // Pairs per sim: [p1, p2, p1, p2, ...]
    unsigned int* energy_state,       // Packed energy state per program: reserve(16) | timer(15) | dead(1)
    const unsigned int* sim_configs,  // Per-sim configs: [death_timer, reserve_duration] pairs
    const unsigned int* energy_map,   // Bitmask: 1 bit per program indicating if in energy zone
    unsigned long long* ops_count,    // Atomic counter for total ops
    // Packed parameters (to fit cudarc's 12-param limit)
    unsigned long long params_packed1, // num_pairs(hi) | num_programs(lo)
    unsigned long long params_packed2, // num_sims(hi) | steps_per_run(lo)
    unsigned long long params_packed3, // mutation_prob(hi) | flags(lo: bit0=energy_enabled, bit1=mega_mode, bits2-31=spontaneous_rate)
    unsigned long long seed,
    unsigned long long epoch
) {
    // Shared memory for per-block ops reduction (256 threads per block)
    __shared__ unsigned long long block_ops[256];
    unsigned int tid = threadIdx.x;

    // Unpack parameters
    unsigned int num_pairs = (unsigned int)(params_packed1 >> 32);
    unsigned int num_programs = (unsigned int)(params_packed1 & 0xFFFFFFFF);
    unsigned int num_sims = (unsigned int)(params_packed2 >> 32);
    unsigned int steps_per_run = (unsigned int)(params_packed2 & 0xFFFFFFFF);
    unsigned int mutation_prob = (unsigned int)(params_packed3 >> 32);
    unsigned int flags = (unsigned int)(params_packed3 & 0xFFFFFFFF);
    unsigned int energy_enabled = flags & 1u;
    unsigned int mega_mode = (flags >> 1) & 1u;
    unsigned int spontaneous_rate = flags >> 2;  // Upper 30 bits for spontaneous_rate
    const int SINGLE_TAPE_SIZE = 64;
    const int FULL_TAPE_SIZE = 128;

    // Global pair index across all sims (normal mode) or across all pairs (mega mode)
    unsigned long long global_idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize ops counter for this thread (will be 0 if we skip work)
    unsigned int ops = 0;

    // Use a flag instead of early returns so all threads can participate in reduction
    bool should_process = true;

    unsigned int pair_idx = 0;
    unsigned int sim_idx = 0;
    unsigned int p1_local = 0;
    unsigned int p2_local = 0;
    unsigned int p1_sim = 0;
    unsigned int p2_sim = 0;
    unsigned long long p1_abs = 0;
    unsigned long long p2_abs = 0;

    if (mega_mode) {
        pair_idx = (unsigned int)global_idx;
        if (pair_idx >= num_pairs) {
            should_process = false;
        } else {
            // Pairs are absolute indices across all sims
            p1_abs = pair_indices[pair_idx * 2];
            p2_abs = pair_indices[pair_idx * 2 + 1];
            p1_sim = (unsigned int)(p1_abs / num_programs);
            p2_sim = (unsigned int)(p2_abs / num_programs);
            p1_local = (unsigned int)(p1_abs % num_programs);
            p2_local = (unsigned int)(p2_abs % num_programs);
            sim_idx = p1_sim; // RNG uses p1's sim index
        }
    } else {
        // Normal mode: pairs are local per sim
        sim_idx = (unsigned int)(global_idx / num_pairs);
        pair_idx = (unsigned int)(global_idx % num_pairs);
        if (sim_idx >= num_sims || pair_idx >= num_pairs) {
            should_process = false;
        } else {
            p1_local = pair_indices[pair_idx * 2];
            p2_local = pair_indices[pair_idx * 2 + 1];

            unsigned long long sim_offset = (unsigned long long)sim_idx * num_programs;
            p1_abs = sim_offset + p1_local;
            p2_abs = sim_offset + p2_local;
            p1_sim = sim_idx;
            p2_sim = sim_idx;
        }
    }

    if (should_process) {
        // Get per-sim energy config (may differ for p1/p2 in mega mode)
        unsigned int p1_death_timer = sim_configs[p1_sim * 2];
        unsigned int p1_reserve_duration = sim_configs[p1_sim * 2 + 1];
        unsigned int p2_death_timer = sim_configs[p2_sim * 2];
        unsigned int p2_reserve_duration = sim_configs[p2_sim * 2 + 1];

        // Check energy zone membership (bitmask lookup) - use bitwise ops for speed
        auto in_energy_zone = [&](unsigned long long prog_idx) -> bool {
            unsigned int word_idx = (unsigned int)(prog_idx >> 5);  // prog_idx / 32
            unsigned int bit_idx = (unsigned int)(prog_idx & 31);   // prog_idx % 32
            return (energy_map[word_idx] & (1u << bit_idx)) != 0;
        };

        // Energy state helpers - packed as: reserve(16 bits) | timer(15 bits) | dead(1 bit)
        // This allows death_epochs up to 32767 (vs 255 with 8-bit packing)
        auto get_reserve = [](unsigned int state) -> unsigned int { return state & 0xFFFF; };
        auto get_timer = [](unsigned int state) -> unsigned int { return (state >> 16) & 0x7FFF; };
        auto is_dead = [](unsigned int state) -> bool { return (state >> 31) != 0; };
        auto pack_state = [](unsigned int reserve, unsigned int timer, bool dead) -> unsigned int {
            return (reserve & 0xFFFF) | ((timer & 0x7FFF) << 16) | ((dead ? 1u : 0u) << 31);
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
            should_process = false;
        }

        if (should_process) {
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
            unsigned int rng = (unsigned int)seed ^ (unsigned int)epoch
                ^ (pair_idx * 0x9E3779B9u) ^ (sim_idx * 0x85EBCA6Bu);

            auto mutate_sparse = [&](unsigned int start, unsigned int end) {
                unsigned int inv_prob = (1u << 30) / (mutation_prob > 0 ? mutation_prob : 1u);
                unsigned int byte_pos = start;
                unsigned int end_byte = end;

                rng = lcg(rng);
                unsigned int skip = ((rng >> 8) * inv_prob) >> 22;
                byte_pos += skip;

                while (byte_pos < end_byte) {
                    rng = lcg(rng);
                    tape[byte_pos] = (unsigned char)((rng >> 8) & 0xFF);

                    rng = lcg(rng);
                    skip = ((rng >> 8) * inv_prob) >> 22;
                    if (skip < 1u) { skip = 1u; }
                    byte_pos += skip;
                }
            };

            if (p1_can_mutate) {
                mutate_sparse(0u, SINGLE_TAPE_SIZE);
            }

            if (p2_can_mutate) {
                mutate_sparse(SINGLE_TAPE_SIZE, FULL_TAPE_SIZE);
            }

            // Track if copies occurred (for energy inheritance)
            bool p1_received_copy = false;
            bool p2_received_copy = false;

            // Skip interpreter if tape is empty
            bool tape_active = false;
            for (int i = 0; i < FULL_TAPE_SIZE; i++) {
                if (tape[i] != 0) {
                    tape_active = true;
                    break;
                }
            }

            // BFF Evaluation
            int pos = 2;
            int head0 = tape[0] & (FULL_TAPE_SIZE - 1);
            int head1 = tape[1] & (FULL_TAPE_SIZE - 1);

            if (tape_active) {
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
                    p1_reserve = p1_reserve_duration;
                    p1_timer = 0;
                } else if (p1_received_copy) {
                    p1_reserve = p2_in_zone ? p2_reserve_duration : get_reserve(p2_state);
                    p1_timer = 0;
                    p1_dead = false;
                } else {
                    if (p1_reserve > 0) p1_reserve--;
                    if (!p1_dead) p1_timer++;
                    // death_timer = 0 means infinite (never dies)
                    if (p1_death_timer > 0 && p1_timer > p1_death_timer && !p1_dead) {
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
                    p2_reserve = p2_reserve_duration;
                    p2_timer = 0;
                } else if (p2_received_copy) {
                    p2_reserve = p1_in_zone ? p1_reserve_duration : get_reserve(p1_state);
                    p2_timer = 0;
                    p2_dead = false;
                } else {
                    if (p2_reserve > 0) p2_reserve--;
                    if (!p2_dead) p2_timer++;
                    if (p2_death_timer > 0 && p2_timer > p2_death_timer && !p2_dead) {
                        p2_dead = true;
                    }
                }
                p2_stays_dead = p2_was_dead && p2_dead;
                energy_state[p2_abs] = pack_state(p2_reserve, p2_timer, p2_dead);
            }

            // Spontaneous generation: dead tapes in energy zones have a chance to spawn new random programs
            bool p1_spawned = false;
            bool p2_spawned = false;
            
            if (energy_enabled && spontaneous_rate > 0) {
                // P1: Check for spontaneous generation
                if (p1_stays_dead && p1_in_zone) {
                    rng = lcg(rng);
                    if (rng % spontaneous_rate == 0) {
                        // Spawn new random program!
                        for (int i = 0; i < SINGLE_TAPE_SIZE; i++) {
                            rng = lcg(rng);
                            tape[i] = (unsigned char)((rng >> 8) & 0xFF);
                        }
                        // Revive the program with per-sim reserve duration
                        unsigned int p1_reserve_dur = sim_configs[p1_sim * 2 + 1];
                        energy_state[p1_abs] = pack_state(p1_reserve_dur, 0, false);
                        p1_spawned = true;
                        p1_stays_dead = false;
                    }
                }
                
                // P2: Check for spontaneous generation
                if (p2_stays_dead && p2_in_zone) {
                    rng = lcg(rng);
                    if (rng % spontaneous_rate == 0) {
                        // Spawn new random program!
                        for (int i = 0; i < SINGLE_TAPE_SIZE; i++) {
                            rng = lcg(rng);
                            tape[SINGLE_TAPE_SIZE + i] = (unsigned char)((rng >> 8) & 0xFF);
                        }
                        // Revive the program with per-sim reserve duration
                        unsigned int p2_reserve_dur = sim_configs[p2_sim * 2 + 1];
                        energy_state[p2_abs] = pack_state(p2_reserve_dur, 0, false);
                        p2_spawned = true;
                        p2_stays_dead = false;
                    }
                }
            }

            // Write back soup (dead tapes stay zeroed unless spontaneously spawned)
            if (p1_stays_dead) {
                for (int i = 0; i < SINGLE_TAPE_SIZE; i++) {
                    soup[p1_byte_offset + i] = 0;
                }
            } else if (energy_enabled && is_dead(energy_state[p1_abs]) && !p1_spawned) {
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
            } else if (energy_enabled && is_dead(energy_state[p2_abs]) && !p2_spawned) {
                for (int i = 0; i < SINGLE_TAPE_SIZE; i++) {
                    soup[p2_byte_offset + i] = 0;
                }
            } else {
                for (int i = 0; i < SINGLE_TAPE_SIZE; i++) {
                    soup[p2_byte_offset + i] = tape[SINGLE_TAPE_SIZE + i];
                }
            }
        }
    }

    // Per-block reduction for ops counter (all threads must participate)
    // This reduces atomic contention from ~65K atomics to ~256 atomics
    block_ops[tid] = (unsigned long long)ops;
    __syncthreads();

    // Tree reduction within block
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            block_ops[tid] += block_ops[tid + stride];
        }
        __syncthreads();
    }

    // Only thread 0 does the atomic add to global counter
    if (tid == 0) {
        atomicAdd(ops_count, block_ops[0]);
    }
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
    // Double-buffered ops counters for async readback
    ops_count_gpu_a: CudaSlice<u64>,
    ops_count_gpu_b: CudaSlice<u64>,
    ops_write_to_a: bool,  // true = kernel writes to A, read from B
    last_ops: u64,         // cached ops from previous epoch
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
    mega_mode: bool,
    spontaneous_rate: u32,
    border_thickness: usize,
    pending_readback: Option<Vec<u8>>,
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
        border_thickness: usize,
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
        let energy_enabled = energy_config
            .map(|c| c.enabled && !c.sources.is_empty())
            .unwrap_or(false);
        let default_death = energy_config.map(|c| c.interaction_death).unwrap_or(10);
        let default_reserve = energy_config.map(|c| c.reserve_duration).unwrap_or(5);
        let spontaneous_rate = energy_config.map(|c| c.spontaneous_rate).unwrap_or(0);
        
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
        
        // Energy map (precomputed zones, with per-sim offsets and border thickness)
        let energy_map = compute_energy_map(
            energy_config,
            num_programs,
            num_sims,
            grid_width,
            grid_height,
            border_thickness,
        );
        
        // Soup with random data
        use rand::Rng;
        let mut rng = rand::rng();
        let soup: Vec<u8> = (0..soup_size).map(|_| rng.random()).collect();
        
        
        // Energy states (all alive with full reserve)
        let packed_initial_state = 0u32;
        let energy_states: Vec<u32> = vec![packed_initial_state; total_programs];
        
        // Allocate AND initialize GPU buffers using htod_sync_copy (copies data in one step)
        let soup_gpu = device.htod_sync_copy(&soup)?;
        let pairs_gpu = device.htod_sync_copy(&pairs)?;
        let energy_state_gpu = device.htod_sync_copy(&energy_states)?;
        let sim_configs_gpu = device.htod_sync_copy(&sim_configs)?;
        let energy_map_gpu = device.htod_sync_copy(&energy_map)?;
        // Double-buffered ops counters for async readback
        let ops_count_gpu_a = device.alloc_zeros::<u64>(1)?;
        let ops_count_gpu_b = device.alloc_zeros::<u64>(1)?;

        Ok(Self {
            device,
            soup_gpu,
            pairs_gpu,
            energy_state_gpu,
            sim_configs_gpu,
            energy_map_gpu,
            ops_count_gpu_a,
            ops_count_gpu_b,
            ops_write_to_a: true,
            last_ops: 0,
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
            mega_mode: false,
            spontaneous_rate,
            border_thickness,
            pending_readback: None,
        })
    }
    
    /// Run one epoch across all simulations
    /// Uses double-buffered ops counter for async readback - returns previous epoch's ops
    /// (returns 0 on first epoch since there's no previous data)
    pub fn step(&mut self) -> u64 {
        // Calculate grid dimensions
        let total_pairs = if self.mega_mode {
            self.num_pairs
        } else {
            self.num_pairs * self.num_sims
        };
        let block_size = 256u32;
        let grid_size = ((total_pairs as u32) + block_size - 1) / block_size;

        // Pack some u32 params together into u64 values for the kernel
        let params_packed1 = ((self.num_pairs as u64) << 32) | (self.num_programs as u64);
        let params_packed2 = ((self.num_sims as u64) << 32) | (self.steps_per_run as u64);
        // Pack flags: bit0=energy_enabled, bit1=mega_mode, bits2-31=spontaneous_rate
        let flags = (if self.energy_enabled { 1u64 } else { 0u64 })
            | (if self.mega_mode { 2u64 } else { 0u64 })
            | ((self.spontaneous_rate as u64) << 2);
        let params_packed3 = ((self.mutation_prob as u64) << 32) | flags;

        let cfg = LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (grid_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Select which buffer to write to and which to read from
        let (write_buf, read_buf) = if self.ops_write_to_a {
            (&self.ops_count_gpu_a, &self.ops_count_gpu_b)
        } else {
            (&self.ops_count_gpu_b, &self.ops_count_gpu_a)
        };

        // Launch kernel writing to current buffer
        unsafe {
            self.kernel.clone().launch(cfg, (
                &self.soup_gpu,
                &self.pairs_gpu,
                &self.energy_state_gpu,
                &self.sim_configs_gpu,
                &self.energy_map_gpu,
                write_buf,
                params_packed1,
                params_packed2,
                params_packed3,
                self.seed,
                self.epoch,
            )).expect("Kernel launch failed");
        }

        // Start async read of the OTHER buffer (previous epoch's data)
        // This overlaps with kernel execution
        let mut ops = [0u64];
        if self.epoch > 0 {
            // Use async copy - overlaps with kernel execution
            let device_ptr = *read_buf.device_ptr();
            let stream = *self.device.cu_stream();
            unsafe {
                // Start async copy while kernel is still running
                let _ = result::memcpy_dtoh_async(&mut ops, device_ptr, stream);
            }
        }

        // Sync and check for errors (kernel errors are caught here, also completes async copy)
        match self.device.synchronize() {
            Ok(()) => {}
            Err(e) => {
                eprintln!("CUDA sync error at epoch {}: {:?}", self.epoch, e);
                return self.last_ops;
            }
        }

        // Reset the read buffer for next use (it will be written to after we swap)
        if self.epoch > 0 {
            let read_buf_mut = if self.ops_write_to_a {
                &mut self.ops_count_gpu_b
            } else {
                &mut self.ops_count_gpu_a
            };
            if let Err(e) = self.device.memset_zeros(read_buf_mut) {
                eprintln!("Failed to reset ops counter: {:?}", e);
            }
        }

        // Swap buffers for next epoch
        self.ops_write_to_a = !self.ops_write_to_a;
        self.epoch += 1;

        // Cache and return the ops (from previous epoch, or 0 if first epoch)
        self.last_ops = ops[0];
        self.last_ops
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

    /// Begin async readback of all soup data. Use finish_async_readback to retrieve it.
    pub fn begin_async_readback(&mut self) {
        if self.pending_readback.is_some() {
            return;
        }

        let size = self.soup_gpu.len();
        let mut data = vec![0u8; size];

        if let Err(e) = self.device.bind_to_thread() {
            eprintln!("CUDA readback bind error: {:?}", e);
            return;
        }

        let device_ptr = *self.soup_gpu.device_ptr();
        let stream = *self.device.cu_stream();
        unsafe {
            if let Err(e) = result::memcpy_dtoh_async(&mut data, device_ptr, stream) {
                eprintln!("CUDA async readback failed: {:?}", e);
                return;
            }
        }

        self.pending_readback = Some(data);
    }

    /// Returns true if async readback data is pending.
    pub fn has_pending_readback(&self) -> bool {
        self.pending_readback.is_some()
    }

    /// Finish async readback and return soup data, or None if not pending.
    pub fn finish_async_readback(&mut self) -> Option<Vec<u8>> {
        let data = self.pending_readback.take()?;
        if let Err(e) = self.device.synchronize() {
            eprintln!("CUDA readback sync error: {:?}", e);
            return None;
        }
        Some(data)
    }

    /// Get all soup data using async readback if pending, otherwise sync.
    pub fn get_all_soup_async(&mut self) -> Vec<u8> {
        if self.pending_readback.is_some() {
            self.finish_async_readback().unwrap_or_else(|| self.get_all_soup())
        } else {
            self.get_all_soup()
        }
    }

    pub fn get_all_energy_states(&self) -> Vec<u32> {
        let size = self.num_sims * self.num_programs;
        let mut data = vec![0u32; size];
        self.device.dtoh_sync_copy_into(&self.energy_state_gpu, &mut data).unwrap();
        data
    }

    /// Restore soup data from checkpoint
    pub fn set_all_soup(&mut self, soup: &[u8]) {
        if let Err(e) = self.device.htod_sync_copy_into(soup, &mut self.soup_gpu) {
            eprintln!("CUDA soup restore failed: {:?}", e);
        }
    }

    /// Restore energy states from checkpoint
    pub fn set_all_energy_states(&mut self, energy_states: &[u32]) {
        if let Err(e) = self.device.htod_sync_copy_into(energy_states, &mut self.energy_state_gpu) {
            eprintln!("CUDA energy restore failed: {:?}", e);
        }
    }

    /// Set pairs (local indices) for all simulations
    pub fn set_pairs_all(&mut self, pairs: &[(u32, u32)]) {
        let flat: Vec<u32> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
        self.pairs_gpu = self.device.htod_sync_copy(&flat).unwrap();
        self.num_pairs = pairs.len();
    }

    /// Enable/disable mega-simulation mode (pairs are absolute indices)
    pub fn set_mega_mode(&mut self, enabled: bool) {
        self.mega_mode = enabled;
    }

    /// Set pairs for mega mode (absolute indices across all sims)
    pub fn set_pairs_mega(&mut self, pairs: &[(u32, u32)]) {
        let flat: Vec<u32> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
        self.pairs_gpu = self.device.htod_sync_copy(&flat).unwrap();
        self.num_pairs = pairs.len();
    }
    
    pub fn num_sims(&self) -> usize { self.num_sims }
    pub fn num_programs(&self) -> usize { self.num_programs }
    pub fn grid_width(&self) -> usize { self.grid_width }
    pub fn grid_height(&self) -> usize { self.grid_height }
    pub fn epoch(&self) -> u64 { self.epoch }
    pub fn set_epoch(&mut self, epoch: u64) { self.epoch = epoch; }

    /// Update energy configuration (recomputes energy map on GPU)
    /// Call this when energy sources change dynamically
    pub fn update_energy_config(&mut self, config: &crate::energy::EnergyConfig) {
        let energy_map = compute_energy_map(
            Some(config),
            self.num_programs,
            self.num_sims,
            self.grid_width,
            self.grid_height,
            self.border_thickness,
        );

        // Upload new energy map to GPU
        if let Err(e) = self.device.htod_sync_copy_into(&energy_map, &mut self.energy_map_gpu) {
            eprintln!("CUDA energy map update failed: {:?}", e);
        }

        // Update enabled flag
        self.energy_enabled = config.enabled && !config.sources.is_empty();
    }
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
