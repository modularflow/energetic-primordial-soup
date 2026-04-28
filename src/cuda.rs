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
use cudarc::driver::safe::CudaStream;
#[cfg(feature = "cuda")]
use cudarc::driver::sys::{CUevent, CUevent_flags};
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// BFF command bytes (used for non-op init - these are EXCLUDED to keep programs inert).
#[cfg(feature = "cuda")]
const BFF_COMMAND_BYTES: [u8; 10] = [b'<', b'>', b'{', b'}', b'+', b'-', b'.', b',', b'[', b']'];

/// Returns true if `b` is a BFF command byte.
#[cfg(feature = "cuda")]
fn is_bff_command(b: u8) -> bool {
    BFF_COMMAND_BYTES.contains(&b)
}

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
///
/// Uses two CUDA streams for concurrent compute + host/device transfers:
/// - `compute_stream`: all kernel launches and ops-buffer zeroing
/// - `xfer_stream`: all DtoH readbacks (ops counter, soup) and HtoD uploads (pairs, energy map)
///
/// Ordering between the streams is enforced with lightweight CUDA events so the
/// host only blocks when it actually needs data, rather than via a global
/// `device.synchronize()` after every step.
#[cfg(feature = "cuda")]
pub struct CudaMultiSimulation {
    device: Arc<CudaDevice>,
    soup_gpu: CudaSlice<u8>,
    /// Pre-allocated at the maximum pair count (`num_sims * num_programs` u32s =
    /// `total_programs/2` pairs × 2 u32 each). Actual in-use length is `num_pairs * 2`.
    pairs_gpu: CudaSlice<u32>,
    energy_state_gpu: CudaSlice<u32>,
    sim_configs_gpu: CudaSlice<u32>,
    energy_map_gpu: CudaSlice<u32>,
    /// Single ops counter. Zeroed before each kernel launch on `compute_stream`,
    /// then DtoH'd on `xfer_stream` after `kernel_done_event` fires.
    ops_gpu: CudaSlice<u64>,
    kernel: CudaFunction,

    // --- Streams & events for dual-stream execution -----------------------
    compute_stream: CudaStream,
    xfer_stream: CudaStream,
    /// Recorded on `compute_stream` after each kernel launch. `xfer_stream`
    /// waits on it before reading the ops counter or the soup.
    kernel_done_event: CUevent,
    /// Recorded on `xfer_stream` after the pair HtoD completes.
    /// `compute_stream` waits on it before the next kernel launch.
    pairs_ready_event: CUevent,

    // --- Host-side staging buffers ----------------------------------------
    /// Owned host buffer for pairs. `memcpy_htod_async` references this slice,
    /// so it must live until the HtoD copy completes (guaranteed by the
    /// `xfer_stream` sync at the top of `set_pairs_*`).
    pairs_host: Vec<u32>,
    /// 1-element host buffer used as the DtoH target for the ops counter.
    pending_ops: Vec<u64>,
    /// Async readback buffer for the full soup (allocated lazily).
    pending_readback: Option<Vec<u8>>,

    // --- Config -----------------------------------------------------------
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
    /// Ops from the previous kernel launch (cached on each `step()`).
    last_ops: u64,
}

#[cfg(feature = "cuda")]
impl Drop for CudaMultiSimulation {
    fn drop(&mut self) {
        // Make sure no more work is queued on either stream before destroying
        // events (the streams themselves are cleaned up by `CudaStream::drop`).
        let _ = unsafe { result::stream::synchronize(self.compute_stream.stream) };
        let _ = unsafe { result::stream::synchronize(self.xfer_stream.stream) };
        unsafe {
            let _ = result::event::destroy(self.kernel_done_event);
            let _ = result::event::destroy(self.pairs_ready_event);
        }
    }
}

#[cfg(feature = "cuda")]
impl CudaMultiSimulation {
    /// Create a new CUDA multi-simulation
    ///
    /// Unlike wgpu, CUDA has no 4GB buffer limit - you can use your full GPU memory.
    ///
    /// `nonop_rate` (0.0..=1.0): fraction of programs that start as a "non-op"
    /// program - 64 bytes drawn uniformly from the 246 non-command byte values.
    /// These programs execute as pure NOPs at epoch 0 but their bytes can still
    /// be read/written by partners during interaction. `0.0` = all random.
    ///
    /// `init_region`: optional (width, height) of a centered "active"
    /// sub-region within each sim's grid. Cells outside this region are
    /// forced to non-op at init, creating a non-op buffer zone. `None` =
    /// the full grid is active.
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
        nonop_rate: f32,
        init_region: Option<(usize, usize)>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize CUDA
        let device = CudaDevice::new(0)?;

        println!("CUDA Device: Initialized successfully");

        // Calculate memory requirements (pairs sized at MAX possible - mega mode)
        let total_programs = num_sims * num_programs;
        let soup_size = total_programs * 64;
        let energy_size = total_programs * 4;
        let max_pairs = total_programs / 2;
        let pairs_size = max_pairs * 2 * 4;
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

        // Soup initialization. Most programs get fully random bytes; with
        // probability `nonop_rate` a program is replaced with 64 random
        // non-command bytes (a pure-NOP program). If `init_region` is set,
        // cells outside the centered active sub-region are unconditionally
        // forced to non-op (buffer zone).
        use rand::Rng;
        let mut rng = rand::rng();
        let nonop_rate = nonop_rate.clamp(0.0, 1.0);
        let mut soup: Vec<u8> = (0..soup_size).map(|_| rng.random()).collect();

        // Compute active-region bounds (centered within each sim's grid).
        let (active_w, active_h) = init_region
            .map(|(w, h)| (w.min(grid_width), h.min(grid_height)))
            .unwrap_or((grid_width, grid_height));
        let has_buffer = active_w < grid_width || active_h < grid_height;
        let active_x0 = (grid_width - active_w) / 2;
        let active_y0 = (grid_height - active_h) / 2;
        let active_x1 = active_x0 + active_w;
        let active_y1 = active_y0 + active_h;

        let mut nonop_count_active: usize = 0;
        let mut nonop_count_buffer: usize = 0;
        if nonop_rate > 0.0 || has_buffer {
            for prog_idx in 0..total_programs {
                let local_idx = prog_idx % num_programs;
                let x = local_idx % grid_width;
                let y = local_idx / grid_width;
                let in_active = x >= active_x0 && x < active_x1
                    && y >= active_y0 && y < active_y1;

                let make_nonop = if !in_active {
                    true
                } else if nonop_rate > 0.0 {
                    rng.random::<f32>() < nonop_rate
                } else {
                    false
                };

                if make_nonop {
                    let base = prog_idx * 64;
                    for b in &mut soup[base..base + 64] {
                        loop {
                            let v: u8 = rng.random();
                            if !is_bff_command(v) {
                                *b = v;
                                break;
                            }
                        }
                    }
                    if in_active {
                        nonop_count_active += 1;
                    } else {
                        nonop_count_buffer += 1;
                    }
                }
            }

            let total_active_cells = active_w * active_h * num_sims;
            let total_buffer_cells = total_programs - total_active_cells;
            if has_buffer {
                println!(
                    "  Init region: {}x{} active center at ({},{})..({},{}) per sim",
                    active_w, active_h, active_x0, active_y0, active_x1, active_y1,
                );
                println!(
                    "  Non-op init: {} active + {} buffer = {} / {} programs",
                    nonop_count_active,
                    nonop_count_buffer,
                    nonop_count_active + nonop_count_buffer,
                    total_programs,
                );
                println!(
                    "    Active: {:.2}% nonop ({} / {}), Buffer: {} / {} (forced)",
                    100.0 * nonop_count_active as f64 / total_active_cells.max(1) as f64,
                    nonop_count_active,
                    total_active_cells,
                    nonop_count_buffer,
                    total_buffer_cells,
                );
            } else {
                println!(
                    "  Non-op init: {} / {} programs ({:.2}% target, {:.2}% actual)",
                    nonop_count_active,
                    total_programs,
                    nonop_rate * 100.0,
                    100.0 * nonop_count_active as f64 / total_programs as f64,
                );
            }
        }


        // Energy states (all zero-packed: reserve=0, timer=0, dead=false)
        let packed_initial_state = 0u32;
        let energy_states: Vec<u32> = vec![packed_initial_state; total_programs];

        // Allocate + populate GPU buffers
        let soup_gpu = device.htod_sync_copy(&soup)?;

        // Pairs buffer: preallocated at max size. Filled lazily by set_pairs_*.
        let mut pairs_gpu = unsafe { device.alloc::<u32>(max_pairs * 2) }?;
        // Seed with a simple default (adjacent pairs) for a single sim so the
        // first epoch has something reasonable if set_pairs_* isn't called.
        let seed_pairs: Vec<u32> = generate_pairs(num_programs)
            .into_iter()
            .flat_map(|(a, b)| [a as u32, b as u32])
            .collect();
        if !seed_pairs.is_empty() {
            device.htod_sync_copy_into(&seed_pairs, &mut pairs_gpu.slice_mut(..seed_pairs.len()))?;
        }
        let mut pairs_host: Vec<u32> = Vec::with_capacity(max_pairs * 2);
        pairs_host.extend_from_slice(&seed_pairs);

        let energy_state_gpu = device.htod_sync_copy(&energy_states)?;
        let sim_configs_gpu = device.htod_sync_copy(&sim_configs)?;
        let energy_map_gpu = device.htod_sync_copy(&energy_map)?;
        let ops_gpu = device.alloc_zeros::<u64>(1)?;

        // Create the two work streams (both NonBlocking, forked from default).
        let compute_stream = device.fork_default_stream()?;
        let xfer_stream = device.fork_default_stream()?;

        // Create events (disable timing for slightly lower overhead).
        let kernel_done_event = result::event::create(CUevent_flags::CU_EVENT_DISABLE_TIMING)?;
        let pairs_ready_event = result::event::create(CUevent_flags::CU_EVENT_DISABLE_TIMING)?;

        Ok(Self {
            device,
            soup_gpu,
            pairs_gpu,
            energy_state_gpu,
            sim_configs_gpu,
            energy_map_gpu,
            ops_gpu,
            kernel,
            compute_stream,
            xfer_stream,
            kernel_done_event,
            pairs_ready_event,
            pairs_host,
            pending_ops: vec![0u64; 1],
            pending_readback: None,
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
            last_ops: 0,
        })
    }
    
    /// Run one epoch across all simulations.
    ///
    /// Dual-stream pipeline:
    ///   1. Sync `xfer_stream` to make the previous epoch's ops readback visible.
    ///   2. Enqueue an async HtoD of the staged `pairs_host` on `xfer_stream`,
    ///      record `pairs_ready_event`.
    ///   3. `compute_stream` waits on `pairs_ready_event`, then `memset_d8_async`
    ///      zeros the ops counter and the kernel is launched on `compute_stream`.
    ///   4. `kernel_done_event` is recorded on `compute_stream`.
    ///   5. `xfer_stream` waits on `kernel_done_event` and starts the DtoH of
    ///      the ops counter into `pending_ops`.
    ///
    /// Because the DtoH is not synced here, the host never blocks on kernel
    /// completion inside `step()` (except via the `xfer_stream` sync at the top,
    /// which only waits for the previous epoch's single-u64 DtoH). Returns the
    /// ops count from the previous epoch (0 on the first epoch).
    pub fn step(&mut self) -> u64 {
        use cudarc::driver::sys::CUevent_wait_flags as EWF;

        // ---- 1. Retrieve previous epoch's ops via xfer_stream sync -------
        if self.epoch > 0 {
            unsafe {
                if let Err(e) = result::stream::synchronize(self.xfer_stream.stream) {
                    eprintln!("CUDA xfer_stream sync error at epoch {}: {:?}", self.epoch, e);
                }
            }
            self.last_ops = self.pending_ops[0];
        }

        // ---- 2. Async HtoD of the staged pair buffer on xfer_stream ------
        // Also enforces "previous kernel is done with pairs_gpu" because
        // xfer_stream already waited on kernel_done_event during the previous
        // epoch's DtoH enqueue (which completed as part of the sync above).
        let flat_len = (self.num_pairs * 2).min(self.pairs_host.len());
        if flat_len > 0 {
            unsafe {
                if let Err(e) = result::memcpy_htod_async(
                    *self.pairs_gpu.device_ptr(),
                    &self.pairs_host[..flat_len],
                    self.xfer_stream.stream,
                ) {
                    eprintln!("CUDA pair HtoD error at epoch {}: {:?}", self.epoch, e);
                }
                if let Err(e) = result::event::record(self.pairs_ready_event, self.xfer_stream.stream) {
                    eprintln!("CUDA pairs_ready record error: {:?}", e);
                }
            }
        }

        // ---- 3. Prepare and launch the kernel on compute_stream ----------
        // compute_stream must wait for the pair HtoD to complete before it
        // reads pairs_gpu.
        unsafe {
            if let Err(e) = result::stream::wait_event(
                self.compute_stream.stream,
                self.pairs_ready_event,
                EWF::CU_EVENT_WAIT_DEFAULT,
            ) {
                eprintln!("CUDA compute wait_event error: {:?}", e);
            }
            // Zero the ops counter on compute_stream before the kernel starts.
            if let Err(e) = result::memset_d8_async(
                *self.ops_gpu.device_ptr(),
                0,
                self.ops_gpu.num_bytes(),
                self.compute_stream.stream,
            ) {
                eprintln!("CUDA ops memset error: {:?}", e);
            }
        }

        let total_pairs = if self.mega_mode {
            self.num_pairs
        } else {
            self.num_pairs * self.num_sims
        };
        let block_size = 256u32;
        let grid_size = ((total_pairs as u32) + block_size - 1) / block_size;

        let params_packed1 = ((self.num_pairs as u64) << 32) | (self.num_programs as u64);
        let params_packed2 = ((self.num_sims as u64) << 32) | (self.steps_per_run as u64);
        let flags = (if self.energy_enabled { 1u64 } else { 0u64 })
            | (if self.mega_mode { 2u64 } else { 0u64 })
            | ((self.spontaneous_rate as u64) << 2);
        let params_packed3 = ((self.mutation_prob as u64) << 32) | flags;

        let cfg = LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (grid_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            if let Err(e) = self.kernel.clone().launch_on_stream(
                &self.compute_stream,
                cfg,
                (
                    &self.soup_gpu,
                    &self.pairs_gpu,
                    &self.energy_state_gpu,
                    &self.sim_configs_gpu,
                    &self.energy_map_gpu,
                    &self.ops_gpu,
                    params_packed1,
                    params_packed2,
                    params_packed3,
                    self.seed,
                    self.epoch,
                ),
            ) {
                eprintln!("CUDA kernel launch failed at epoch {}: {:?}", self.epoch, e);
            }
            // ---- 4. Record kernel_done_event --------------------------
            if let Err(e) = result::event::record(self.kernel_done_event, self.compute_stream.stream) {
                eprintln!("CUDA kernel_done record error: {:?}", e);
            }
        }

        // ---- 5. Enqueue ops DtoH on xfer_stream (waits for kernel_done) --
        unsafe {
            if let Err(e) = result::stream::wait_event(
                self.xfer_stream.stream,
                self.kernel_done_event,
                EWF::CU_EVENT_WAIT_DEFAULT,
            ) {
                eprintln!("CUDA xfer wait_event error: {:?}", e);
            }
            if let Err(e) = result::memcpy_dtoh_async(
                &mut self.pending_ops[..],
                *self.ops_gpu.device_ptr(),
                self.xfer_stream.stream,
            ) {
                eprintln!("CUDA ops DtoH error: {:?}", e);
            }
        }

        self.epoch += 1;
        self.last_ops
    }

    /// Block the host until all queued compute work is complete.
    /// Use after the final `step()` when you need to make sure the last
    /// kernel's ops count (and soup writes) are fully committed.
    pub fn sync(&self) {
        unsafe {
            let _ = result::stream::synchronize(self.compute_stream.stream);
            let _ = result::stream::synchronize(self.xfer_stream.stream);
        }
    }
    
    /// Make sure the most-recently-launched kernel has finished writing soup /
    /// energy state. Use before any synchronous readback that runs on streams
    /// other than `compute_stream`.
    fn wait_for_compute(&self) {
        unsafe {
            let _ = result::stream::synchronize(self.compute_stream.stream);
        }
    }

    /// Get soup data for a specific simulation (blocking).
    pub fn get_sim_soup(&self, sim_idx: usize) -> Vec<u8> {
        self.wait_for_compute();
        let offset = sim_idx * self.num_programs * 64;
        let size = self.num_programs * 64;

        let mut data = vec![0u8; size];
        self.device.dtoh_sync_copy_into(
            &self.soup_gpu.slice(offset..offset + size),
            &mut data
        ).unwrap();
        data
    }

    /// Get all soup data (blocking).
    pub fn get_all_soup(&self) -> Vec<u8> {
        self.wait_for_compute();
        let size = self.num_sims * self.num_programs * 64;
        let mut data = vec![0u8; size];
        self.device.dtoh_sync_copy_into(&self.soup_gpu, &mut data).unwrap();
        data
    }

    /// Begin async readback of all soup data on `xfer_stream`, waiting for the
    /// latest kernel to finish via `kernel_done_event`. Use
    /// `finish_async_readback` to retrieve it.
    pub fn begin_async_readback(&mut self) {
        use cudarc::driver::sys::CUevent_wait_flags as EWF;
        if self.pending_readback.is_some() {
            return;
        }

        let size = self.soup_gpu.len();
        let mut data = vec![0u8; size];

        if let Err(e) = self.device.bind_to_thread() {
            eprintln!("CUDA readback bind error: {:?}", e);
            return;
        }

        unsafe {
            // Ensure the current kernel is done before we start reading soup.
            if let Err(e) = result::stream::wait_event(
                self.xfer_stream.stream,
                self.kernel_done_event,
                EWF::CU_EVENT_WAIT_DEFAULT,
            ) {
                eprintln!("CUDA readback wait_event failed: {:?}", e);
                return;
            }
            if let Err(e) = result::memcpy_dtoh_async(
                &mut data,
                *self.soup_gpu.device_ptr(),
                self.xfer_stream.stream,
            ) {
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
    /// Syncs only `xfer_stream` (not the full device), so other compute work
    /// queued on `compute_stream` keeps running.
    pub fn finish_async_readback(&mut self) -> Option<Vec<u8>> {
        let data = self.pending_readback.take()?;
        unsafe {
            if let Err(e) = result::stream::synchronize(self.xfer_stream.stream) {
                eprintln!("CUDA readback sync error: {:?}", e);
                return None;
            }
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
        self.wait_for_compute();
        let size = self.num_sims * self.num_programs;
        let mut data = vec![0u32; size];
        self.device.dtoh_sync_copy_into(&self.energy_state_gpu, &mut data).unwrap();
        data
    }

    /// Stage new pair indices in the host-side buffer. The actual HtoD happens
    /// inside the next `step()` on `xfer_stream`. Syncs `xfer_stream` first so
    /// we don't overwrite a buffer that's still being copied.
    fn stage_pairs(&mut self, pairs: &[(u32, u32)]) {
        unsafe {
            // Make sure any in-flight HtoD of pairs_host from a previous
            // step() has completed before we mutate it.
            let _ = result::stream::synchronize(self.xfer_stream.stream);
        }
        let max_pairs = self.num_sims * self.num_programs / 2;
        let n = pairs.len().min(max_pairs);
        self.pairs_host.clear();
        self.pairs_host.reserve(n * 2);
        for &(a, b) in &pairs[..n] {
            self.pairs_host.push(a);
            self.pairs_host.push(b);
        }
        self.num_pairs = n;
    }

    /// Restore soup data from checkpoint. Syncs compute first so we don't
    /// stomp on a running kernel.
    pub fn set_all_soup(&mut self, soup: &[u8]) {
        self.wait_for_compute();
        if let Err(e) = self.device.htod_sync_copy_into(soup, &mut self.soup_gpu) {
            eprintln!("CUDA soup restore failed: {:?}", e);
        }
    }

    /// Restore energy states from checkpoint.
    pub fn set_all_energy_states(&mut self, energy_states: &[u32]) {
        self.wait_for_compute();
        if let Err(e) = self.device.htod_sync_copy_into(energy_states, &mut self.energy_state_gpu) {
            eprintln!("CUDA energy restore failed: {:?}", e);
        }
    }

    /// Set pairs (local indices) for all simulations.
    pub fn set_pairs_all(&mut self, pairs: &[(u32, u32)]) {
        self.stage_pairs(pairs);
    }

    /// Enable/disable mega-simulation mode (pairs are absolute indices).
    pub fn set_mega_mode(&mut self, enabled: bool) {
        self.mega_mode = enabled;
    }

    /// Set pairs for mega mode (absolute indices across all sims).
    pub fn set_pairs_mega(&mut self, pairs: &[(u32, u32)]) {
        self.stage_pairs(pairs);
    }
    
    pub fn num_sims(&self) -> usize { self.num_sims }
    pub fn num_programs(&self) -> usize { self.num_programs }
    pub fn grid_width(&self) -> usize { self.grid_width }
    pub fn grid_height(&self) -> usize { self.grid_height }
    pub fn epoch(&self) -> u64 { self.epoch }
    pub fn set_epoch(&mut self, epoch: u64) { self.epoch = epoch; }

    /// Update energy configuration (recomputes energy map on GPU).
    /// Call this when energy sources change dynamically. Waits for the current
    /// compute kernel to finish first so the update is safe.
    pub fn update_energy_config(&mut self, config: &crate::energy::EnergyConfig) {
        let energy_map = compute_energy_map(
            Some(config),
            self.num_programs,
            self.num_sims,
            self.grid_width,
            self.grid_height,
            self.border_thickness,
        );

        self.wait_for_compute();
        if let Err(e) = self.device.htod_sync_copy_into(&energy_map, &mut self.energy_map_gpu) {
            eprintln!("CUDA energy map update failed: {:?}", e);
        }

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
