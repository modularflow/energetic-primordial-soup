# BFF Primordial Soup Simulation

A high-performance GPU-accelerated implementation of self-modifying program soups, derived from the research on computational life and emergent self-replicators. This project is 100% "vibe" coded in rust because rust forces a lot of guardrails that the original in c++ does not. I don't know what I don't know, so I'd rather this be safe (not to wacky things with memory) and user friendly (a simple binary and yaml files without having to mess around with compiling c++ or making the python bindings work).

## Overview

This project simulates a "primordial soup" of programs that can interact, mutate, and potentially evolve self-replicating behavior. Programs are written in BFF (Brainfuck Family), a minimal instruction set that supports:

- Two read/write heads on a shared 128-byte tape (64 bytes per program)
- Copy operations between paired programs
- Stochastic mutations
- 2D spatial topology with neighbor pairing

An optional **energy system** adds evolutionary pressure by restricting mutation to localized zones, forcing programs to adapt to their environment for survival and reproduction. 

## BFF Instruction Set

| Instruction | Description |
|-------------|-------------|
| `<` `>` | Move head 0 left/right |
| `{` `}` | Move head 1 left/right |
| `+` `-` | Increment/decrement byte at head 0 |
| `.` | Copy byte from head 0 to head 1 |
| `,` | Copy byte from head 1 to head 0 |
| `[` `]` | Loop (jump if byte at head 0 is zero/non-zero) |

Programs are paired and execute on a combined 128-byte tape (64 bytes from each program).

## Key Features

- **GPU Acceleration**: CUDA (NVIDIA, no buffer limits) or WGPU (Vulkan/Metal, cross-platform), plus CPU fallback
- **Batched Simulations**: Run hundreds of parallel simulations simultaneously with different parameters
- **Mega-Simulation Mode**: Grid of sub-simulations with cross-border program interactions
- **Energy System**: Dynamic energy zones with configurable shapes (circles, strips, ellipses, etc.)
- **Metrics Tracking**: Brotli compression ratio analysis to detect phase transitions and self-replicator emergence
- **Checkpointing**: Save and resume complete simulation state
- **Async I/O**: Non-blocking data saves and post-processing frame rendering
- **YAML Configuration**: Simple text-based configuration

## Inspirations

> **Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple Interaction**  
> Blaise Aguera y Arcas, Jyrki Alakuijala, James Evans, Ben Laurie, Alexander Mordvintsev, Eyvind Niklasson, Ettore Randazzo, Luca Versari  
> arXiv:2406.19108 [cs.NE], 2024  
> https://arxiv.org/abs/2406.19108

The original CUDA implementation can be found at:
https://github.com/paradigms-of-intelligence/cubff


The "energy" portion of this implementation is also inspired by research done on slime mould and its evolved thermodynamic computing capabilities. 
> Reid CR. Thoughts from the forest floor: a review of cognition in the slime mould Physarum polycephalum. Anim Cogn. 2023 Nov;26(6):1783-1797. doi: 10.1007/s10071-023-01782-1. Epub 2023 May 11. 
> PMID: 37166523; PMCID: PMC10770251.
> https://pmc.ncbi.nlm.nih.gov/articles/PMC10770251/
## Requirements

- Rust 1.70+
- GPU (optional but recommended): NVIDIA (CUDA), AMD/Intel (Vulkan), or Apple (Metal)
- ffmpeg (for video generation)

**Compute Backends:**
- `cuda`: NVIDIA GPU (fastest, no buffer limits)
- `wgpu`: Cross-platform GPU via Vulkan/Metal (4GB buffer limit)
- `cpu`: Fallback (slow)

## Installation

```bash
git clone https://github.com/modularflow/energetic-primordial-soup.git
cd energetic-primordial-soup

# For NVIDIA GPUs (recommended - fastest, no buffer limits)
cargo build --release --features cuda

# For cross-platform GPU (Vulkan/Metal - works on AMD, Intel, Apple)
cargo build --release --features wgpu-compute

# For CPU-only (no GPU required, slow)
cargo build --release
```

## Quick Start

### Using the run script

```bash
# Run with default config.yaml
./run.sh

# Run with a specific config file
./run.sh my_config.yaml

# Override parameters via environment variables
MAX_EPOCHS=5000 ./run.sh

# Use paths from config.yaml instead of run.sh defaults
USE_CONFIG_DIRS=true ./run.sh
```

### Using the binary directly

```bash
# Generate a default config file
./target/release/energetic-primordial-soup --generate-config

# Run with config file
./target/release/energetic-primordial-soup --config config.yaml

# Run with command-line arguments
./target/release/energetic-primordial-soup \
    --grid-width 512 \
    --grid-height 256 \
    --max-epochs 10000 \
    --energy \
    --energy-sources 4 \
    --energy-radius 64
```

## Configuration

Create a `config.yaml` file to configure the simulation. Key options:

```yaml
# Grid size per simulation
grid:
  width: 1024
  height: 1024

# Core simulation
simulation:
  seed: 42
  mutation_rate: 2048           # 1 in N chance per byte
  steps_per_run: 4096           # BFF execution steps per epoch
  max_epochs: 1000000
  neighbor_range: 2             # Spatial pairing range
  parallel_sims: 256            # Run N simulations in parallel
  parallel_layout: [16, 16]     # Grid layout for mega-simulation
  border_interaction: true      # Enable cross-border program pairing
  border_thickness: 2           # Dead zone width between simulations

# Output
output:
  frame_interval: 256           # Save every N epochs
  frames_dir: "frames"
  frame_format: "png"           # png/ppm/jpeg
  save_raw: true                # Save binary data for post-processing
  async_save: true              # Non-blocking saves

# Checkpointing
checkpoint:
  enabled: true
  interval: 50000
  path: "checkpoints"
  resume_from: ""               # Path to resume from

# Energy system
energy:
  enabled: true
  sources: 6                    # Number of energy zones
  radius: 64
  reserve_epochs: 50000         # Energy reserve when leaving zone
  death_epochs: 10000           # Timeout before death (0 = immortal)
  spontaneous_rate: 10          # 1 in N chance to respawn dead programs
  shape: "random"               # circle, ellipse, strip_h, strip_v, half_circle_*
  
  dynamic:
    random_placement: true      # Randomize positions per sim
    source_lifetime: 10000      # Expire after N epochs (0 = permanent)
    spawn_rate: 5000            # Spawn new source every N epochs
  
  # Optional: Different death timers per simulation group
  sim_groups:
    - death_epochs: 10000
      count: 85
    - death_epochs: 100000
      count: 85
    - death_epochs: 0           # Immortal
      count: 86

# Metrics (optional)
metrics:
  enabled: true                 # Track compression ratio
  interval: 1000
  output_file: "metrics.csv"
```

## Energy System

The energy system creates spatial evolutionary pressure:

- **Energy Zones**: Configurable shapes (circle, ellipse, strip, half-circle) placed strategically or randomly
- **Mutation Control**: Only programs in zones or with reserve energy can mutate
- **Reserve Energy**: Temporary mutation ability when leaving a zone
- **Death Timer**: Programs die if isolated too long outside zones (configurable, 0 = immortal)
- **Spontaneous Generation**: Dead programs in zones can randomly respawn with new code
- **Dynamic Behavior**: Sources spawn, expire, and move over time
- **Per-Simulation Variation**: Each parallel sim gets unique energy field positions
- **Simulation Groups**: Compare different death timers across batched simulations

## Mega-Simulation Mode

Enable `border_interaction` to arrange parallel simulations in a grid where adjacent sims can exchange programs:

```yaml
simulation:
  parallel_sims: 256
  parallel_layout: [16, 16]   # 16x16 grid of sub-simulations
  border_interaction: true
```

This creates a single large grid (e.g., 16×16 layout of 256×256 sims = 4096×4096 total programs) where genetic information flows across simulation boundaries.

## Checkpointing and Post-Processing

**Checkpoints** save complete simulation state (soup, energy states, epoch, config) in a binary format with YAML header:

```yaml
checkpoint:
  enabled: true
  interval: 10000
  resume_from: "checkpoints/checkpoint_epoch_00010000_sims_256.bff"
```

**Async Raw Data Saving** maximizes simulation speed by saving binary dumps in the background:

```yaml
output:
  save_raw: true
  async_save: true
  render_frames: false  # Render later for max speed
```

Render frames after simulation:
```bash
./target/release/energetic-primordial-soup \
  --render-raw /path/to/raw_data \
  --frames-dir /path/to/frames
```

## Output

- **Frames**: PNG/JPEG/PPM images color-coded by byte values
- **Videos**: MP4 via ffmpeg from frame sequences
- **Checkpoints**: Binary files with complete simulation state
- **Raw Data**: Fast binary dumps for post-processing
- **Metrics CSV**: Compression ratio and phase transition tracking

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

```
Copyright 2024

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Programs execute in pairs on a 128-byte tape (64 bytes each).

## References

- Aguera y Arcas, B., et al. (2024). "Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple Interaction." arXiv:2406.19108
- CuBFF - Original CUDA implementation: https://github.com/paradigms-of-intelligence/cubff
- Reid CR. (2023). "Cognition in the slime mould Physarum polycephalum." Anim Cogn. 26(6):1783-1797

## License

Apache License 2.0 - see LICENSE file for details.
