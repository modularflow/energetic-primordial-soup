# BFF Primordial Soup Simulation

A high-performance GPU-accelerated implementation of self-modifying program soups, derived from the research on computational life and emergent self-replicators. This project is 100% "vibe" coded in rust because rust forces a lot of guardrails that the original in c++ does not. I don't know what I don't know, so I'd rather this be safe (not to wacky things with memory) and user friendly (a simple binary and yaml files without having to mess around with compiling c++ or making the python bindings work).

## Overview

This project simulates a "primordial soup" of programs that can interact, mutate, and potentially evolve self-replicating behavior. Programs are written in BFF (Brainfuck Family), a minimal instruction set that supports:

- Two read/write heads operating on a shared tape
- Copy operations between program pairs
- Random mutations
- 2D toroidal grid topology for spatial structure

The simulation includes an optional **energy system** that creates localized zones where mutation is permitted, adding evolutionary pressure and spatial dynamics. The intent is to see if our replicators can compute logic of the underlying simulation as a means to survive. If the hypothesis is feasible through running the simulation, I believe this method can be expanded to other simulations of our world that can lead to new ways of evolving artifical intelligence architectures, rather than pre-determining  them. 


## Inspirations

This implementation is directly derived from the paper:

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

## Background

This Rust implementation provides:
- GPU acceleration via wgpu (Vulkan/Metal backend for accessibility of all GPU brands)
- Batched parallel simulations (run multiple seeds simultaneously)
- Mega-simulation mode (parallel sims arranged in a grid with cross-border interaction)
- Configurable energy zones with variable shapes for spatial evolutionary dynamics
- Checkpointing for saving and resuming simulations
- Async raw data saving for maximum simulation speed
- Post-simulation frame rendering
- YAML configuration files
- Video generation from simulation frames

## Requirements

- Rust 1.70+
- GPU with Vulkan or Metal support (for GPU acceleration)
- ffmpeg (for video generation)

## Installation

```bash
git clone <repository-url>
cd energetic-primordial-soup
cargo build --release --features wgpu-compute
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

Create a `config.yaml` file to configure the simulation:

```yaml
# Grid dimensions
grid:
  width: 256
  height: 256

# Core simulation parameters
simulation:
  seed: 42                      # Random seed for reproducibility
  mutation_rate: 4096           # 1 in N chance per byte (higher = less mutation)
  steps_per_run: 8192           # BFF execution steps per epoch
  max_epochs: 100000            # Total epochs to run
  neighbor_range: 2             # Pairing range (2 = 5x5 neighborhood)
  auto_terminate_dead_epochs: 0 # Terminate if all dead for N epochs (0=disabled)
  parallel_sims: 256            # Run N simulations in parallel on GPU
  parallel_layout: [16, 16]     # Arrange as [cols, rows] grid for mega-simulation
  border_interaction: true      # Enable cross-simulation pairing at borders

# Output settings
output:
  frame_interval: 128           # Save every N epochs (0 = disabled)
  frames_dir: "frames"          # Output directory (relative or absolute)
  frame_format: "png"           # "png", "jpeg", or "ppm"
  thumbnail_scale: 4            # Downscale factor (1 = full, 4 = 1/4 size)
  
  # Raw data saving (for post-simulation rendering)
  save_raw: true                # Save raw soup data (fast binary dumps)
  raw_dir: "raw_data"           # Directory for raw data files
  async_save: true              # Save in background thread (non-blocking)
  render_frames: false          # Render frames during simulation

# Checkpoint settings
checkpoint:
  enabled: true                 # Enable checkpointing
  interval: 10000               # Save every N epochs (0 = only at end)
  path: "checkpoints"           # Directory for checkpoint files
  resume_from: ""               # Path to checkpoint to resume (empty = fresh start)

# Energy system
energy:
  enabled: true
  sources: 6                    # Number of sources (1-8)
  radius: 64                    # Radius of each source
  reserve_epochs: 50            # Reserve energy when leaving zone
  death_epochs: 100             # Epochs without interaction until death
  spontaneous_rate: 10          # 1 in N chance for dead tape in zone to respawn (0=disabled)
  shape: "random"               # Shape: circle, strip_h, strip_v, half_circle,
                                # half_circle_bottom, half_circle_left, half_circle_right,
                                # ellipse, ellipse_v, random
  
  # Dynamic energy settings
  dynamic:
    random_placement: true      # Randomize source positions
    max_sources: 10             # Maximum simultaneous sources
    source_lifetime: 10000      # Epochs until source expires (0 = infinite)
    spawn_rate: 5000            # Spawn new source every N epochs (0 = disabled)
```

## Energy System

The energy system adds spatial structure to the simulation:

- **Energy Sources**: Fixed or randomly placed zones on the grid with configurable shapes
- **Mutation Permission**: Only programs within energy zones (or with reserve energy) can mutate
- **Reserve Energy**: Programs leaving an energy zone retain mutation ability for a limited time
- **Death Timer**: Programs outside energy zones that don't interact for too long become inactive
- **Dynamic Sources**: Sources can spawn, expire, and move over time
- **Spontaneous Generation**: Dead tapes within energy zones can randomly spawn new programs
- **Per-Simulation Variation**: Each parallel simulation gets unique energy field positions

### Energy Zone Shapes

Available shapes for energy zones:
- `circle` - Standard circular zone
- `strip_h` - Horizontal strip
- `strip_v` - Vertical strip
- `half_circle` - Top half of a circle
- `half_circle_bottom` - Bottom half of a circle
- `half_circle_left` - Left half of a circle
- `half_circle_right` - Right half of a circle
- `ellipse` - Horizontal ellipse
- `ellipse_v` - Vertical ellipse
- `random` - Random shape per source

## Mega-Simulation Mode

When `border_interaction` is enabled, parallel simulations are arranged in a grid where adjacent simulations can interact at their borders:

```yaml
simulation:
  parallel_sims: 256
  parallel_layout: [16, 16]   # 16x16 grid of simulations
  border_interaction: true    # Enable cross-border pairing
```

This creates a single large simulation grid (e.g., 4096x4096 programs for 16x16 layout of 256x256 sims) where programs at simulation edges can pair with programs in adjacent simulations. This enables genetic information to flow across the entire mega-grid.

Cross-border pairs are generated first to ensure edge programs interact with neighbors, then internal pairs fill in the remaining programs.

## Checkpointing

Save and restore complete simulation state:

```yaml
checkpoint:
  enabled: true
  interval: 10000            # Save every 10000 epochs
  path: "checkpoints"
  resume_from: ""            # Set to checkpoint path to resume
```

Checkpoint files contain:
- All program tapes (soup data)
- Energy states (reserve, timer, dead status)
- Current epoch
- Configuration metadata for validation

To resume from a checkpoint:
```yaml
checkpoint:
  resume_from: "checkpoints/checkpoint_epoch_00010000_sims_256.bff"
```

## Raw Data Saving and Post-Processing

For maximum simulation speed, save raw binary data during the run and render frames afterwards:

### During Simulation (Maximum Speed)
```yaml
output:
  frame_interval: 128
  save_raw: true           # Save raw soup data
  async_save: true         # Non-blocking saves
  render_frames: false     # Skip rendering during sim
```

### Post-Simulation Rendering
```bash
# Render frames from saved raw data
./render_frames.sh /path/to/raw_data /path/to/frames

# Or use the binary directly
./target/release/energetic-primordial-soup \
  --render-raw /path/to/raw_data \
  --frames-dir /path/to/frames \
  --config config.yaml
```

Raw data files are compact binary dumps (~16MB for 256 sims at 256x256). The async writer ensures saves happen in a background thread without blocking the GPU.

## Output

The simulation generates:

- **Frames**: PNG, JPEG, or PPM images showing program state (color-coded by byte values)
- **MP4 videos**: Compressed videos created from frames via ffmpeg
- **Log files**: Simulation statistics and progress
- **Checkpoints**: Binary files for resuming simulations
- **Raw data**: Binary soup dumps for post-processing

When running mega-simulations, a combined frame shows all simulations arranged in their grid layout.

## CLI Options

```
USAGE:
    energetic-primordial-soup [OPTIONS]
    energetic-primordial-soup --config config.yaml
    energetic-primordial-soup --generate-config [output.yaml]
    energetic-primordial-soup --render-raw <raw_data_dir>

CONFIG FILE:
    -c, --config <FILE>       Load settings from YAML config file
    --generate-config [FILE]  Generate template config (default: config.yaml)

OPTIONS:
    -w, --grid-width <N>      Grid width (default: 512)
    -h, --grid-height <N>     Grid height (default: 256)
    -s, --seed <N>            Random seed (default: 42)
    -m, --mutation-prob <N>   Mutation probability (default: 262144)
    --steps-per-run <N>       Steps per BFF run (default: 8192)
    -e, --max-epochs <N>      Maximum epochs (default: 10000)
    -n, --neighbor-range <N>  Neighbor range (default: 2)
    -f, --frame-interval <N>  Save frame every N epochs (0 = disabled)
    -d, --frames-dir <PATH>   Frames output directory

ENERGY SYSTEM:
    --energy                  Enable energy sources
    --energy-sources <N>      Initial sources 1-8 (default: 4)
    --energy-radius <N>       Radius of each source (default: 64)
    --energy-reserve <N>      Reserve epochs when leaving zone (default: 5)
    --energy-death <N>        Epochs until program death (default: 10)

DYNAMIC ENERGY:
    --energy-random           Randomize source positions
    --energy-max-sources <N>  Max simultaneous sources (default: 8)
    --energy-source-lifetime <N>  Epochs until source expires (0=infinite)
    --energy-spawn-rate <N>   Spawn new source every N epochs (0=disabled)

RAW DATA / ASYNC SAVE:
    --save-raw                Save raw soup data
    --raw-dir <PATH>          Raw data output directory
    --async-save              Non-blocking saves (default)
    --no-async-save           Blocking saves
    --render-frames           Render frames during simulation
    --no-render-frames        Skip rendering

POST-PROCESSING:
    --render-raw <PATH>       Render frames from raw data directory
```

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

## Performance

On an NVIDIA RTX 4090:

| Configuration | Throughput |
|--------------|------------|
| Single simulation (1024x512) | ~166 billion ops/sec |
| 8 parallel simulations | ~198 billion ops/sec |
| 256 mega-simulation (16x16 layout) | ~1900 billion ops/sec |

Raw data saving with async mode has minimal performance impact as saves occur in a background thread.

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

```
Copyright 2024

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## References

1. Aguera y Arcas, B., et al. (2024). "Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple Interaction." arXiv:2406.19108. https://arxiv.org/abs/2406.19108

2. CuBFF - Original CUDA implementation. https://github.com/paradigms-of-intelligence/cubff

3. Reid CR. Thoughts from the forest floor: a review of cognition in the slime mould Physarum polycephalum. Anim Cogn. 2023 Nov;26(6):1783-1797. doi: 10.1007/s10071-023-01782-1. Epub 2023 May 11.  PMID: 37166523; PMCID: PMC10770251. https://pmc.ncbi.nlm.nih.gov/articles/PMC10770251/

## Acknowledgments

This implementation builds upon the foundational research by the Paradigms of Intelligence team at Google, exploring how self-replicating programs can emerge from simple computational substrates without explicit fitness landscapes.
