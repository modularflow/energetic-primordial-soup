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
- Configurable energy zones for spatial evolutionary dynamics
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
grid:
  width: 1024
  height: 512

simulation:
  seed: 42
  mutation_rate: 4096      # 1 in N chance per byte
  steps_per_run: 8192      # BFF execution steps per epoch
  max_epochs: 50000
  neighbor_range: 2        # Pairing range (2 = 5x5 neighborhood)
  parallel_sims: 8         # Run N simulations in parallel on GPU

output:
  frame_interval: 256      # Save frame every N epochs (0 = disabled)
  frames_dir: "frames"

energy:
  enabled: true
  sources: 8               # Number of energy sources (1-8)
  radius: 128              # Radius of each source in grid cells
  reserve_epochs: 50       # Reserve energy when leaving zone
  death_epochs: 100        # Epochs without interaction until death
  dynamic:
    random_placement: true
    max_sources: 20
    source_lifetime: 10000  # Epochs until source expires (0 = infinite)
    spawn_rate: 5000        # Spawn new source every N epochs (0 = disabled)
```

## Energy System

The energy system adds spatial structure to the simulation:

- **Energy Sources**: Fixed or randomly placed zones on the grid
- **Mutation Permission**: Only programs within energy zones (or with reserve energy) can mutate
- **Reserve Energy**: Programs leaving an energy zone retain mutation ability for a limited time
- **Death Timer**: Programs outside energy zones that don't interact for too long become inactive
- **Dynamic Sources**: Sources can spawn, expire, and move over time

This creates evolutionary pressure where programs must either stay near energy sources or develop behaviors that allow them to survive and propagate outside the zones.

## Parallel Simulations

The GPU implementation supports running multiple simulations in parallel using a single dispatch:

```yaml
simulation:
  parallel_sims: 8  # Run 8 different seeds simultaneously
```

This achieves near-linear scaling, allowing you to explore multiple random seeds at approximately the same speed as a single simulation.

## Output

The simulation generates:

- **PPM frames**: Raw image frames showing program state (color-coded by byte values)
- **MP4 videos**: Compressed videos created from frames via ffmpeg
- **Log files**: Simulation statistics and progress

When running parallel simulations, each simulation gets its own video:
- `simulation.mp4` - Main video (first simulation)
- `sim_0.mp4`, `sim_1.mp4`, ... - Individual simulation videos

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

On an NVIDIA RTX 4090 with a 1024x512 grid (524,288 programs):

| Configuration | Throughput |
|--------------|------------|
| Single simulation | ~166 billion ops/sec |
| 8 parallel simulations | ~198 billion ops/sec total |

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

