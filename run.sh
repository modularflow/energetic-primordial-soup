#!/bin/bash
#
# BFF Primordial Soup Simulation Runner
# ======================================
#
# USAGE:
#   ./run.sh                     # Uses config.yaml (if exists) or defaults
#   ./run.sh config.yaml         # Use specific config file
#   MAX_EPOCHS=5000 ./run.sh     # Override specific values
#
# The config file controls parallel_sims (GPU batched parallelism).
# No need for separate parallel runner - it's built into the GPU simulation!
#

set -e

# ============================================================================
# CONFIG FILE
# ============================================================================
# Use argument, or CONFIG_FILE env var, or default to config.yaml if it exists
if [ -n "$1" ]; then
    CONFIG_FILE="$1"
elif [ -n "$CONFIG_FILE" ]; then
    CONFIG_FILE="$CONFIG_FILE"
elif [ -f "config.yaml" ]; then
    CONFIG_FILE="config.yaml"
else
    CONFIG_FILE=""
fi

# ============================================================================
# PARAMETERS (used only if no config file)
# ============================================================================
GRID_WIDTH=${GRID_WIDTH:-512}
GRID_HEIGHT=${GRID_HEIGHT:-256}
SEED=${SEED:-42}
MUTATION_RATE=${MUTATION_RATE:-4096}
STEPS_PER_RUN=${STEPS_PER_RUN:-8192}
MAX_EPOCHS=${MAX_EPOCHS:-10000}
NEIGHBOR_RANGE=${NEIGHBOR_RANGE:-2}
FRAME_INTERVAL=${FRAME_INTERVAL:-64}
VIDEO_FPS=${VIDEO_FPS:-30}
KEEP_FRAMES=${KEEP_FRAMES:-false}

# Energy system
ENERGY=${ENERGY:-false}
ENERGY_SOURCES=${ENERGY_SOURCES:-4}
ENERGY_RADIUS=${ENERGY_RADIUS:-64}
ENERGY_RESERVE=${ENERGY_RESERVE:-5}
ENERGY_DEATH=${ENERGY_DEATH:-10}
ENERGY_RANDOM=${ENERGY_RANDOM:-false}
ENERGY_MAX_SOURCES=${ENERGY_MAX_SOURCES:-8}
ENERGY_SOURCE_LIFETIME=${ENERGY_SOURCE_LIFETIME:-0}
ENERGY_SPAWN_RATE=${ENERGY_SPAWN_RATE:-0}

# ============================================================================
# SETUP
# ============================================================================
RUN_ID=$(date +"%Y%m%d_%H%M%S")
FRAMES_DIR="runs/${RUN_ID}_frames"
VIDEO_FILE="runs/${RUN_ID}_simulation.mp4"
LOG_FILE="runs/${RUN_ID}_log.txt"

mkdir -p runs
mkdir -p "$FRAMES_DIR"

# ============================================================================
# BUILD
# ============================================================================
echo "Building..."
cargo build --release --features wgpu-compute 2>&1 | tail -3
echo ""

# ============================================================================
# RUN SIMULATION
# ============================================================================
echo "=============================================="
echo "BFF Primordial Soup Simulation"
echo "=============================================="
echo ""
echo "Run ID: $RUN_ID"
echo ""

if [ -n "$CONFIG_FILE" ]; then
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Build override args from env vars (these override config file values)
    OVERRIDE_ARGS=""
    [ -n "${MAX_EPOCHS_OVERRIDE:-}" ] && OVERRIDE_ARGS="$OVERRIDE_ARGS --max-epochs $MAX_EPOCHS_OVERRIDE"
    [ -n "${SEED_OVERRIDE:-}" ] && OVERRIDE_ARGS="$OVERRIDE_ARGS --seed $SEED_OVERRIDE"
    [ -n "${FRAME_INTERVAL_OVERRIDE:-}" ] && OVERRIDE_ARGS="$OVERRIDE_ARGS --frame-interval $FRAME_INTERVAL_OVERRIDE"
    
    ./target/release/energetic-primordial-soup \
        --config "$CONFIG_FILE" \
        --frames-dir "$FRAMES_DIR" \
        $OVERRIDE_ARGS \
        2>&1 | tee "$LOG_FILE"
else
    # No config file - use env vars
    echo "Parameters:"
    echo "  Grid: ${GRID_WIDTH}x${GRID_HEIGHT}"
    echo "  Epochs: $MAX_EPOCHS"
    echo "  Seed: $SEED"
    if [ "$ENERGY" = true ]; then
        echo "  Energy: enabled ($ENERGY_SOURCES sources, radius $ENERGY_RADIUS)"
    fi
    echo ""
    
    MUTATION_PROB=$((1073741824 / MUTATION_RATE))
    
    ENERGY_ARGS=""
    if [ "$ENERGY" = true ]; then
        ENERGY_ARGS="--energy --energy-sources $ENERGY_SOURCES --energy-radius $ENERGY_RADIUS"
        ENERGY_ARGS="$ENERGY_ARGS --energy-reserve $ENERGY_RESERVE --energy-death $ENERGY_DEATH"
        ENERGY_ARGS="$ENERGY_ARGS --energy-max-sources $ENERGY_MAX_SOURCES"
        ENERGY_ARGS="$ENERGY_ARGS --energy-source-lifetime $ENERGY_SOURCE_LIFETIME"
        ENERGY_ARGS="$ENERGY_ARGS --energy-spawn-rate $ENERGY_SPAWN_RATE"
        [ "$ENERGY_RANDOM" = true ] && ENERGY_ARGS="$ENERGY_ARGS --energy-random"
    fi
    
    ./target/release/energetic-primordial-soup \
        --grid-width "$GRID_WIDTH" \
        --grid-height "$GRID_HEIGHT" \
        --seed "$SEED" \
        --mutation-prob "$MUTATION_PROB" \
        --steps-per-run "$STEPS_PER_RUN" \
        --max-epochs "$MAX_EPOCHS" \
        --neighbor-range "$NEIGHBOR_RANGE" \
        --frame-interval "$FRAME_INTERVAL" \
        --frames-dir "$FRAMES_DIR" \
        $ENERGY_ARGS \
        2>&1 | tee "$LOG_FILE"
fi

echo ""

# ============================================================================
# VIDEO GENERATION
# ============================================================================
echo ""
echo "Generating videos..."

# Main simulation video (from sim 0 / root frames)
MAIN_FRAME_COUNT=$(find "$FRAMES_DIR" -maxdepth 1 -name "*.ppm" 2>/dev/null | wc -l)
if [ "$MAIN_FRAME_COUNT" -gt 0 ]; then
    echo "  Main: $MAIN_FRAME_COUNT frames -> $VIDEO_FILE"
    ffmpeg -y -framerate "$VIDEO_FPS" \
        -pattern_type glob -i "${FRAMES_DIR}/*.ppm" \
        -c:v libx264 -pix_fmt yuv420p -crf 18 \
        "$VIDEO_FILE" 2>/dev/null
fi

# Per-simulation videos (sim_0, sim_1, etc.)
for sim_dir in "$FRAMES_DIR"/sim_*; do
    if [ -d "$sim_dir" ]; then
        sim_name=$(basename "$sim_dir")
        sim_frames=$(find "$sim_dir" -name "*.ppm" 2>/dev/null | wc -l)
        if [ "$sim_frames" -gt 0 ]; then
            sim_video="runs/${RUN_ID}_${sim_name}.mp4"
            echo "  $sim_name: $sim_frames frames -> $sim_video"
            ffmpeg -y -framerate "$VIDEO_FPS" \
                -pattern_type glob -i "${sim_dir}/*.ppm" \
                -c:v libx264 -pix_fmt yuv420p -crf 18 \
                "$sim_video" 2>/dev/null
        fi
    fi
done

# Cleanup frames if requested
if [ "$KEEP_FRAMES" = false ]; then
    TOTAL_FRAMES=$(find "$FRAMES_DIR" -name "*.ppm" 2>/dev/null | wc -l)
    if [ "$TOTAL_FRAMES" -gt 0 ]; then
        echo "  Cleaning up $TOTAL_FRAMES frame files..."
        rm -rf "$FRAMES_DIR"
    fi
fi

echo ""
echo "=============================================="
echo "Complete!"
echo "=============================================="
echo ""
echo "Results:"
echo "  Log: $LOG_FILE"
[ -f "$VIDEO_FILE" ] && echo "  Video: $VIDEO_FILE ($(ls -lh "$VIDEO_FILE" | awk '{print $5}'))"
for v in runs/${RUN_ID}_sim_*.mp4; do
    [ -f "$v" ] && echo "  Video: $v ($(ls -lh "$v" | awk '{print $5}'))"
done
[ -d "$FRAMES_DIR" ] && echo "  Frames: $FRAMES_DIR/"
echo ""

