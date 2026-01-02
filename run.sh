#!/bin/bash
#
# BFF Primordial Soup Simulation Runner
# ======================================
#
# USAGE:
#   ./run.sh                     # Uses config.yaml (if exists) or defaults
#   ./run.sh config.yaml         # Use specific config file
#   MAX_EPOCHS=5000 ./run.sh     # Override specific values
#   BACKEND=cuda ./run.sh        # Override backend (cuda/wgpu/cpu)
#
# BACKEND SELECTION (in order of priority):
#   1. BACKEND env var           # BACKEND=cuda ./run.sh
#   2. config.yaml backend:      # backend: "cuda" in config file
#   3. Default: wgpu             # Cross-platform GPU via Vulkan/Metal
#
# DIRECTORY BEHAVIOR:
#   With config file: Uses frames_dir and checkpoint.path from config.yaml
#   Without config:   Uses runs/{timestamp}_frames/ automatically
#   
#   To override and use runs/ folder even with config:
#     USE_RUN_DIRS=true ./run.sh config.yaml
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
VIDEO_FPS=${VIDEO_FPS:-15}
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

# When using a config file, respect its directories by default
# Set USE_RUN_DIRS=true to override and use runs/{timestamp}/ instead
if [ -n "$CONFIG_FILE" ] && [ "${USE_RUN_DIRS:-false}" != true ]; then
    # Config file mode: use directories from config.yaml
    FRAMES_DIR=""
    LOG_FILE="simulation_${RUN_ID}.log"
else
    # No config or explicit override: use runs/ folder
    FRAMES_DIR="${FRAMES_DIR:-runs/${RUN_ID}_frames}"
    LOG_FILE="runs/${RUN_ID}_log.txt"
    mkdir -p runs
    mkdir -p "$FRAMES_DIR"
fi

# ============================================================================
# BUILD
# ============================================================================
# Detect backend from config file, env var, or default to wgpu
# Priority: BACKEND env var > config file > default (wgpu)
if [ -n "${BACKEND:-}" ]; then
    SELECTED_BACKEND="$BACKEND"
elif [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    # Extract backend from config (handles quoted and unquoted values)
    SELECTED_BACKEND=$(grep -E "^backend:" "$CONFIG_FILE" | head -1 | sed 's/backend:\s*["'\'']\?\([^"'\''#]*\)["'\'']\?.*/\1/' | xargs)
fi
SELECTED_BACKEND="${SELECTED_BACKEND:-wgpu}"

echo "Building..."
case "$SELECTED_BACKEND" in
    cuda|CUDA)
        echo "  Backend: CUDA (NVIDIA GPU, no buffer limit)"
        PATH="/usr/local/cuda/bin:$PATH" cargo build --release --features cuda 2>&1 | tail -3
        ;;
    cpu|CPU)
        echo "  Backend: CPU (slow fallback)"
        cargo build --release 2>&1 | tail -3
        ;;
    wgpu|WGPU|vulkan|metal|*)
        echo "  Backend: wgpu/Vulkan (cross-platform GPU)"
        cargo build --release --features wgpu-compute 2>&1 | tail -3
        ;;
esac
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
    
    # Only override frames-dir if explicitly set (not using config dirs)
    FRAMES_ARG=""
    [ -n "$FRAMES_DIR" ] && FRAMES_ARG="--frames-dir $FRAMES_DIR"
    
    ./target/release/energetic-primordial-soup \
        --config "$CONFIG_FILE" \
        $FRAMES_ARG \
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

# If using config dirs, extract base frames_dir from config file
if [ -z "$FRAMES_DIR" ] && [ -n "$CONFIG_FILE" ]; then
    BASE_FRAMES_DIR=$(grep -E "^\s*frames_dir:" "$CONFIG_FILE" | head -1 | sed 's/.*frames_dir:\s*["'\'']\?\([^"'\''#]*\)["'\'']\?.*/\1/' | xargs)
    # The Rust program appends a timestamp, so find the most recently created matching directory
    FRAMES_DIR=$(ls -dt "${BASE_FRAMES_DIR}"_* 2>/dev/null | head -1)
    if [ -z "$FRAMES_DIR" ]; then
        # Fallback to exact match if no timestamped version found
        FRAMES_DIR="$BASE_FRAMES_DIR"
    fi
    echo "  Using frames directory: $FRAMES_DIR"
fi

# Save video inside the frames directory (alongside the frames)
VIDEO_FILE="${FRAMES_DIR}/simulation.mp4"

# Copy the log file into the frames directory for complete run archive
if [ -f "$LOG_FILE" ] && [ -d "$FRAMES_DIR" ]; then
    cp "$LOG_FILE" "${FRAMES_DIR}/simulation.log"
fi

# Mega simulation video (combined multi-sim view) - check for mega_epoch_*.png
MEGA_FRAME_COUNT=$(find "$FRAMES_DIR" -maxdepth 1 -name "mega_epoch_*.png" 2>/dev/null | wc -l)
if [ "$MEGA_FRAME_COUNT" -gt 0 ]; then
    MEGA_VIDEO="${FRAMES_DIR}/mega_simulation.mp4"
    echo "  Mega: $MEGA_FRAME_COUNT frames -> $MEGA_VIDEO"
    ffmpeg -y -framerate "$VIDEO_FPS" \
        -pattern_type glob -i "${FRAMES_DIR}/mega_epoch_*.png" \
        -c:v libx264 -pix_fmt yuv420p -crf 18 \
        "$MEGA_VIDEO" 2>/dev/null
fi

# Main simulation video (from sim 0 / root frames) - check for numbered frames (not mega_)
MAIN_FRAME_COUNT=$(find "$FRAMES_DIR" -maxdepth 1 \( -name "[0-9]*.ppm" -o -name "[0-9]*.png" \) 2>/dev/null | wc -l)
if [ "$MAIN_FRAME_COUNT" -gt 0 ]; then
    # Detect format
    if ls "${FRAMES_DIR}"/[0-9]*.png 1>/dev/null 2>&1; then
        FRAME_PATTERN="${FRAMES_DIR}/[0-9]*.png"
    else
        FRAME_PATTERN="${FRAMES_DIR}/[0-9]*.ppm"
    fi
    echo "  Main: $MAIN_FRAME_COUNT frames -> $VIDEO_FILE"
    ffmpeg -y -framerate "$VIDEO_FPS" \
        -pattern_type glob -i "$FRAME_PATTERN" \
        -c:v libx264 -pix_fmt yuv420p -crf 18 \
        "$VIDEO_FILE" 2>/dev/null
fi

# Per-simulation videos (sim_0, sim_1, etc.) - saved inside frames directory
for sim_dir in "$FRAMES_DIR"/sim_*; do
    if [ -d "$sim_dir" ]; then
        sim_name=$(basename "$sim_dir")
        sim_frames=$(find "$sim_dir" \( -name "*.ppm" -o -name "*.png" \) 2>/dev/null | wc -l)
        if [ "$sim_frames" -gt 0 ]; then
            sim_video="${FRAMES_DIR}/${sim_name}.mp4"
            # Detect format
            if ls "${sim_dir}"/*.png 1>/dev/null 2>&1; then
                SIM_PATTERN="${sim_dir}/*.png"
            else
                SIM_PATTERN="${sim_dir}/*.ppm"
            fi
            echo "  $sim_name: $sim_frames frames -> $sim_video"
            ffmpeg -y -framerate "$VIDEO_FPS" \
                -pattern_type glob -i "$SIM_PATTERN" \
                -c:v libx264 -pix_fmt yuv420p -crf 18 \
                "$sim_video" 2>/dev/null
        fi
    fi
done

# Cleanup frames if requested (only for auto-generated run dirs)
if [ "$KEEP_FRAMES" = false ] && [[ "$FRAMES_DIR" == runs/* ]]; then
    TOTAL_FRAMES=$(find "$FRAMES_DIR" \( -name "*.ppm" -o -name "*.png" \) 2>/dev/null | wc -l)
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
if [ -d "$FRAMES_DIR" ]; then
    echo "  Output directory: $FRAMES_DIR/"
    echo "  Contains:"
    [ -f "${FRAMES_DIR}/simulation.log" ] && echo "    - simulation.log"
    [ -f "${FRAMES_DIR}/mega_simulation.mp4" ] && echo "    - mega_simulation.mp4 ($(ls -lh "${FRAMES_DIR}/mega_simulation.mp4" | awk '{print $5}'))"
    [ -f "$VIDEO_FILE" ] && echo "    - simulation.mp4 ($(ls -lh "$VIDEO_FILE" | awk '{print $5}'))"
    for v in "$FRAMES_DIR"/sim_*.mp4; do
        [ -f "$v" ] && echo "    - $(basename "$v") ($(ls -lh "$v" | awk '{print $5}'))"
    done
    FRAME_COUNT=$(find "$FRAMES_DIR" -name "*.png" -o -name "*.ppm" 2>/dev/null | wc -l)
    [ "$FRAME_COUNT" -gt 0 ] && echo "    - $FRAME_COUNT frame images"
fi
echo ""
