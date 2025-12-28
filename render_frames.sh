#!/bin/bash
# ============================================================================
# BFF Primordial Soup - Post-Simulation Frame Renderer
# ============================================================================
#
# Renders frames from saved raw data files after simulation completes.
# This allows you to run simulations at maximum speed without rendering,
# then generate visualizations later with different settings.
#
# Usage:
#   ./render_frames.sh <raw_data_dir> <output_dir> [options]
#
# Examples:
#   ./render_frames.sh /path/to/raw_data /path/to/frames
#   ./render_frames.sh /path/to/raw_data /path/to/frames --format png --scale 2
#   FORMAT=png SCALE=4 ./render_frames.sh /path/to/raw_data /path/to/frames
#
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings (can be overridden by environment variables or CLI args)
FORMAT="${FORMAT:-png}"           # png, ppm
SCALE="${SCALE:-1}"               # Downscale factor (1 = full, 4 = 1/4 size)
CONFIG="${CONFIG:-config.yaml}"   # Config file for grid dimensions
BINARY="${BINARY:-./target/release/energetic-primordial-soup}"

# Print usage
usage() {
    echo -e "${BLUE}BFF Post-Simulation Frame Renderer${NC}"
    echo ""
    echo "Usage: $0 <raw_data_dir> <output_dir> [options]"
    echo ""
    echo "Arguments:"
    echo "  raw_data_dir    Directory containing raw_epoch_*.bin files"
    echo "  output_dir      Directory to save rendered frames"
    echo ""
    echo "Options:"
    echo "  --format <fmt>  Output format: png (default) or ppm"
    echo "  --scale <n>     Downscale factor (default: 1 = full resolution)"
    echo "  --config <file> Config file path (default: config.yaml)"
    echo "  --binary <path> Path to compiled binary"
    echo "  --help          Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  FORMAT          Output format (png, ppm)"
    echo "  SCALE           Downscale factor"
    echo "  CONFIG          Config file path"
    echo "  BINARY          Binary path"
    echo ""
    echo "Examples:"
    echo "  $0 /data/raw_data /data/frames"
    echo "  $0 /data/raw_data /data/frames --format png --scale 4"
    echo "  FORMAT=ppm SCALE=2 $0 /data/raw_data /data/frames"
    exit 1
}

# Parse arguments
RAW_DIR=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --scale)
            SCALE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --binary)
            BINARY="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        -*)
            echo -e "${RED}Error: Unknown option $1${NC}"
            usage
            ;;
        *)
            if [[ -z "$RAW_DIR" ]]; then
                RAW_DIR="$1"
            elif [[ -z "$OUTPUT_DIR" ]]; then
                OUTPUT_DIR="$1"
            else
                echo -e "${RED}Error: Too many arguments${NC}"
                usage
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$RAW_DIR" ]] || [[ -z "$OUTPUT_DIR" ]]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    usage
fi

# Check raw data directory exists
if [[ ! -d "$RAW_DIR" ]]; then
    echo -e "${RED}Error: Raw data directory not found: $RAW_DIR${NC}"
    exit 1
fi

# Count raw data files
RAW_FILES=$(find "$RAW_DIR" -name "raw_epoch_*.bin" 2>/dev/null | wc -l)
if [[ "$RAW_FILES" -eq 0 ]]; then
    echo -e "${RED}Error: No raw_epoch_*.bin files found in $RAW_DIR${NC}"
    exit 1
fi

# Check binary exists
if [[ ! -x "$BINARY" ]]; then
    echo -e "${YELLOW}Binary not found at $BINARY, attempting to build...${NC}"
    
    # Try to build
    if command -v cargo &> /dev/null; then
        echo "Building with cargo..."
        cargo build --release --features wgpu-compute
        
        if [[ ! -x "$BINARY" ]]; then
            echo -e "${RED}Error: Build failed or binary not found${NC}"
            exit 1
        fi
    else
        echo -e "${RED}Error: cargo not found and binary missing${NC}"
        exit 1
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print settings
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}BFF Post-Simulation Frame Renderer${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "Raw data directory: ${GREEN}$RAW_DIR${NC}"
echo -e "Output directory:   ${GREEN}$OUTPUT_DIR${NC}"
echo -e "Raw files found:    ${GREEN}$RAW_FILES${NC}"
echo -e "Format:             ${GREEN}$FORMAT${NC}"
echo -e "Scale:              ${GREEN}1/$SCALE${NC}"
echo ""

# Build command
CMD="$BINARY"

# Add config if it exists
if [[ -f "$CONFIG" ]]; then
    CMD="$CMD --config $CONFIG"
fi

# Add render-raw and other options
CMD="$CMD --render-raw $RAW_DIR --frames-dir $OUTPUT_DIR"

# Note: frame_format and thumbnail_scale are read from config
# We could add CLI overrides if needed

echo -e "Running: ${YELLOW}$CMD${NC}"
echo ""

# Run the renderer
START_TIME=$(date +%s)

$CMD

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Count output files
OUTPUT_FILES=$(find "$OUTPUT_DIR" -name "*.${FORMAT}" 2>/dev/null | wc -l)

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Rendering Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "Time elapsed:  ${ELAPSED}s"
echo -e "Frames output: $OUTPUT_FILES"
echo -e "Output dir:    $OUTPUT_DIR"
echo ""

# Suggest video generation
echo -e "${BLUE}To generate a video from frames:${NC}"
echo "  ffmpeg -framerate 30 -pattern_type glob -i '$OUTPUT_DIR/*.${FORMAT}' \\"
echo "    -c:v libx264 -pix_fmt yuv420p -crf 18 output.mp4"
echo ""

