#!/bin/bash
#
# Build script for RTL1 SCIP solver
#
# Usage: ./build_rtl1_scip.sh [clean]
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="$SCRIPT_DIR/cpp/modules/rtl1_scip/build"
SCIP_DIR="/home/local.isima.fr/antran/UFF/scip-9.2.0"

echo "=========================================="
echo "Building RTL1 SCIP Solver"
echo "=========================================="
echo "Build directory: $BUILD_DIR"
echo "SCIP directory: $SCIP_DIR"
echo ""

# Check if SCIP is available
if [ ! -d "$SCIP_DIR" ]; then
    echo "ERROR: SCIP directory not found at $SCIP_DIR"
    exit 1
fi

# Clean if requested
if [ "$1" == "clean" ]; then
    echo "Cleaning old build..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Run cmake
echo "Running CMake..."
cmake -DSCIP_DIR="$SCIP_DIR" ..

# Build
echo ""
echo "Building..."
make -j$(nproc)

echo ""
echo "=========================================="
echo "Build complete!"
echo "Library: $BUILD_DIR/librtl1_scip.so"
echo "=========================================="

# Check if library exists
if [ -f "$BUILD_DIR/librtl1_scip.so" ]; then
    echo "✓ Shared library built successfully"
    ls -lh "$BUILD_DIR/librtl1_scip.so"
else
    echo "✗ Error: Shared library not found"
    exit 1
fi
