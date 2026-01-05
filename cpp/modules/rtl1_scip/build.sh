#!/bin/bash
# Build RTL1 SCIP Solver

set -e

cd "$(dirname "$0")"

# Check if SCIP is installed
if ! find /opt -name "libscip*" 2>/dev/null | grep -q .; then
    echo "SCIP not found in /opt. Checking alternative locations..."
    # Try to find SCIP
    SCIP_DIR=$(find / -name "SCIPConfig.cmake" 2>/dev/null | head -1 | xargs dirname)
    if [ -z "$SCIP_DIR" ]; then
        echo "Error: SCIP not found. Please install SCIP 9.2.0 first."
        exit 1
    fi
else
    SCIP_DIR=$(find /opt -name "SCIPConfig.cmake" 2>/dev/null | head -1 | xargs dirname)
fi

echo "Using SCIP at: $SCIP_DIR"

# Create build directory
mkdir -p build
cd build

# Configure and build
cmake -DSCIP_DIR="$SCIP_DIR" ..
make -j4

echo "Build complete! Binary: $(pwd)/rtl1_solver"
