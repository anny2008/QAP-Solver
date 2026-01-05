#!/bin/bash
# Build script for SFD SCIP solver

set -e

# Configuration
BUILD_DIR="build"
SCIP_DIR="/home/local.isima.fr/antran/UFF/scip-9.2.0"

echo "Building SFD SCIP solver..."

# Create build directory
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# Run CMake
cmake .. -DSCIP_DIR=${SCIP_DIR}

# Build
make -j4

echo "Build complete! Executable: ${BUILD_DIR}/sfd_solver"
