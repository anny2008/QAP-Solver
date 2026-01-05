# RTL1 SCIP Solver Configuration

RTL1 formulation solver using SCIP solver backend via C++ integration.

## Features

- **Relaxation-based Tightened Linear (RTL1)** formulation
- **C++ Performance** - SCIP solver via ctypes interface
- **Warm-start Support** - Initialize with known solutions
- **Fixed Variables** - Partial assignment support
- **Flexible Variables** - Binary or relaxed 0-1 continuous mode

## Configuration

```json
{
  "solver": "rtl1_scip",
  "formulation": "rtl1",
  "is_relax": false,
  "time_limit": 120,
  "threads": 8,
  "log_output": false,
  "preprocessing_symmetry": 5
}
```

## Build Instructions

```bash
cd cpp/modules/rtl1_scip
mkdir -p build
cd build
cmake ..
make
```

Requires SCIP 9.2.0+ installed at `/home/local.isima.fr/antran/UFF/scip-9.2.0`

## Usage

See `python/examples/demo_rtl1_scip.py` for examples.
