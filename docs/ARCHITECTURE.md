# Architecture

## Directory Structure

```
QAP-Solver/
├── python/
│   ├── qap/
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── problem.py          # QAP problem definition and utilities
│   │   │   ├── io.py               # QAPLIB loader, result writer
│   │   │   └── solution.py         # Solution class definition
│   │   ├── modules/
│   │   │   ├── rtl1_cplex/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── solver.py       # RTL1 formulation & CPLEX solver
│   │   │   │   ├── config_schema.json
│   │   │   │   ├── example_config.json
│   │   │   │   └── README.md
│   │   │   ├── form3_cplex/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── solver.py       # Form3 formulation & CPLEX solver
│   │   │   │   ├── config_schema.json
│   │   │   │   ├── example_config.json
│   │   │   │   └── README.md
│   │   │   ├── form3_scip/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── solver.py       # Form3 formulation & SCIP (C++ backend)
│   │   │   │   ├── config_schema.json
│   │   │   │   ├── example_config.json
│   │   │   │   └── README.md
│   │   │   └── local_search/
│   │   │       ├── __init__.py
│   │   │       ├── solver.py       # Tabu search, simulated annealing, GA
│   │   │       ├── config_schema.json
│   │   │       ├── example_config.json
│   │   │       └── README.md
│   └── examples/
│       ├── solve.py                # Main entry point (CLI dispatcher)
│       └── benchmark.py            # Benchmarking utilities
├── cpp/
│   └── modules/
│       └── form3_scip/
│           ├── CMakeLists.txt      # C++ build configuration
│           ├── src/
│           │   └── form3_scip.cpp  # Form3 SCIP solver implementation
│           └── bin/
│               └── form3_scip      # Compiled executable
├── configs/                        # User-provided configuration files
│   ├── rtl1_cplex.json
│   ├── form3_cplex.json
│   ├── form3_scip.json
│   └── local_search.json
├── data/                           # QAP instances (QAPLIB format)
├── results/                        # Output directory for solver results
├── README.md                       # Quick start guide
├── ARCHITECTURE.md                 # This file
├── CONFIGURATION.md                # Config format documentation
├── DATA_FORMAT.md                  # Instance and result format
├── INSTALLATION.md                 # Setup and dependencies
└── SCAFFOLDING_SUMMARY.md          # Implementation roadmap
```

## Solver Modules

### 1. RTL1 CPLEX
- **Solver**: CPLEX (via Gurobi/DOCplex)
- **Formulation**: Relaxation-based Tightened Linear (RTL1)
- **Type**: Exact (branch-and-bound)
- **Location**: `python/qap/modules/rtl1_cplex/`
- **Config**: See [CONFIGURATION.md](CONFIGURATION.md)

### 2. Form3 CPLEX
- **Solver**: CPLEX (via Gurobi/DOCplex)
- **Formulation**: Binary cubic Form3 (with value_layer or value_only decomposition)
- **Type**: Exact (branch-and-bound)
- **Location**: `python/qap/modules/form3_cplex/`
- **Config**: See [CONFIGURATION.md](CONFIGURATION.md)

### 3. Form3 SCIP
- **Solver**: SCIP (C++ backend via file I/O)
- **Formulation**: Binary cubic Form3 (with value_layer or value_only decomposition)
- **Type**: Exact (branch-and-cut)
- **Location**: `python/qap/modules/form3_scip/`
- **Config**: See [CONFIGURATION.md](CONFIGURATION.md)

### 4. Local Search
- **Methods**: Tabu Search, Simulated Annealing, Genetic Algorithm
- **Type**: Heuristic (approximate, fast)
- **Location**: `python/qap/modules/local_search/`
- **Config**: See [CONFIGURATION.md](CONFIGURATION.md)

## Design Principles

- **Modularity**: Each solver is independent; no cross-module dependencies.
- **File-Based I/O**: Python and C++ communicate only via files (no direct marshaling).
- **Config-Driven**: All parameters specified in JSON; no hardcoded values.
- **Unified Result Format**: All solvers output the same JSON structure.
- **QAPLIB Standard**: Work directly with standard instance format.

## Core Utilities

The `python/qap/core/` package provides shared utilities:
- **problem.py** — QAP problem class (n, F, D matrices)
- **io.py** — QAPLIB instance loader, JSON result writer
- **solution.py** — Solution class (assignment permutation, objective)

These are used by all solver modules to avoid duplication.

## CLI Entry Point

**File**: `python/examples/solve.py`

Main dispatcher that:
1. Parses command-line arguments (--module, --config, --instance, --output)
2. Validates config against module's JSON schema
3. Routes to correct solver module
4. Writes standardized result JSON
