# QAP-Solver

A flexible, modular toolkit for solving Quadratic Assignment Problem (QAP) instances with multiple exact and heuristic methods.

## Purpose

This framework provides a unified interface to solve QAP instances using different solver methods:
- **Exact methods**: RTL1 (Relaxation-based Tightened Linear) and Form3 (binary cubic formulation) with CPLEX or SCIP
- **Heuristic methods**: Tabu Search, Simulated Annealing, Genetic Algorithm
- **Unified configuration**: JSON-based config format for all solvers
- **Standard result format**: Consistent JSON output across all methods
- **File-based I/O**: Python and C++ components communicate via files (QAPLIB input, JSON output)

---

## Quick Start

### 1. Prepare an instance

Place a QAPLIB-format instance in `data/`:
```
data/chr12a.dat
```

### 2. Create a configuration

Copy an example config and modify if needed:
```bash
cp python/qap/modules/form3_cplex/example_config.json configs/my_config.json
```

### 3. Run the solver

```bash
python python/examples/solve.py \
  --module form3_cplex \
  --config configs/form3_cplex.json \
  --instance data/chr12a.dat \
  --output results/chr12a_result.json
```

**Available modules:**
- `rtl1_cplex` — RTL1 formulation with CPLEX
- `form3_cplex` — Form3 formulation with CPLEX
- `form3_scip` — Form3 formulation with SCIP (C++ backend)
- `local_search` — Heuristic methods (Tabu/SA/GA)

### 4. Check results

Output is saved to JSON with solution, objective value, bounds, and timing:
```json
{
  "instance": "chr12a",
  "solver": "form3_cplex",
  "objective": 11156,
  "lower_bound": 11156,
  "gap": 0.0,
  "time": 60.5,
  "assignment": [0, 3, 1, 5, 2, 4, 7, 6, 9, 8, 11, 10]
}
```

---

## Documentation

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Project structure, folder layout, module descriptions
- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** — Config file format, schema details, examples
- **[docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)** — QAPLIB instance format and result format
- **[docs/INSTALLATION.md](docs/INSTALLATION.md)** — Dependencies, setup, and build instructions
- **[docs/SCAFFOLDING_SUMMARY.md](docs/SCAFFOLDING_SUMMARY.md)** — Implementation roadmap and next steps

---

## Basic Directory Overview

```
QAP-Solver/
├── python/qap/modules/          # Solver modules (rtl1_cplex, form3_cplex, form3_scip, local_search)
├── python/examples/solve.py     # Main entry point (CLI dispatcher)
├── configs/                     # User configuration files (JSON)
├── data/                        # QAP instances (QAPLIB format)
└── results/                     # Solver outputs (JSON results)
```

For full details, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
