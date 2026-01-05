# QAP-Solver

This repository will host a flexible toolkit for solving Quadratic Assignment Problem (QAP) instances with both exact and heuristic methods.

## Purpose
- Provide multiple solving approaches (MIP/branch-and-bound, decomposition/volume, metaheuristics) under a single, consistent interface.
- Work directly with datasets in the standard QAPLIB format (square flow and distance matrices, .dat/.qap style files).
- Enable fair benchmarking across methods using the same problem definitions and reporting metrics (objective, lower/upper bounds, gap, time).

## Data
- Instances are expected in QAPLIB format. Place them under a data/ directory (e.g., data/QAPLIB/instance.dat).
- Known optimal or best-known solutions (when available) can be stored alongside the instances for warm starts or validation.

## Current status
- Fresh repository scaffold. Codebase structure and implementations will be added next.

## Next steps
- Define the code layout (core problem/IO, solver interfaces, exact and heuristic modules).
- Add initial loaders for QAPLIB and a minimal runner script to solve a single instance.
- Implement the first solver baseline and CI smoke tests.
