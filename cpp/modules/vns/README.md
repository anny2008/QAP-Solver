# Variable Neighborhood Search (VNS) Module

This module implements the Variable Neighborhood Search algorithm for the Quadratic Assignment Problem (QAP) based on Pradeepmon 2020.

## Usage

1. **Build the solver:**
   ```sh
   make
   ```

2. **Run the solver:**
   ```sh
   ./vns_solver <problem_file> [--config <config.json>] [--max_iters N] [--seed S]
   ```
   - `<problem_file>`: Path to a QAP instance file compatible with your core/problem.h loader.
   - `--config <config.json>`: Optional JSON config file (see configs/vns.json)
   - `--max_iters N`: Override max iterations
   - `--seed S`: Override random seed


## Configuration
- Use `configs/vns.json` to specify parameters such as max_iterations, neighborhoods, seed, and input_file.
- CLI arguments override config file values.
- Integrates with your existing QAP core and solution classes.

## Files
- `vns_solver.hpp` / `vns_solver.cpp`: VNS algorithm implementation
- `main.cpp`: Command-line interface
- `Makefile`: Build instructions

## Reference
Pradeepmon, K. P., et al. "A variable neighborhood search for the quadratic assignment problem." (2020).
