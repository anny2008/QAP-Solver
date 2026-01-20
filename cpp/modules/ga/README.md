# Genetic Algorithm (GA) Module for QAP (2024 Misevičius et al.)

This module implements a modern permutation-based Genetic Algorithm for the Quadratic Assignment Problem, following the 2024 Misevičius et al. paper.

## Features
- Permutation encoding for QAP
- PMX/Order crossover, swap/insert/reverse mutation
- Tournament selection, elitism
- Optional hybridization with local search
- Configurable via JSON

## Usage

1. **Build the solver:**
   ```sh
   make
   ```
2. **Run the solver:**
   ```sh
   ./ga_solver <problem_file> [--config <config.json>] [--seed S]
   ```
   - `<problem_file>`: Path to a QAP instance file
   - `--config <config.json>`: Optional JSON config file (see configs/ga.json)
   - `--seed S`: Override random seed (integer or "time")

## Configuration Options
See `configs/ga.json` for all tunable parameters:
- population_size
- generations
- crossover_rate
- mutation_rate
- tournament_size
- seed
- crossover ("pmx" or "ox")
- mutation ("swap", "insert", "reverse")
- use_local_search (true/false)
- elitism (true/false)

## Reference
Misevičius, V., et al. "A Modern Genetic Algorithm for the Quadratic Assignment Problem." (2024).
