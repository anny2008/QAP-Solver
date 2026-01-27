# RLT1 Volume Algorithm Solver

Implementation of the Volume Algorithm for solving the RLT1 (Reformulation-Linearization Technique Level 1) relaxation of the Quadratic Assignment Problem.

## Overview

The Volume Algorithm is a subgradient-based method that produces both:
- **Lower bounds** through Lagrangian relaxation
- **Primal solutions** through convex combinations of subgradient solutions

## Formulation

### RLT1 Relaxation

Variables:
- `x[i,u]`: Binary assignment variables (1 if facility i is at location u)
- `y[i,u,j,v]`: Linearization variables (= x[i,u] * x[j,v])

Objective:
```
minimize  sum_{i,j,u,v} d[i,j] * f[u,v] * y[i,u,j,v]
```

Constraints:
1. Assignment: `sum_i x[i,u] = 1` for all u (dual: mu[u])
2. Linking: `sum_v y[i,u,j,v] = x[i,u]` for all i,u,j (dual: theta[i,u,j])
3. Symmetry: `y[i,u,j,v] = y[j,v,i,u]` for all i,u,j,v (dual: lambda[i,u,j,v])

### Lagrangian Subproblem

The algorithm solves:
```
L(mu, lambda, theta) = min { original_objective - dual_penalties }
```

The subproblem decomposes into:
1. Compute `beta[i,u,v] = min_j { cost of y[i,u,j,v]=1 }`
2. Compute `alpha[i] = min_u { cost of x[i,u]=1 + sum_v beta[i,u,v] }`
3. Set variables according to optimal choices

## Usage

```bash
./rlt1_volume_solver <instance.dat> [options]
```

### Options

- `--time <seconds>`: Time limit (default: 3600)
- `--threads <n>`: Number of OpenMP threads (default: 8)
- `--log`: Enable detailed iteration logging
- `--output <file>`: Save final solution to file

### Examples

```bash
# Solve nug12 with 60 second time limit
./rlt1_volume_solver nug12.dat --time 60 --threads 4 --output nug12.sln

# Solve with detailed logging
./rlt1_volume_solver chr15a.dat --log --threads 8
```

## Algorithm Parameters

Default parameters (from qap.par):

- **lambdainit**: 0.1 - Initial step size multiplier
- **alphainit**: 1.0 - Initial convex combination parameter
- **alphamin**: 0.001 - Minimum alpha value
- **alphafactor**: 0.66 - Alpha reduction factor
- **alphaint**: 50 - Iterations between alpha adjustments

- **maxsgriters**: Based on time limit
- **primal_abs_precision**: 0.001 - Primal feasibility tolerance
- **gap_rel_precision**: 0.001 - Relative gap tolerance

- **greentestinvl**: 1 - Green iteration test interval
- **yellowtestinvl**: 4 - Yellow iteration test interval  
- **redtestinvl**: 20 - Red iteration test interval

- **ascent_first_check**: 500 - First ascent check iteration
- **ascent_check_invl**: 500 - Ascent check interval
- **minimum_rel_ascent**: 0.0001 - Minimum relative ascent required

## Output

The solver prints:
- **Lower bound**: Best Lagrangian lower bound found
- **Primal objective**: Objective value of extracted assignment
- **Time**: Total solution time
- Iteration progress with Lagrangian value (L), primal value (P), infeasibility

## Implementation Details

- **Parallelization**: Uses OpenMP for subproblem computation
  - Beta computation: Parallel over (i,u,v)
  - Alpha computation: Dynamic scheduling over facilities
  - Violation computation: Parallel constraint evaluation

- **Memory**: O(n^4) for y variables and lambda duals
- **Complexity**: Each iteration is O(n^4) due to RLT1 structure

## References

1. Barahona & Anbil (1998): "The Volume algorithm: producing primal solutions with a subgradient method", IBM Research Report RC 21103

2. Adams & Sherali (1990): "Linearization strategies for a class of zero-one mixed integer programming problems", Operations Research

## Building

```bash
make clean
make
```

Requires:
- C++17 compiler with OpenMP support
- Volume algorithm source files (VolVolume.cpp/.hpp)
