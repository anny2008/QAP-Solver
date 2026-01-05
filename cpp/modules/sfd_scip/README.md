# SFD SCIP Solver

Subgraph Flow Decomposition (SFD/Form3) solver for the Quadratic Assignment Problem using SCIP 9.2.0.

## Features

- **Flow Decomposition**: Two strategies available
  - `value_layer`: Layer-based decomposition (default)
  - `value_only`: Value-based decomposition
- **Warm-start Support**: Load initial solution from QAPLIB `.sln` files
- **Configurable**: Time limit, threads, relaxation, logging
- **SCIP Integration**: Uses SCIP 9.2.0 MIP solver

## Building

### Using CMake (recommended)

```bash
cd /home/local.isima.fr/antran/UFF/QAP-Solver/cpp/modules/sfd_scip
./build.sh
```

### Using Make

```bash
cd /home/local.isima.fr/antran/UFF/QAP-Solver/cpp/modules/sfd_scip
make
```

## Usage

### Basic usage

```bash
./sfd_solver <instance.dat>
```

### With warm-start

```bash
./sfd_solver nug12.dat --warmstart nug12.sln
```

### Full options

```bash
./sfd_solver <instance.dat> [options]

Options:
  --warmstart <file.sln>    : Warm-start solution file
  --decomposition <type>    : value_layer (default) or value_only
  --time <seconds>          : Time limit (default: 3600)
  --threads <n>             : Number of threads (default: 4)
  --relax                   : Use relaxed constraints
  --log                     : Enable solver output
```

### Examples

```bash
# Solve nug12 with warm-start, 5 minute limit, 8 threads, logging
./sfd_solver /path/to/nug12.dat \
  --warmstart /path/to/nug12.sln \
  --time 300 \
  --threads 8 \
  --log

# Solve with value_only decomposition
./sfd_solver chr12a.dat --decomposition value_only --time 600

# Solve with relaxed constraints
./sfd_solver had12.dat --relax --log
```

## Algorithm

The SFD formulation decomposes the QAP flow matrix into subgraphs and introduces:

**Variables:**
- `x[i,u]`: Binary assignment (facility i → location u)
- `e[i,j,k]`: Continuous edge flow for subgraph k

**Constraints:**
- Assignment: Each facility/location assigned exactly once
- Edge linking: `e[i,j,k] >= x[i,u] + Σ x[j,v] - 1` for edges (u,v) in subgraph k
- Flow conservation: Out-flow = In-flow for each node

**Objective:**
- `min Σ D[i,j] * f_k * e[i,j,k]`

## Implementation Notes

- Based on Python implementation in `QAP-Solver/python/qap/modules/sfd/`
- Decomposition logic from `QAP_New_formulation/FORM3/qap_new_formulation.py`
- Uses SCIP 9.2.0 for solving MIP
- Warm-start format: QAPLIB `.sln` files (1-indexed permutation)

## Performance

The SFD formulation typically produces:
- For n=12: ~144-1500 variables, ~1000-15000 constraints (depends on decomposition)
- Fewer subgraphs with `value_only` vs `value_layer`
- Warm-start can significantly reduce solve time

## Dependencies

- SCIP 9.2.0
- C++17 compiler (g++, clang++)
- CMake 3.10+ (optional, for CMake build)

## References

- Form3 formulation from QAP_New_formulation
- SCIP: https://scipopt.org/
