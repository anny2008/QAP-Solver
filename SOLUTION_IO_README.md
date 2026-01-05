# Unified Solution I/O Interface

## Overview

A unified solution I/O system for reading and writing QAP solutions in QAPLIB format across all solver modules. Available in both Python and C++.

**Key Features:**
- ✅ Consistent file format across all solvers
- ✅ Automatic 1-indexed ↔ 0-indexed conversion
- ✅ Type-safe operations with Solution class
- ✅ Easy warm-start integration
- ✅ Language-specific implementations (Python + C++)

## File Format (QAPLIB)

Solutions are stored in QAPLIB standard format:

```
n  optimal_value
location_1 location_2 ... location_n
```

**Example (nug12.sln):**
```
    12          578
12  7  9  3  4  8  11  1  5  6  10  2
```

- **Line 1:** `n` (problem size) and `optimal_value` (objective)
- **Line 2:** Assignment with locations **1-indexed** (facility i is assigned to location[i])

## Python Interface

### Location: `python/qap/core/solution_io.py`

#### Read Solution

```python
from qap.core.solution_io import read_solution

n, assignment_0indexed, objective = read_solution("chr12a.sln")
# n = 12
# assignment_0indexed = [11, 6, 8, 2, 3, 7, 10, 0, 4, 5, 9, 1]  # 0-indexed
# objective = 9552.0
```

**Parameters:**
- `filepath` (str): Path to .sln file

**Returns:**
- `n` (int): Problem size
- `assignment` (list): 0-indexed assignment array
- `objective` (float): Objective value

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If file format is invalid

#### Write Solution

```python
from qap.core.solution_io import write_solution

write_solution("output.sln", n=12, assignment=[11, 6, 8, 2, 3, 7, 10, 0, 4, 5, 9, 1], objective=9552.0)
# Writes in QAPLIB format with automatic 1-indexing
```

**Parameters:**
- `filepath` (str): Output path
- `n` (int): Problem size
- `assignment` (list): 0-indexed assignment
- `objective` (float): Objective value

#### Load Warm-Start

```python
from qap.core.solution_io import read_warmstart

warm_start = read_warmstart("chr12a.sln")
# warm_start = {(0, 11): 1.0, (1, 6): 1.0, (2, 8): 1.0, ...}

# Use with RTL1 solver
solver = RTL1SCIPSolver(config)
solution = solver.solve(problem, warm_start=warm_start)
```

**Returns:**
- `dict`: {(i, u): 1.0} format for solver warm-start

#### Solution Class

```python
from qap.core.solution_io import Solution

# Create solution object
sol = Solution(n=12, assignment=[11, 6, 8, 2, 3, 7, 10, 0, 4, 5, 9, 1], objective=9552.0)

# Convert to 1-indexed (for file writing)
assignment_1indexed = sol.to_1indexed()
# assignment_1indexed = [12, 7, 9, 3, 4, 8, 11, 1, 5, 6, 10, 2]

# Convert from 1-indexed (for file reading)
assignment_0indexed = Solution.from_1indexed([12, 7, 9, 3, 4, 8, 11, 1, 5, 6, 10, 2])
# assignment_0indexed = [11, 6, 8, 2, 3, 7, 10, 0, 4, 5, 9, 1]
```

### Integration with RTL1 SCIP Solver

```python
from qap.modules.rtl1_scip import RTL1SCIPSolver

solver = RTL1SCIPSolver(config)

# Method 1: Direct file loading
warm_start = solver.load_warmstart_from_file("previous_solution.sln")
solution = solver.solve(problem, warm_start=warm_start)

# Method 2: Using solution_io directly
from qap.core.solution_io import read_warmstart
warm_start = read_warmstart("solution.sln")
solution = solver.solve(problem, warm_start=warm_start)
```

## C++ Interface

### Location: `cpp/include/qap_solution_io.hpp`

Header-only library with identical functionality to Python version.

#### Include

```cpp
#include "qap_solution_io.hpp"
using namespace qap;
```

#### Read Solution

```cpp
#include "qap_solution_io.hpp"
using namespace qap;

int n;
std::vector<int> assignment;  // 0-indexed
double objective;

read_solution("chr12a.sln", n, assignment, objective);
// n = 12
// assignment = {11, 6, 8, 2, 3, 7, 10, 0, 4, 5, 9, 1}
// objective = 9552.0
```

**Parameters:**
- `filepath` (const std::string&): Path to .sln file
- `n` (int&): Output - problem size
- `assignment` (std::vector<int>&): Output - 0-indexed assignment
- `objective` (double&): Output - objective value

**Throws:**
- `std::runtime_error`: If file format is invalid or file not found

#### Write Solution

```cpp
write_solution("output.sln", 12, assignment, 9552.0);
// Writes in QAPLIB format with automatic 1-indexing
```

**Parameters:**
- `filepath` (const std::string&): Output path
- `n` (int): Problem size
- `assignment` (const std::vector<int>&): 0-indexed assignment
- `objective` (double): Objective value

#### Load Warm-Start

```cpp
std::map<std::pair<int, int>, double> warm_start = read_warmstart("chr12a.sln");
// warm_start[{0, 11}] = 1.0
// warm_start[{1, 6}] = 1.0
// ...
```

**Returns:**
- `std::map<std::pair<int,int>, double>`: {(i, u): 1.0} format

#### CMake Integration

```cmake
# Add to CMakeLists.txt
include_directories(${CMAKE_SOURCE_DIR}/cpp/include)

# In your C++ code
#include "qap_solution_io.hpp"
```

## Testing

### Python Tests

**Test unified I/O:**
```bash
cd QAP-Solver
python3 test_unified_solution_io.py
```

Output: Verifies read-write roundtrip and QAPLIB format validity.

**Test solver integration:**
```bash
cd QAP-Solver
python3 test_warmstart_integration.py
```

Output: Tests RTL1 solver with warm-start loaded from file.

### C++ Tests

To be created when C++ compilation is resolved (TBB dependency fix).

## Implementation Details

### Index Conversion

The module automatically handles 1-indexed ↔ 0-indexed conversion:

- **Files:** 1-indexed (QAPLIB format)
  ```
  12  7  9  3  ...  (1-indexed locations)
  ```

- **Code:** 0-indexed (array indices)
  ```python
  assignment = [11, 6, 8, 2, ...]  # Facility i → location assignment[i]
  ```

- **Conversion:**
  ```python
  # File → Code
  assignment_0indexed = [x - 1 for x in assignment_1indexed]
  
  # Code → File
  assignment_1indexed = [x + 1 for x in assignment_0indexed]
  ```

### Warm-Start Format

For solver integration, solutions are converted to warm-start dictionaries:

```python
# From solution file: assignment = [11, 6, 8, 2, ...]
warm_start = {
    (0, 11): 1.0,  # Facility 0 → Location 11
    (1, 6): 1.0,   # Facility 1 → Location 6
    (2, 8): 1.0,   # Facility 2 → Location 8
    ...
}
```

This format is used by:
- RTL1 SCIP solver: `solver.solve(problem, warm_start=warm_start)`
- Other solvers can implement similar pattern

## Examples

### Example 1: Save and Reload Solution

**Python:**
```python
from qap.core.solution_io import read_solution, write_solution

# Solve and get results
objective = 9552.0
assignment = [11, 6, 8, 2, 3, 7, 10, 0, 4, 5, 9, 1]

# Save to file
write_solution("my_solution.sln", n=12, assignment=assignment, objective=objective)

# Later, reload it
n, reloaded_assignment, reloaded_objective = read_solution("my_solution.sln")
assert reloaded_assignment == assignment
assert reloaded_objective == objective
```

### Example 2: Warm-Start from Previous Run

**Python:**
```python
from qap.modules.rtl1_scip import RTL1SCIPSolver
from qap.core import Problem

solver = RTL1SCIPSolver(config)
problem = Problem.from_qaplib("chr12a.dat")

# First run: save solution
solution1 = solver.solve_instance("chr12a.dat")
# solution1 has assignment and objective

# Second run: use warm-start from first run
warm_start = solver.load_warmstart_from_file("first_run.sln")
solution2 = solver.solve(problem, warm_start=warm_start)
# Typically faster or finds better solutions
```

### Example 3: Batch Processing

**Python:**
```python
from pathlib import Path
from qap.core.solution_io import read_solution

sln_dir = Path("./solutions")

# Process all solutions
for sln_file in sln_dir.glob("*.sln"):
    n, assignment, objective = read_solution(str(sln_file))
    print(f"{sln_file.name}: n={n}, obj={objective}")
```

## Troubleshooting

### File Format Issues

**Problem:** `ValueError: Invalid format in file.sln`

**Solutions:**
1. Verify file has 2 lines exactly
2. Check Line 1 has: `n optimal_value`
3. Check Line 2 has `n` space-separated integers (1-indexed)

**Example valid file:**
```
    12          578
12  7  9  3  4  8  11  1  5  6  10  2
```

### Index Confusion

**Problem:** Solutions are off by 1

**Solution:** Remember:
- **Input/Output files:** 1-indexed (QAPLIB format)
- **Python code:** 0-indexed (array indices)
- **Conversion:** Automatic in read_solution/write_solution

### Warm-Start Not Helping

**Problem:** Solver not faster with warm-start

**Reasons:**
1. Instance may be easy (solver already fast)
2. Warm-start solution may be poor initial point
3. Solver may need more configurations (branch-and-cut parameters)

## Version History

- **v1.0** (2024): Initial unified interface for Python and C++
  - QAPLIB format support
  - Automatic index conversion
  - Warm-start integration with RTL1 SCIP solver

## See Also

- [RTL1 SCIP Solver Documentation](../modules/rtl1_scip/README.md)
- [Problem Class Documentation](../core/problem.py)
- [QAPLIB Format Specification](https://www.opt.math.tugraz.at/qaplib/)
