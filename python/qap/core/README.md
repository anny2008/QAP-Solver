# Core Utilities

The `python/qap/core/` and `cpp/core/` packages provide shared utilities used by all solver modules.

## Python Core (`python/qap/core/`)

### `problem.py` — Problem Class

Load and represent QAP instances.

```python
from python.qap.core import Problem

# Load from QAPLIB file
problem = Problem.from_qaplib('data/chr12a.dat')

# Access properties
print(f"Problem size: {problem.n}")
print(f"Flow matrix shape: {problem.F.shape}")

# Evaluate an assignment
assignment = [0, 1, 2, 3]
obj = problem.evaluate(assignment)
```

**Methods:**
- `__init__(n, F, D)` — Create problem from matrices
- `from_qaplib(filepath)` — Load from QAPLIB file
- `evaluate(assignment)` — Compute objective value

### `solution.py` — Solution Class

Store and manage solver results.

```python
from python.qap.core import Solution

# Create solution
solution = Solution(
    instance='chr12a',
    solver='form3_cplex',
    assignment=[0, 3, 1, 5, ...],
    objective=11156,
    lower_bound=11156,
    time=60.5
)

# Access properties
print(solution.gap)  # Optimality gap (%)
print(solution.to_dict())  # Convert to dict
```

**Properties:**
- `instance` (str) — Instance name
- `solver` (str) — Solver identifier
- `assignment` (list) — Permutation
- `objective` (float | None) — Best objective value
- `lower_bound` (float | None) — Lower bound
- `time` (float) — Execution time
- `gap` (property) — Optimality gap (%)

**Methods:**
- `to_dict()` — Convert to dictionary

### `io.py` — Input/Output Functions

Load and save results in JSON format.

```python
from python.qap.core import write_result, read_result

# Write result
write_result(solution, 'results/chr12a_result.json')

# Read result
solution = read_result('results/chr12a_result.json')

# Access format specification
from python.qap.core import RESULT_FORMAT
print(RESULT_FORMAT)
```

**Functions:**
- `write_result(solution, filepath)` — Save solution to JSON
- `read_result(filepath)` — Load solution from JSON

**Constants:**
- `RESULT_FORMAT` — Schema/specification of result format

---

## C++ Core (`cpp/core/`)

Header-only library for C++ solver backends (e.g., SCIP).

### `problem.h` — Problem Class

```cpp
#include "cpp/core/problem.h"

// Load from QAPLIB file
Problem problem = Problem::fromQAPLIB("data/chr12a.dat");

// Access properties
int n = problem.n;
std::vector<std::vector<double>> F = problem.F;

// Evaluate assignment
std::vector<int> assignment = {0, 1, 2, 3};
double obj = problem.evaluate(assignment);
```

**Methods:**
- `Problem(n, F, D)` — Constructor
- `fromQAPLIB(filepath)` — Load from file
- `evaluate(assignment)` — Compute objective

### `solution.h` — Solution Class

```cpp
#include "cpp/core/solution.h"

// Create solution
Solution solution("chr12a", "form3_scip");
solution.assignment = {0, 3, 1, 5, ...};
solution.objective = 11156;
solution.lower_bound = 11156;
solution.time = 60.5;

// Get gap
double gap = solution.getGap();

// Convert to JSON and save
json result_json = solution.toJSON();
solution.write("results/chr12a_result.json");
```

**Properties:**
- `instance` (std::string) — Instance name
- `solver` (std::string) — Solver identifier
- `assignment` (vector<int>) — Permutation
- `objective` (double) — Objective value (-1 = not set)
- `lower_bound` (double) — Lower bound (-1 = not set)
- `time` (double) — Execution time

**Methods:**
- `getGap()` — Calculate gap (%)
- `toJSON()` — Convert to JSON
- `write(filepath)` — Save to JSON file

---

## Usage in Solver Modules

All solver modules should follow this pattern:

```python
# solver.py in a module

from python.qap.core import Problem, Solution, write_result
import json

class MyModule:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)
    
    def solve(self, instance_path, output_path=None):
        # 1. Load problem
        problem = Problem.from_qaplib(instance_path)
        
        # 2. Solve problem
        assignment = self._solve_impl(problem)
        
        # 3. Create solution
        solution = Solution(
            instance=Path(instance_path).stem,
            solver=self.config['solver'],
            assignment=assignment,
            objective=problem.evaluate(assignment),
            lower_bound=compute_lower_bound(problem),
            time=elapsed_time
        )
        
        # 4. Save result
        if output_path:
            write_result(solution, output_path)
        
        return solution
```

---

## Dependencies

**Python:**
- `numpy` — Numerical computations
- `pathlib` — File handling
- `json` — JSON serialization

**C++:**
- `nlohmann/json` — JSON library (for Solution::toJSON)
- Standard C++11 or later
