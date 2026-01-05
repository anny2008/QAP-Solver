# SFD Solver - Subgraph Flow Decomposition

## Overview

The **Subgraph Flow Decomposition (SFD)** solver is a decomposition-based formulation for the Quadratic Assignment Problem (QAP). It decomposes the problem into value layers and solves using CPLEX with lazy constraint callbacks for cutting planes.

**Key Features:**
- ✅ Decomposition-based approach (value layers or value-only)
- ✅ Lazy constraint callback for valid inequality generation
- ✅ CPLEX Python (DOcplex) integration
- ✅ Warm-start support via unified solution I/O
- ✅ Configurable decomposition strategies
- ✅ Support for relaxed and binary constraints

## Name: "SFD" (Subgraph Flow Decomposition)

Instead of generic "Form3", we use **SFD** to clearly indicate:
- **S**ubgraph-based approach (decomposes into subgraphs)
- **F**low-based (tracks flow on edges between assignments)
- **D**ecomposition (systematic decomposition of the problem)

## Mathematical Formulation

### Variables
- **x[i,u]** ∈ {0,1}: Binary assignment variable (facility i → location u)
- **e[i,j,k]** ∈ [0,1]: Continuous edge variable for facility pair (i,j) in subgraph k

### Objective
```
minimize: Σ distances[i,j] × f_k × e[i,j,k]
```
where f_k is the weight/value of subgraph k.

### Constraints

1. **Assignment Constraints:**
   - Each facility assigned to exactly one location: Σ_u x[i,u] = 1 for all i
   - Each location assigned to exactly one facility: Σ_i x[i,u] = 1 for all u

2. **Edge-Assignment Linking (per subgraph k):**
   ```
   e[i,j,k] >= x[i,u] + Σ_{v:(u,v)∈G_k} x[j,v] - 1
   ```
   for all i,j, u ∈ G_n_k

3. **Flow Conservation (per subgraph k):**
   ```
   Σ_j e[i,j,k] = Σ_u x[i,u] × degree_out(u)
   Σ_j e[j,i,k] = Σ_v x[i,v] × degree_in(v)
   ```

4. **Lazy Constraints (Valid Inequalities):**
   When two assignments (i,u) and (j,v) are fixed at 1, valid cuts are added based on:
   ```
   Σ_k Σ_g δ_u_ij(k,g) x[k,g] - M·x[i,u] + Σ_k Σ_g δ_v_ji(k,g) x[k,g] - M·x[j,v] ≥ -2M
   ```

## Decomposition Strategies

### Value Layer Decomposition
Decomposes flows by their value levels. Each value layer becomes a subgraph.
```python
subgraphs = decompose_flow_by_value_layer(problem)
```

### Value Only Decomposition
Simpler decomposition based on distinct flow values only.
```python
subgraphs = decompose_flow_by_value_only(problem)
```

## Usage

### Python Interface

#### Basic Solving

```python
from qap.modules.sfd import SFDSolver
from qap.core import Problem
from qap_new_formulation import decompose_flow_by_value_layer

# Create solver
config = {
    "solver": "sfd",
    "formulation": "sfd",
    "decomposition": "value_layer",
    "use_cuts": True,
    "time_limit": 3600,
}
solver = SFDSolver(config)

# Load problem
problem = Problem.from_qaplib("chr12a.dat")

# Create decomposition
subgraphs = decompose_flow_by_value_layer(problem)

# Solve
solution = solver.solve(problem, subgraphs)
print(f"Objective: {solution.objective}")
print(f"Time: {solution.time:.2f}s")
```

#### With Warm-Start

```python
# Load warm-start from file
warm_start = solver.load_warmstart_from_file("previous.sln")

# Solve with warm-start
solution = solver.solve(problem, subgraphs, warm_start=warm_start)
```

#### With Fixed Variables

```python
# Fix some assignments
fixed_vars = [(0, 5), (1, 3)]  # Facility 0→Location 5, Facility 1→Location 3

# Solve with fixed variables
solution = solver.solve(problem, subgraphs, fixed_variables=fixed_vars)
```

### Configuration

**File:** `configs/sfd.json`

```json
{
  "solver": "sfd",
  "formulation": "sfd",
  "decomposition": "value_layer",
  "use_cuts": true,
  "time_limit": 3600,
  "threads": 8,
  "log_output": false,
  "is_relax": false
}
```

**Parameters:**
- **solver**: "sfd" (identifier)
- **formulation**: "sfd" (identifier)
- **decomposition**: "value_layer" or "value_only"
- **use_cuts**: true/false (enable lazy constraint cuts)
- **time_limit**: seconds (solver time limit)
- **threads**: number of CPLEX threads
- **log_output**: true/false (print solver progress)
- **is_relax**: true/false (relaxed or binary constraints)

## Implementation Details

### Lazy Constraint Callback

The `SFDLazyCallback` class implements lazy constraint generation:

1. **Check Solution:** Extract variables set to 1.0 from current node solution
2. **Generate Cuts:** For each pair of assigned facilities (i,u) and (j,v):
   - Compute delta values (cost improvements from swaps)
   - If swap would improve solution, add valid inequality
3. **Add to Model:** Add cuts as lazy constraints via CPLEX

```python
class SFDLazyCallback(ConstraintCallbackMixin, LazyConstraintCallback):
    def __call__(self):
        # Find assigned pairs
        # Compute delta values
        # Add violated cuts
        # Register with CPLEX
```

### Warm-Start Integration

SFD solver uses the unified solution I/O module:

```python
# Load from QAPLIB format file
warm_start = read_warmstart("solution.sln")

# Add to model as MIP start
ws = model.new_solution()
for (i, u), val in warm_start.items():
    ws.add_var_value(x[i, u], val)
model.add_mip_start(ws)
```

## Testing

### Test Files
- `python/examples/demo_sfd.py` - Usage example
- `test_sfd.py` (to be created) - Unit tests

### Example Test

```bash
cd QAP-Solver
python3 python/examples/demo_sfd.py
```

## Computational Considerations

### Advantages
- Decomposition reduces problem size conceptually
- Lazy cuts strengthen LP relaxation
- Warm-start can significantly reduce solving time

### Challenges
- Subgraph decomposition depends on flow structure
- Number of variables scales with number of subgraphs
- Callback overhead for dense subgraphs

### Recommendations
- Use value_layer decomposition for problems with sparse flows
- Enable cuts for better bounds
- Provide warm-start from simpler heuristic if available
- Adjust time_limit based on problem size

## File Structure

```
QAP-Solver/
├── python/
│   ├── qap/
│   │   ├── modules/
│   │   │   └── sfd/
│   │   │       ├── __init__.py
│   │   │       └── solver.py        ← Main SFD implementation
│   │   └── examples/
│   │       └── demo_sfd.py          ← Usage example
│   └── 
├── configs/
│   └── sfd.json                     ← Configuration
└── 
```

## Version History

- **v1.0** (2024): Initial SFD solver implementation
  - CPLEX Python (DOcplex) backend
  - Lazy constraint callback
  - Warm-start support
  - Value layer decomposition

## References

- Form3 from [QAP_New_formulation](../QAP_New_formulation/FORM3/)
- [DOcplex Documentation](https://ibmdecisionoptimization.github.io/docplex-doc/)
- [CPLEX Callbacks](https://www.ibm.com/docs/en/icos/20.1.0?topic=callbacks-lazy-constraints-user-cuts)
- [QAPLIB Instances](https://www.opt.math.tugraz.at/qaplib/)

## See Also

- [RTL1 SCIP Solver](modules/rtl1_scip/README.md)
- [Unified Solution I/O](SOLUTION_IO_README.md)
- [QAP Problem Class](core/problem.py)
