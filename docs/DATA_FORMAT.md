# Data Format

## Instance Format (QAPLIB)

Instances follow the standard QAPLIB format with flow and distance matrices.

### File Structure

```
n
f(0,0) f(0,1) ... f(0,n-1)
f(1,0) f(1,1) ... f(1,n-1)
...
f(n-1,0) ... f(n-1,n-1)
d(0,0) d(0,1) ... d(0,n-1)
d(1,0) d(1,1) ... d(1,n-1)
...
d(n-1,0) ... d(n-1,n-1)
```

### Components

- **n**: Problem size (number of facilities and locations)
- **F**: Flow matrix (n × n) — flow between facilities i and j
- **D**: Distance matrix (n × n) — distance between locations u and v

### Objective

Minimize: $$\sum_{i=0}^{n-1} \sum_{j=0}^{n-1} f(i,j) \cdot d(\pi[i], \pi[j])$$

Where π is a permutation of {0, 1, ..., n-1}, such that π[i] = u means facility i is assigned to location u.

### Example

For n=3:
```
3
0 1 2
3 0 4
5 6 0
0 1 2
3 0 4
5 6 0
```

This defines:
- Flow matrix F (3×3)
- Distance matrix D (3×3)

## Result Format

All solvers output results in a consistent JSON format.

### File Structure

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

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `instance` | string | Instance name (without path or extension) |
| `solver` | string | Solver/method identifier (rlt1_cplex, form3_cplex, form3_scip, local_search) |
| `objective` | number \| null | Best objective value found (null if no solution found) |
| `lower_bound` | number \| null | Lower bound on optimal solution (null for heuristics) |
| `gap` | number \| null | Optimality gap: (objective - lower_bound) / lower_bound × 100% (null if not applicable) |
| `time` | number | Wall-clock execution time in seconds |
| `assignment` | array | Permutation π where assignment[i] = j means facility i → location j |

### Examples

**Exact Solver (optimal solution found)**:
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

**Exact Solver (time limit, no optimal)**:
```json
{
  "instance": "kra30a",
  "solver": "form3_cplex",
  "objective": 88900,
  "lower_bound": 87250,
  "gap": 1.89,
  "time": 120.0,
  "assignment": [0, 5, 10, 15, ...]
}
```

**Heuristic Solver (no bound)**:
```json
{
  "instance": "chr12a",
  "solver": "local_search",
  "objective": 11300,
  "lower_bound": null,
  "gap": null,
  "time": 5.2,
  "assignment": [0, 2, 4, 1, 3, 5, 7, 6, 9, 8, 11, 10]
}
```

**No Solution Found**:
```json
{
  "instance": "kra30a",
  "solver": "form3_cplex",
  "objective": null,
  "lower_bound": 85000,
  "gap": null,
  "time": 120.0,
  "assignment": null
}
```

## File Organization

Place instances in the `data/` directory:

```
data/
├── chr12a.dat
├── chr12b.dat
├── kra30a.dat
└── ...
```

Place results in the `results/` directory:

```
results/
├── chr12a_rlt1_cplex.json
├── chr12a_form3_cplex.json
├── chr12a_form3_scip.json
├── chr12a_local_search.json
├── kra30a_form3_cplex.json
└── ...
```

## Known Optimal Solutions

If known optimal solutions are available, they can be stored in a separate file for validation:

```
data/
├── chr12a.dat
├── chr12a.opt      # Known optimal value (e.g., "11156")
├── chr12b.dat
├── chr12b.opt
└── ...
```

This allows for automatic validation and gap calculation even for unsolved instances.
