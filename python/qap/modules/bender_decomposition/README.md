# RLT1 CPLEX Module

**Solver**: CPLEX (via Gurobi or DOCplex)  
**Formulation**: Relaxation-based Tightened Linear (RLT1)  
**Method**: Exact (branch-and-bound)

## Configuration

See `config_schema.json` for full schema. Key options:

- `solver`: Must be `"rlt1_cplex"`
- `formulation`: Must be `"rlt1"`
- `time_limit`: Time limit in seconds (default: 120)
- `threads`: Number of threads (default: 8)

## Example Config

```json
{
  "solver": "rlt1_cplex",
  "formulation": "rlt1",
  "time_limit": 120,
  "threads": 8
}
```

## Output Format

```json
{
  "instance": "chr12a",
  "solver": "rlt1_cplex",
  "objective": 11156,
  "lower_bound": 11156,
  "gap": 0.0,
  "time": 45.2,
  "assignment": [0, 3, 1, 5, ...]
}
```
