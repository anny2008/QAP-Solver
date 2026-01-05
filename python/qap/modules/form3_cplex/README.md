# Form3 CPLEX Module

**Solver**: CPLEX (via Gurobi or DOCplex)  
**Formulation**: Binary cubic Form3  
**Method**: Exact (branch-and-bound)

## Configuration

See `config_schema.json` for full schema. Key options:

- `solver`: Must be `"form3_cplex"`
- `formulation`: Must be `"form3"`
- `decomposition`: Either `"value_layer"` or `"value_only"` (default: `"value_layer"`)
- `time_limit`: Time limit in seconds (default: 120)
- `threads`: Number of threads (default: 8)

## Example Config

```json
{
  "solver": "form3_cplex",
  "formulation": "form3",
  "decomposition": "value_layer",
  "time_limit": 120,
  "threads": 8
}
```

## Output Format

```json
{
  "instance": "chr12a",
  "solver": "form3_cplex",
  "objective": 11156,
  "lower_bound": 11156,
  "gap": 0.0,
  "time": 60.5,
  "assignment": [0, 3, 1, 5, ...]
}
```
