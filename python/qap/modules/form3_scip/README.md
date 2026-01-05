# Form3 SCIP Module

**Solver**: SCIP (C++ backend via file I/O)  
**Formulation**: Binary cubic Form3  
**Method**: Exact (branch-and-cut)

## Configuration

See `config_schema.json` for full schema. Key options:

- `solver`: Must be `"form3_scip"`
- `formulation`: Must be `"form3"`
- `decomposition`: Either `"value_layer"` or `"value_only"` (default: `"value_layer"`)
- `time_limit`: Time limit in seconds (default: 120)
- `threads`: Number of threads (default: 8)

## Implementation

The Python module generates an LP/MPS file, invokes the C++ SCIP binary (`cpp/modules/form3_scip/form3_scip`), and parses the result from a JSON output file.

## Example Config

```json
{
  "solver": "form3_scip",
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
  "solver": "form3_scip",
  "objective": 11156,
  "lower_bound": 11156,
  "gap": 0.0,
  "time": 55.3,
  "assignment": [0, 3, 1, 5, ...]
}
```
