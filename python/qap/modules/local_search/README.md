# Local Search Module

**Methods**: Tabu Search, Simulated Annealing, Genetic Algorithm  
**Type**: Heuristic (approximate, fast)

## Configuration

See `config_schema.json` for full schema. Key options:

- `solver`: Must be `"local_search"`
- `method`: One of `"tabu_search"`, `"simulated_annealing"`, `"genetic_algorithm"`
- `time_limit`: Time limit in seconds (default: 120)
- `initial_solution`: How to initialize (`"random"` or `"greedy"`)
- `seed`: Random seed for reproducibility

## Example Config

```json
{
  "solver": "local_search",
  "method": "tabu_search",
  "time_limit": 120,
  "initial_solution": "greedy",
  "seed": 42
}
```

## Output Format

```json
{
  "instance": "chr12a",
  "solver": "local_search",
  "method": "tabu_search",
  "objective": 11250,
  "lower_bound": null,
  "gap": null,
  "time": 120.0,
  "assignment": [0, 3, 1, 5, ...]
}
```
