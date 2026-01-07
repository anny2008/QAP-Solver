# Configuration

## Overview

Each solver module has its own configuration schema. Configurations are JSON files that specify solver parameters, time limits, formulation options, and other runtime settings.

## Configuration Files

Each module has:
- **`config_schema.json`** — JSON Schema for validating configurations
- **`example_config.json`** — Sample configuration with sensible defaults
- **Module location** — `python/qap/modules/<module_name>/`

User configurations should be placed in the `configs/` directory:
```
configs/
├── rtl1_cplex.json
├── form3_cplex.json
├── form3_scip.json
└── local_search.json
```

## Module Schemas

### RTL1 SCIP (C++)

The C++ RTL1 with SCIP supports a `relaxation` option to choose between the default SCIP behavior and a custom Volume-based relaxation handler:

```json
{
  "solver": "rtl1_scip",
  "formulation": "rtl1",
  "instance": "data/chr12a.dat",
  "time_limit": 120,
  "threads": 8,
  "log_output": true,
  "relaxation": "volume"  // or "default"
  "relaxation_info": true  // print extra info from relaxation handler
}
```

- relaxation: "volume" uses the custom Volume relaxation; "default" disables it and relies on SCIP's default relaxation.
- relaxation_info: if true, prints relaxation value, number of fixed x/y variables, and violation summary at each call.

### RTL1 CPLEX

**Schema** (`python/qap/modules/rtl1_cplex/config_schema.json`):
```json
{
  "title": "RTL1 CPLEX Configuration",
  "type": "object",
  "properties": {
    "solver": {
      "type": "string",
      "enum": ["rtl1_cplex"],
      "description": "Solver to use"
    },
    "formulation": {
      "type": "string",
      "enum": ["rtl1"],
      "description": "Formulation type"
    },
    "time_limit": {
      "type": "number",
      "minimum": 0,
      "description": "Time limit in seconds"
    },
    "threads": {
      "type": "integer",
      "minimum": 1,
      "description": "Number of threads"
    }
  },
  "required": ["solver", "formulation"]
}
```

**Example** (`python/qap/modules/rtl1_cplex/example_config.json`):
```json
{
  "solver": "rtl1_cplex",
  "formulation": "rtl1",
  "time_limit": 120,
  "threads": 8
}
```

### Form3 CPLEX

**Schema** (`python/qap/modules/form3_cplex/config_schema.json`):
```json
{
  "title": "Form3 CPLEX Configuration",
  "type": "object",
  "properties": {
    "solver": {
      "type": "string",
      "enum": ["form3_cplex"],
      "description": "Solver to use"
    },
    "formulation": {
      "type": "string",
      "enum": ["form3"],
      "description": "Formulation type"
    },
    "decomposition": {
      "type": "string",
      "enum": ["value_layer", "value_only"],
      "description": "Decomposition strategy"
    },
    "time_limit": {
      "type": "number",
      "minimum": 0,
      "description": "Time limit in seconds"
    },
    "threads": {
      "type": "integer",
      "minimum": 1,
      "description": "Number of threads"
    }
  },
  "required": ["solver", "formulation"]
}
```

**Example** (`python/qap/modules/form3_cplex/example_config.json`):
```json
{
  "solver": "form3_cplex",
  "formulation": "form3",
  "decomposition": "value_layer",
  "time_limit": 120,
  "threads": 8
}
```

### Form3 SCIP

**Schema** (`python/qap/modules/form3_scip/config_schema.json`):
```json
{
  "title": "Form3 SCIP Configuration",
  "type": "object",
  "properties": {
    "solver": {
      "type": "string",
      "enum": ["form3_scip"],
      "description": "Solver to use"
    },
    "formulation": {
      "type": "string",
      "enum": ["form3"],
      "description": "Formulation type"
    },
    "decomposition": {
      "type": "string",
      "enum": ["value_layer", "value_only"],
      "description": "Decomposition strategy"
    },
    "time_limit": {
      "type": "number",
      "minimum": 0,
      "description": "Time limit in seconds"
    },
    "threads": {
      "type": "integer",
      "minimum": 1,
      "description": "Number of threads"
    }
  },
  "required": ["solver", "formulation"]
}
```

**Example** (`python/qap/modules/form3_scip/example_config.json`):
```json
{
  "solver": "form3_scip",
  "formulation": "form3",
  "decomposition": "value_layer",
  "time_limit": 120,
  "threads": 8
}
```

### Local Search

**Schema** (`python/qap/modules/local_search/config_schema.json`):
```json
{
  "title": "Local Search Configuration",
  "type": "object",
  "properties": {
    "solver": {
      "type": "string",
      "enum": ["local_search"],
      "description": "Solver/method to use"
    },
    "method": {
      "type": "string",
      "enum": ["tabu_search", "simulated_annealing", "genetic_algorithm"],
      "description": "Heuristic method"
    },
    "time_limit": {
      "type": "number",
      "minimum": 0,
      "description": "Time limit in seconds"
    },
    "initial_solution": {
      "type": "string",
      "enum": ["random", "greedy"],
      "description": "How to initialize the solution"
    },
    "seed": {
      "type": "integer",
      "description": "Random seed for reproducibility"
    }
  },
  "required": ["solver", "method"]
}
```

**Example** (`python/qap/modules/local_search/example_config.json`):
```json
{
  "solver": "local_search",
  "method": "tabu_search",
  "time_limit": 120,
  "initial_solution": "greedy",
  "seed": 42
}
```

## Validation

Configurations are validated using JSON Schema. In Python, validation can be done using the `jsonschema` library:

```python
import json
import jsonschema

# Load config and schema
with open('configs/form3_cplex.json') as f:
    config = json.load(f)
with open('python/qap/modules/form3_cplex/config_schema.json') as f:
    schema = json.load(f)

# Validate
jsonschema.validate(config, schema)
```

## Creating a Custom Config

1. Copy an example config:
   ```bash
   cp python/qap/modules/form3_cplex/example_config.json configs/my_form3_config.json
   ```

2. Edit to your preferences:
   ```json
   {
     "solver": "form3_cplex",
     "formulation": "form3",
     "decomposition": "value_only",
     "time_limit": 300,
     "threads": 16
   }
   ```

3. Use with the solver:
   ```bash
   python python/examples/solve.py \
     --module form3_cplex \
     --config configs/my_form3_config.json \
     --instance data/instance.dat
   ```

## Pre-created Configs

The following example configs are already in `configs/`:
- `rtl1_cplex.json` — RTL1 with 120s time limit, 8 threads
- `form3_cplex.json` — Form3 value_layer with 120s time limit, 8 threads
- `form3_scip.json` — Form3 value_layer SCIP with 120s time limit, 8 threads
- `local_search.json` — Tabu search with 120s time limit, greedy init, seed 42
