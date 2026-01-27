# QAP-Solver Project Scaffolding - Summary

## What Has Been Created

### 1. **Folder Structure**
- `python/qap/` — Main Python package
  - `core/` — Shared utilities (problem.py, io.py, solution.py to be implemented)
  - `modules/` — Solver modules (4 sub-packages)
    - `rlt1_cplex/` — RLT1 formulation with CPLEX
    - `form3_cplex/` — Form3 formulation with CPLEX
    - `form3_scip/` — Form3 formulation with SCIP backend
    - `local_search/` — Heuristic methods (Tabu, SA, GA)
- `python/examples/` — Entry points (solve.py, benchmark.py to be implemented)
- `cpp/modules/form3_scip/` — C++ SCIP backend
- `configs/` — User configuration files (JSON)
- `data/` — QAP instances (QAPLIB format)
- `results/` — Solver outputs (JSON results)

### 2. **Configuration System**
Each of the 4 solver modules has:
- **`config_schema.json`**: JSON Schema for validating configurations
- **`example_config.json`**: Sample configuration with reasonable defaults
- **Module location**: `python/qap/modules/<module_name>/`

**Example modules:**
| Module | Solver | Formulation | Type |
|--------|--------|-------------|------|
| `rlt1_cplex` | CPLEX | RLT1 | Exact |
| `form3_cplex` | CPLEX | Form3 (binary cubic) | Exact |
| `form3_scip` | SCIP (C++) | Form3 (binary cubic) | Exact |
| `local_search` | Native Python | Tabu/SA/GA | Heuristic |

### 3. **User Config Files**
Pre-created example configs in `configs/`:
- `rlt1_cplex.json` — 120s time limit, 8 threads
- `form3_cplex.json` — value_layer decomposition, 120s time limit, 8 threads
- `form3_scip.json` — value_layer decomposition, 120s time limit, 8 threads
- `local_search.json` — Tabu search, greedy init, 120s time limit, seed 42

### 4. **Python Package Structure**
- `python/qap/__init__.py` — Package marker
- `python/qap/core/__init__.py` — Core utilities package
- `python/qap/modules/<module>/__init__.py` — 4 modules with docstrings

### 5. **Documentation**
- **Main README.md** — Complete project documentation with:
  - Architecture overview
  - Module descriptions with links to module READMEs
  - Configuration schema examples
  - Result format specification
  - Usage instructions with CLI examples
  - Data format documentation
  - Dependencies and setup instructions
  - Development notes (no data marshaling, config-driven design)
  - Status and roadmap

- **Module READMEs** — One per solver module:
  - `python/qap/modules/rlt1_cplex/README.md`
  - `python/qap/modules/form3_cplex/README.md`
  - `python/qap/modules/form3_scip/README.md`
  - `python/qap/modules/local_search/README.md`

  Each includes solver details, config schema excerpt, example config, and output format.

---

## Next Steps

### Short-term (Implementation)
1. **Core utilities**: Implement `python/qap/core/`:
   - `problem.py` — QAP problem class (n, F, D matrices)
   - `io.py` — QAPLIB instance loader, result JSON writer
   - `solution.py` — Solution class (assignment permutation, objective)

2. **Solver modules**: Implement `solver.py` in each module:
   - Read config JSON and instance data
   - Call CPLEX/SCIP/native solver
   - Return standardized result JSON

3. **Main dispatcher**: `python/examples/solve.py`
   - CLI argument parser (--module, --config, --instance, --output)
   - Config validation against module schema
   - Route to correct solver module
   - Write results to JSON

4. **C++ backend**: `cpp/modules/form3_scip/`
   - Implement Form3 formulation in C++
   - SCIP solver integration
   - Read LP/MPS file, output JSON result

### Medium-term (Testing & Validation)
- Add unit tests for each module
- Create benchmarking suite (benchmark.py)
- Validate on QAPLIB instances (chr12a, etc.)
- Compare solver outputs and performance

### Long-term (Enhancement)
- Additional solver methods (decomposition, cutting planes)
- Warm-start strategies
- Distributed solving
- REST API wrapper

---

## Design Principles

1. **Modularity**: Each solver is independent; no cross-module dependencies.
2. **File-Based I/O**: Python and C++ communicate only via files (no marshaling).
3. **Config-Driven**: All parameters specified in JSON; no hardcoded values.
4. **Unified Result Format**: All solvers output the same JSON structure.
5. **QAPLIB Standard**: Work directly with standard instance format.

---

## File Count Summary

| Category | Count |
|----------|-------|
| Config schema files | 4 |
| Example config files | 4 |
| User config files (in configs/) | 4 |
| Python __init__.py files | 6 |
| Module README files | 4 |
| Main README | 1 |
| **Total** | **23** |

---

## Configuration Validation

Each module's `config_schema.json` defines:
- Required properties (solver, formulation/method)
- Optional properties (time_limit, threads, etc.)
- Valid enum values
- Type constraints

Configs can be validated using `jsonschema` in Python:
```python
import json
import jsonschema

with open('configs/form3_cplex.json') as f:
    config = json.load(f)
with open('python/qap/modules/form3_cplex/config_schema.json') as f:
    schema = json.load(f)

jsonschema.validate(config, schema)
```

---

## Usage Example

Once implementation is complete:

```bash
# Solve chr12a.dat with Form3 CPLEX
python python/examples/solve.py \
  --module form3_cplex \
  --config configs/form3_cplex.json \
  --instance data/chr12a.dat \
  --output results/chr12a_form3_cplex.json

# Result in results/chr12a_form3_cplex.json:
# {
#   "instance": "chr12a",
#   "solver": "form3_cplex",
#   "objective": 11156,
#   "lower_bound": 11156,
#   "gap": 0.0,
#   "time": 60.5,
#   "assignment": [0, 3, 1, 5, ...]
# }
```
