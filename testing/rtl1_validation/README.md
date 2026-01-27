# RLT1 CPLEX Solver - Testing & Validation Suite

This directory contains comprehensive testing and debugging scripts for the RLT1 CPLEX solver implementation.

## Contents

### Core Testing Scripts
- **test_rlt1.py** - Single instance test on chr12a with detailed verification
- **test_rlt1_configs.py** - Configuration comparison (binary, relaxed, with/without preprocessing)
- **batch_test_rlt1_small.py** - Batch test on 12×12 instances (10s time limit)
- **batch_test_rlt1_small_60s.py** - Extended test on 12×12 instances (60s time limit)

### Warm-Start Testing
- **test_rlt1_known_warmstart.py** - Test with known optimal solutions as warm-starts
- **test_rlt1_warmstart.py** - Detailed warm-start comparison
- **test_warmstart_implementation.py** - Warm-start API verification
- **test_warmstart_logging.py** - CPLEX logging with warm-start enabled
- **test_warmstart_multiple.py** - Warm-start testing on multiple instances

### Debug & Analysis Scripts
- **debug_rlt1.py** - Basic solver debugging
- **debug_assignment_quality.py** - Verify solution quality and assignment extraction
- **debug_batch.py** - Batch processing debugging
- **debug_bounds.py** - Lower bound calculation verification
- **debug_warmstart_api.py** - Warm-start API internals

### Analysis Reports
- **TESTING_SUMMARY.md** - Complete batch test results and feature validation
- **WARMSTART_ANALYSIS.md** - Detailed warm-start mechanism analysis

## Quick Start

### Run Tests
```bash
cd /home/local.isima.fr/antran/UFF/QAP-Solver

# Single instance test
python testing/rlt1_validation/test_rlt1.py

# Batch test on chr12 instances
python testing/rlt1_validation/batch_test_rlt1_small.py

# Configuration comparison
python testing/rlt1_validation/test_rlt1_configs.py

# Warm-start testing
python testing/rlt1_validation/test_warmstart_multiple.py
```

## Test Coverage

### Instances Tested
- **chr12a, chr12b, chr12c** - n=12, known optimal
- **nug12** - n=12, known optimal
- **tai12a** - n=12, known optimal

### Features Verified
✅ Binary variable mode (integer assignments)
✅ Relaxed 0-1 continuous mode
✅ Configurable time limits (10s, 60s)
✅ Multi-threaded solving
✅ Symmetry preprocessing
✅ Fixed variables support
✅ Warm-start API (docplex integration)
✅ Result extraction and verification
✅ JSON output format

### Performance Summary
- **Average solve time**: 2.5-2.7 seconds per 12×12 instance
- **Fast convergence**: Same solutions at 10s and 60s limits
- **Solution quality**: 239-516% gap to RLT1 lower bound (normal for relaxation)

## Key Findings

### Warm-Start Behavior
- ✅ Warm-start successfully registered with CPLEX
- ✅ Provides initial upper bounds
- ⚠️ Limited improvement on extracted assignment for small instances
  - Reason: CPLEX heuristics find solutions as good or better than warm-start
  - Expected: Better results on larger instances (n>30)

### RLT1 Linearization
- ✅ Proper linearization with linking constraints
- ✅ Consistent objective calculation
- ✅ Correct handling of binary/continuous variables

## Notes for Future Work

1. **Test Organization**: These scripts should be reorganized into pytest-compatible structure
2. **Results Archive**: Results directory contains JSON output from test runs
3. **Warm-Start Enhancement**: Consider using heuristic solutions instead of optimal for better guidance
4. **Larger Instances**: Test on n>30 instances to verify scalability

## Environment Requirements

```bash
conda activate LP
# CPLEX must be installed via conda
# numpy 2.0+ compatibility: np.float_ = np.float64 (applied in solver)
```

## References

- Core solver: `python/qap/modules/rlt1_cplex/solver.py`
- Demo usage: `python/examples/demo_rlt1.py`
- Documentation: Root README and docs/ folder
