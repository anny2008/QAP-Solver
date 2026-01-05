# Installation & Setup

## Python Dependencies

### Core Requirements

Install dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Key Packages

| Package | Purpose | Version |
|---------|---------|---------|
| `docplex` | IBM CPLEX Python API | Latest |
| `gurobipy` | Gurobi solver (optional alternative to CPLEX) | Latest |
| `numpy` | Numerical computing | Latest |
| `scipy` | Scientific computing utilities | Latest |
| `jsonschema` | JSON schema validation | Latest |

### Optional Packages

- **Gurobi** — Alternative to IBM CPLEX for form3_cplex module
- **pytest** — For running unit tests
- **jupyter** — For interactive notebooks/analysis

## CPLEX/Gurobi Setup

### Option 1: IBM CPLEX via DOCplex (Recommended)

1. Install DOCplex:
   ```bash
   pip install docplex
   ```

2. Configure CPLEX path (if not auto-detected):
   ```bash
   export CPLEX_STUDIO_DIR=/path/to/CPLEX_Studio/
   ```

### Option 2: Gurobi

1. Install Gurobi:
   ```bash
   pip install gurobipy
   ```

2. Obtain and install Gurobi license from [gurobi.com](https://gurobi.com/)

## C++ Dependencies (SCIP)

The `form3_scip` module requires SCIP to be compiled and available as an executable.

### Installation Steps

1. **Download SCIPOptSuite**:
   ```bash
   cd /tmp
   wget https://github.com/scipopt/scip/archive/refs/tags/v9.2.0.tar.gz
   tar -xzf v9.2.0.tar.gz
   cd scip-9.2.0
   ```

2. **Build and Install**:
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_INSTALL_PREFIX=/opt/scip
   make -j 8
   make install
   ```

3. **Add to PATH**:
   ```bash
   export PATH=/opt/scip/bin:$PATH
   export LD_LIBRARY_PATH=/opt/scip/lib:$LD_LIBRARY_PATH
   ```

4. **Verify Installation**:
   ```bash
   scip --version
   ```

### Pre-compiled Binaries

Alternatively, download pre-compiled binaries from [scipopt.org](https://scipopt.org/):
```bash
# Example for Linux Debian 12
wget https://www.scipopt.org/download/release/SCIPOptSuite-9.2.0-Linux-debian12.sh
bash SCIPOptSuite-9.2.0-Linux-debian12.sh
```

## Project Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd QAP-Solver
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

Test that modules can be imported:
```bash
python -c "import python.qap.modules.form3_cplex; print('OK')"
```

Test CPLEX:
```bash
python -c "from docplex.mp.model import Model; print('CPLEX OK')"
```

Test SCIP (if needed):
```bash
which scip && echo 'SCIP OK'
```

## Configuration Files

Pre-created example configs are in `configs/`:
```
configs/
├── rtl1_cplex.json
├── form3_cplex.json
├── form3_scip.json
└── local_search.json
```

Copy and modify as needed for your use case.

## Data Preparation

Place QAPLIB instances in `data/`:
```bash
mkdir -p data
# Copy .dat files to data/
cp /path/to/instances/*.dat data/
```

See [DATA_FORMAT.md](DATA_FORMAT.md) for instance format details.

## Testing Installation

### Quick Test

```bash
# Test with a small instance (create if needed)
python python/examples/solve.py \
  --module local_search \
  --config configs/local_search.json \
  --instance data/chr12a.dat \
  --output results/test_result.json

# Check result
cat results/test_result.json
```

### Troubleshooting

**CPLEX/Gurobi not found**:
- Verify installation: `python -c "from docplex.mp.model import Model"`
- Check license: `cplex` or `gurobi_cl --license`
- Install missing dependencies: `pip install docplex`

**SCIP not found**:
- Verify PATH: `which scip`
- Verify installation: `scip --version`
- Rebuild if needed: See SCIP installation steps above

**Import errors**:
- Ensure you're using Python 3.7+: `python --version`
- Ensure virtual environment is activated (if using one)
- Reinstall packages: `pip install -r requirements.txt --force-reinstall`

## Development Setup

For developers contributing to the codebase:

1. Install development dependencies:
   ```bash
   pip install pytest pytest-cov jupyter notebook
   ```

2. Clone and install in editable mode:
   ```bash
   git clone <repository-url>
   cd QAP-Solver
   pip install -e .
   ```

3. Run tests:
   ```bash
   pytest python/
   ```

## Environment Variables (Optional)

```bash
# CPLEX path
export CPLEX_STUDIO_DIR=/path/to/CPLEX_Studio/

# SCIP path
export PATH=/opt/scip/bin:$PATH
export LD_LIBRARY_PATH=/opt/scip/lib:$LD_LIBRARY_PATH

# Solver thread count (can be overridden in config)
export OMP_NUM_THREADS=8
```

## Compute Cluster Setup (SLURM)

Example SLURM job script:

```bash
#!/bin/bash
#SBATCH --job-name=qap_solver
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

cd /path/to/QAP-Solver

# Activate environment if using conda/venv
# source activate qap_env

python python/examples/solve.py \
  --module form3_cplex \
  --config configs/form3_cplex.json \
  --instance data/chr12a.dat \
  --output results/chr12a_result.json
```

Submit with:
```bash
sbatch job_script.sh
```
