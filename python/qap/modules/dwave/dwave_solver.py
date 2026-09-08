import dimod
import numpy as np
from dwave.system import LeapHybridCQMSampler

filename = '/home/local.isima.fr/antran/UFF/QAP_New_formulation/data/QAPLIB/tai30a.dat'
def load_qaplib(filename):
    """Parses standard QAPLIB data format."""
    with open(filename, 'r') as f:
        content = f.read().split()
    
    # First number is the size (N)
    n = int(content[0])
    
    # Extract Flow (F) and Distance (D) matrices
    # QAPLIB stores them as flat lists after the size
    data = [int(x) for x in content[1:]]
    
    flow = np.array(data[:n*n]).reshape(n, n)
    dist = np.array(data[n*n:2*n*n]).reshape(n, n)
    
    return n, flow, dist

def build_cqm(n, flow, dist):
    """Constructs the Constrained Quadratic Model for QAP."""
    cqm = dimod.ConstrainedQuadraticModel()

    # 1. Create Binary Variables x_{i,j} 
    # x[i][j] = 1 if Facility i is at Location j
    x = [[dimod.Binary(f'x_{i}_{j}') for j in range(n)] for i in range(n)]

    # 2. Set Objective Function: min sum(Flow * Distance)
    # Cost = sum_{i,j,k,l} F_{ik} * D_{jl} * x_{ij} * x_{kl}
    objective = dimod.QuadraticModel()

    # Add all variables to the objective first
    for i in range(n):
        for j in range(n):
            varname = f'x_{i}_{j}'
            objective.add_variable('BINARY', varname)

    print("Building objective function...")
    for i in range(n):
        for k in range(n):
            if flow[i][k] == 0:
                continue  # Optimization: skip zero flows
            for j in range(n):
                for l in range(n):
                    coeff = flow[i][k] * dist[j][l]
                    if coeff != 0:
                        objective.add_quadratic(f'x_{i}_{j}', f'x_{k}_{l}', coeff)

    cqm.set_objective(objective)

    # 3. Add Constraints
    print("Adding constraints...")
    
    # Constraint A: Each Facility i must be assigned to exactly 1 Location
    for i in range(n):
        cqm.add_constraint(sum(x[i]) == 1, label=f'Facility_{i}_assigned')

    # Constraint B: Each Location j must host exactly 1 Facility
    for j in range(n):
        cqm.add_constraint(sum(x[i][j] for i in range(n)) == 1, label=f'Location_{j}_filled')

    return cqm

def solve_qap(filename='tai30a.dat'):
    # Load Data
    print(f"Loading {filename}...")
    try:
        n, flow, dist = load_qaplib(filename)
    except FileNotFoundError:
        print("Error: Please download 'tai30a.dat' from QAPLIB first.")
        return

    # Build Model
    cqm = build_cqm(n, flow, dist)
    print(f"Model built: {n*n} variables.")

    # Submit to D-Wave Hybrid Solver
    # Note: The paper used a time_limit of 150s for tai30a
    sampler = dimod.ExactDQMSolver()
    
    print("Submitting to LeapHybridCQMSampler (this may take 2-3 minutes)...")
    sampleset = sampler.sample_dqm(cqm, time_limit=150, label="QAP Reproduction tai30a")

    # Process Results
    feasible_samples = sampleset.filter(lambda row: row.is_feasible)
    
    if len(feasible_samples) == 0:
        print("No feasible solution found.")
    else:
        best = feasible_samples.first
        print("\n=== RESULTS ===")
        print(f"Best Energy (Cost): {best.energy}")
        print(f"Best Known Solution (Target): 181812") 
        print(f"Optimality Gap: {100 * (best.energy - 181812) / 181812:.2f}%")

if __name__ == "__main__":
    solve_qap(filename=filename)