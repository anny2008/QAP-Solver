# implement the Bundle Method
from dataclasses import dataclass
from typing import List
# import torch

from abc import ABC, abstractmethod
import time
import numpy as np


@dataclass
class BundleElement:
    g: np.array # subgradient
    f: float #f(lambd)
    f_star: float = None # f*(lambd) = f(lambd) - g^T lambd, the linearization of f at lambd
    

# enum for status of the Oracle
class OracleStatus:
    SUCCESS = 0
    ERROR = 1
    STOP = 2

class Oracle(ABC):
    def compute_value(self):
        if self.has_changed:
            self.solve_subproblem()
        return self.function_value
            
    def compute_subgradient(self):
        if self.has_changed:
            self.solve_subproblem()
        return self.subgradient
    
    def compute_f_star(self):
        if self.has_changed:
            self.solve_subproblem()
        return self.f_star
    
    def set_lambda(self, lambd: np.array):
        self.lambd = lambd
        self.has_changed = True
        
    @abstractmethod
    def solve_subproblem(self):
        pass
    
    lambd_size: int
    lambd_ub: np.array = None
    lambd_lb: np.array = None
    lambd: np.array = None
    has_changed: bool = False
    status: OracleStatus = OracleStatus.SUCCESS

class BundleMethodConfig:
    def __init__(self, 
                 t: float = 1.0,
                 max_iters: int = 100, 
                 max_time: float = 60.0,
                 max_bundle_size: int = 10,
                 max_null_steps: int = 20,
                 beta: float = 0.0025, #acceptance parameter for the serious step
                 epsilon: float = 1e-5, #tolerance for stopping criterion
                 mu: float = 1.0,  
                 tol: float = 1e-5, 
                 log_interval: int = 10,
                 verbose: int = 2,
                 device: str = "cpu",
                 Dt_point_strategy: str = "qp",
                 master_problem_solver: str = "cvxpy"
                 ):
        self.max_iters = max_iters
        self.tol = tol
        self.max_time = max_time
        self.max_bundle_size = max_bundle_size
        self.beta = beta
        self.epsilon = epsilon
        self.mu = mu
        self.log_interval = log_interval
        self.verbose = verbose
        # self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        self.Dt_point_strategy = Dt_point_strategy
        self.master_problem_solver = master_problem_solver
        self.max_null_steps = max_null_steps

class BundleMethod:
    # minimize the oracle convex function f(lambd)
    def __init__(self, config: BundleMethodConfig, oracle: Oracle):
        self.config = config
        self.oracle = oracle
        self.bundle = []
        self.lambd_center = np.zeros(oracle.lambd_size)
        self.f_center = 0.0
        self.lambd_size = oracle.lambd_size
        self.f_best = float('inf')
        self.current_precision = float('inf')
        self.f_predicted = 0.0
        self.has_changed = True
        self.t = 1

    # Store the bundle of subgradients and function values
    bundle: list[BundleElement]
    f_center: float
    lambd_center: np.array
    f_candidate: float
    lambd_candidate: np.array
    DSTS_candidate: float
    f_predicted: float
    f_best: float
    
    
    def initialize(self):
        pass
    
    def solve_master_problem(self):
        # Solve the master problem to get the trial point
        # min_d f_bundle(x_bar + d) + D_t(d)
        # f_bundle(x) = max_{i=1,...,k} {g_i^T x - f*_i}
        # D_t(d) = (1/2t) ||d||^2
        # d = lambd - center_lambd
        # lambd = center_lambd + d
        
        if self.config.Dt_point_strategy == "qp":
            # D_t(lambd) = (1/2t) ||lambd - center_lambd||^2
            if self.config.master_problem_solver == "cvxpy":
                # Use cvxpy to solve the master problem
                import cvxpy as cp
                lambd = cp.Variable(self.lambd_size)
                lambd_center = cp.Constant(self.lambd_center)
                cuts = [
                    e.g @ lambd - e.f_star
                    for e in self.bundle
                ]
                f_bundle = cp.max(cp.hstack(cuts))
                D_t = (1 / (2*self.t)) * cp.sum_squares(lambd - lambd_center)
                # D_t = (1 / (2 * self.iteration)) * cp.sum_squares(lambd - lambd_center)
                objective = cp.Minimize(f_bundle + D_t)
                
                constraints = []
                if self.oracle.lambd_ub is not None:
                    for i in range(self.lambd_size):
                        if self.oracle.lambd_ub[i] < np.inf:
                            constraints.append(lambd[i] <= self.oracle.lambd_ub[i])
                if self.oracle.lambd_lb is not None:
                    for i in range(self.lambd_size):
                        if self.oracle.lambd_lb[i] > -np.inf:
                            constraints.append(lambd[i] >= self.oracle.lambd_lb[i])
                prob = cp.Problem(objective, constraints=constraints)
                prob.solve()
                # print the problem to debug
                # set DSTS to D_t
                self.DSTS_candidate = D_t.value
                # return the trial point and the predicted value
                self.lambd_candidate = lambd.value
                self.f_predicted = f_bundle.value
                
                print(f"    BM: Master problem solved, predicted value = {self.f_predicted:.2f}, DSTS = {self.DSTS_candidate:.2f}")
            else:
                raise NotImplementedError("Master problem solver not implemented.")
            
        
        
    def add_to_bundle(self, element: BundleElement):
        self.bundle.append(element)

    def is_optimal(self):
        if len(self.bundle) == 0:
            return False
        
        # precision = bundle_value - real_value
        self.current_precision = self.f_center - self.f_predicted
        return abs(self.current_precision) < self.config.epsilon

    def solve(self):
        # Initialize the center point
        start_time = time.time()
        self.initialize()
        self.iteration = 0
        # self.count_NS = self.config.max_null_steps + 1 # to force the first step to be a serious step
        self.count_NS = 0
        self.has_changed = True
        while True:
            # if verbose > 0, and the iteration is a multiple of log_interval, print the current status
            if self.config.verbose > 0 and len(self.bundle) % self.config.log_interval == 0:
                print(f"BM: Iteration {self.iteration}, current precision = {self.current_precision:.2e}, f_center = {self.f_center:.2f}, f_best = {self.f_best:.2f}, bundle size = {len(self.bundle)}, time elapsed = {time.time() - start_time:.2f}s")
            # call the oracle to compute subgradient and function value at the current center
            if self.has_changed:
                self.oracle.set_lambda(self.lambd_center)
                self.f_center = self.oracle.compute_value()
                self.g_center = self.oracle.compute_subgradient()
                self.f_star = self.oracle.compute_f_star()
            status = self.oracle.status
                        
            if status == OracleStatus.ERROR:
                print("BM: Oracle returned an error. Stopping.")
                break
            elif status == OracleStatus.STOP:
                print("BM: Oracle requested to stop. Stopping.")
                break
            
            if self.is_optimal():
                print("BM: Optimality condition met. Stopping.")
                break
            if time.time() - start_time > self.config.max_time:
                print("BM: Time limit exceeded. Stopping.")
                break
            if self.iteration >= self.config.max_iters:
                print("BM: Maximum iterations reached. Stopping.")
                break
            
            self.iteration += 1
            self.bundle.append(BundleElement(g=self.g_center, f=self.f_center, f_star=self.f_star))
            
            
            # compute the trial point from the master problem
            self.solve_master_problem()
            
            self.oracle.set_lambda(self.lambd_candidate)
            self.f_candidate = self.oracle.compute_value()
            #  if the trial point is a serious step, update the center
            delta_actual = self.f_center - self.f_candidate  # actual improvement <= 0
            delta_predict = self.f_center - self.f_predicted  # predicted improvement <= 0
            print(f"    BM: delta_actual = {delta_actual:.2e}, delta_predict = {delta_predict:.2e}")
            rho = delta_actual / (delta_predict + 1e-8)  # add a small value to avoid division by zero
            print(f"    BM: rho = {rho:.2e}")
            SS_step_condition = rho >= self.config.beta
            # if abs(delta_F) >= self.config.beta * abs(delta_v): #actual improvement is greater than beta times predicted improvement, then it's a serious step
            if self.count_NS > self.config.max_null_steps or SS_step_condition:
                self.count_NS = 0
                # serious step
                self.lambd_center = self.lambd_candidate
                self.has_changed = True
                if self.f_candidate < self.f_best:
                    self.f_best = self.f_candidate
                if SS_step_condition:
                    self.t *= 1.5
                    if self.t > 1e6:
                        self.t = 1e6  # avoid t becoming too large
                    print(f"    BM: Serious step taken.")
                else:
                    print(f"    BM: Serious step taken due to max null steps reached.")
            else:
                self.count_NS += 1
                self.t *= 0.66  # decrease the penalty parameter for the next iteration
                if self.t < 1e-6:
                    self.t = 1e-6  # avoid t becoming too small
                # null step, add the new subgradient to the bundle
                print(f"    BM: Null step taken.")
        
        print("========================================================================")
        
            
            
        
    
        
