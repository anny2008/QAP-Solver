# implement the Oracle for the Bundle Method
from .bundle import Oracle, BundleElement, OracleStatus
# import torch
import numpy as np

class RLT1Oracle(Oracle):
    # solving max f(lambd) = min sum_iujv f_uv*d_ij * y_iujv
    #                           + sum_i lambd_i * (1 - sum_u x_iu)
    #                           + sum_u lambd_u * (1 - sum_i x_iu)
    #                           + sum_iuj lambd_iuj * (sum_v x_iu - y_iujv)
    #                           + sum_iuv lambd_iuv * (sum_j x_iu - y_iujv)
    
    def __init__(self, qap_instance):
        self.lambd_size = qap_instance.n*2 + 2*qap_instance.n**3
        self.qap_instance = qap_instance

    def solve_subproblem(self):
        n = self.qap_instance.n
        lambd = self.lambd.copy()
        lambd_i = lambd[:self.qap_instance.n]
        lambd_u = lambd[self.qap_instance.n:2*self.qap_instance.n]
        lambd_iuj = lambd[2*n:2*n + n**3].reshape((n, n, n))
        lambd_iuv = lambd[2*n + n**3:].reshape((n, n, n))

        # solving the inner problem 
        #  min sum_iujv c_iujv * y_iujv + c_iu * x_iu + sum_i lambd_i + sum_u lambd_u
        #  c_iujv = f_uv*d_ij + f_vu*d_ji - lambd_iuj - lambd_jvi - lambd_iuv - lambd_jvu
        #  c_iu = -lambd_i - lambd_u
        lcost = np.sum(lambd_i) + np.sum(lambd_u) #lcost = sum_i lambd_i + sum_u lambd_u + sum_iujv c_iujv * y_iujv + sum_iu c_iu * x_iu
        pcost = 0
        vio_i = np.ones(n) # (1 - sum_u x_iu)
        vio_u = np.ones(n) # (1 - sum_i x_iu)
        vio_iuj = np.zeros((n, n, n)) # (sum_v x_iu - y_iujv)
        vio_iuv = np.zeros((n, n, n)) # (sum_j x_iu - y_iujv)
        n = self.qap_instance.n
        x_star = np.zeros((n, n))
        for i in range(n):
            for u in range(n):
                c_iu = -lambd_i[i] - lambd_u[u]
                c_iu += np.sum(lambd_iuj[i, u, :]) - lambd_iuj[i, u, i]
                c_iu += np.sum(lambd_iuv[i, u, :]) - lambd_iuv[i, u, u]
                # solve the inner problem for x_iu
                #  min c_iu * x_iu
                #       x_iu >= 0
                # The optimal solution is x_iu = 1 if c_iu < 0, else x_iu = 0
                if c_iu < 0:
                    x_star[i, u] = 1
                    lcost += c_iu * x_star[i, u]
                    vio_i[i] -= x_star[i, u]
                    vio_u[u] -= x_star[i, u]
        
        for i in range(n):
            for u in range(n):
                for j in range(i + 1, n):
                    for v in range(n):
                        if u == v: continue
                        c_p_iujv = self.qap_instance.F[u, v] * self.qap_instance.D[i, j]\
                                + self.qap_instance.F[v, u] * self.qap_instance.D[j, i]
                        c_iujv = c_p_iujv \
                            - lambd_iuj[i, u, j] - lambd_iuj[j, v, i] \
                            - lambd_iuv[i, u, v] - lambd_iuv[j, v, u]
                        # solve the inner problem for y_iujv
                        #  min c_iujv * y_iujv
                        #  s.t. y_iujv >= 0
                        # The optimal solution is y_iujv = 1 if c_iujv < 0, else y_iujv = 0
                        if c_iujv < 0:
                            y_iujv = 1
                            pcost += c_p_iujv
                            vio_iuj[i, u, j] -= y_iujv
                            vio_iuv[i, u, v] -= y_iujv
                            vio_iuj[j, v, i] -= y_iujv
                            vio_iuv[j, v, u] -= y_iujv
                            lcost += c_iujv * y_iujv

        for i in range(n):
            for u in range(n):
                for j in range(n):
                    if i == j: continue
                    vio_iuj[i, u, j] += x_star[i, u]
                for v in range(n):
                    if u == v: continue
                    vio_iuv[i, u, v] += x_star[i, u]
        
        lcost_verify = pcost + np.sum(vio_i * lambd_i) + np.sum(vio_u * lambd_u) + np.sum(vio_iuj * lambd_iuj) + np.sum(vio_iuv * lambd_iuv)
        print(f"    Oracle: lcost = {lcost:.2f}, pcost = {pcost:.2f}, lcost_verify = {lcost_verify:.2f}")
        
        self.subgradient = -np.concatenate([vio_i, vio_u, vio_iuj.flatten(), vio_iuv.flatten()])
        self.function_value = -lcost
        self.f_star = pcost
        # check if self.f_star+ self.function_value is equal to self.subgradient @ self.lambd
        self.has_changed = False

class TestOracle(Oracle):
    # solving min f(lambd) = [x^2 + y^2] + lambd * (x + y - 1)
    def __init__(self):
        self.lambd_size = 1
        # solve problem by cvxpy
        import cvxpy as cp
        x = cp.Variable()
        y = cp.Variable()
        constraints = [x + y - 1 == 0, x >= 0, y >= 0]
        problem = cp.Problem(cp.Minimize(x**2 + y**2), constraints)
        problem.solve()
        print(f"Oracle2: optimal solution = ({x.value}, {y.value}), optimal value = {problem.value}")

    def compute_subgradient(self, lambd):
        # subgradient = x* + y* - 1, where (x*, y*) is the optimal solution of
        # the inner problem 
        # optimize: min [x^2 + y^2] + lambd * (x + y - 1)
        # subject to: x >= 0, y >= 0
        # The optimal solution can be found analytically:
        x_star = max(0, lambd[0] / 2)
        y_star = max(0, lambd[0] / 2)
        print(f"Oracle: lambd = {lambd}, x* = {x_star}, y* = {y_star}")
        subgradient = np.array([x_star + y_star - 1])
        function_value = (x_star**2 + y_star**2) + lambd[0] * (x_star + y_star - 1)
        f_star = subgradient @ lambd - function_value
        
        bundle_element = BundleElement(g=subgradient, f=function_value, f_star=f_star)
        return bundle_element, OracleStatus.SUCCESS

class TestOracle2(Oracle):
    # solving min f(lambd) = max_x,y -[x^2 + y^2] + lambd * (x + 2y - 3) , x >= 0, y >= 0
    #                    max min_x,y  [x^2 + y^2] - lambd * (x + 2y - 3) , x >= 0, y >= 0
    def __init__(self):
        self.lambd_size = 1
        self.subgradient = np.zeros(self.lambd_size)
        self.function_value = 0.0
        self.f_star = 0.0
        # solve problem by cvxpy
        import cvxpy as cp
        x = cp.Variable()
        y = cp.Variable()
        constraints = [x + 2*y - 3 == 0, x >= 0, y >= 0]
        problem = cp.Problem(cp.Minimize(x**2 + y**2), constraints)
        problem.solve()
        print(f"    Oracle2: optimal solution = ({x.value:.2f}, {y.value:.2f}), optimal value = {problem.value:.2f}")
        

    def solve_subproblem(self):
        lambd = self.lambd
        # subgradient = x* + 2y* - 3, where (x*, y*) is the optimal solution of
        # the inner problem 
        # optimize: max -[x^2 + y^2] + lambd * (x + 2y - 3)
        #  min x² + y² - λ(x + 2y - 3)
        #  = min x² - λx + y² - 2λy + 3λ
        #  = min (x - λ/2)² - λ²/4 + (y - λ)² - λ² + 3λ
        #  = min (x - λ/2)² + (y - λ)² - 5λ²/4 + 3λ
        # The optimal solution can be found analytically:
        x_star = max(0, lambd[0] / 2) #x_star >= 0
        y_star = max(0, lambd[0]) #y_star >= 0
        subgradient = -np.array([x_star + 2*y_star - 3])
        function_value = x_star**2 + y_star**2 - lambd[0] * (x_star + 2*y_star - 3)
        f_star = subgradient @ lambd - function_value
        self.subgradient = -subgradient
        self.function_value = -function_value
        self.f_star = -f_star
        self.has_changed = False
    
class TestOracle3(Oracle):
    # solving min f(lambd) = max_x,y -[x^2 + y^2 + z^2] + lambd_0 * (x + 2y - 3) + lambd_1 * (z + y - 1), x >= 0, y >= 0, z >= 0
    #                    max min_x,y  [x^2 + y^2 + z^2] - lambd_0 * (x + 2y - 3) - lambd_1 * (z + y - 1), x >= 0, y >= 0, z >= 0
    def __init__(self):
        self.lambd_size = 2
        # solve problem by cvxpy
        import cvxpy as cp
        x = cp.Variable()
        y = cp.Variable()
        z = cp.Variable()
        constraints = [x + 2*y - 3 == 0, z + y - 1 == 0, x >= 0, y >= 0, z >= 0]
        problem = cp.Problem(cp.Minimize(x**2 + y**2 + z**2), constraints)
        problem.solve()
        print(f"    Oracle3: optimal solution = ({x.value:.2f}, {y.value:.2f}, {z.value:.2f}), optimal value = {problem.value:.2f}")
        

    def solve_subproblem(self):
        lambd = self.lambd
        # subgradient = [x* + 2y* - 3, z* + y* - 1], where (x*, y*, z*) is the optimal solution of
        # the inner problem 
        # optimize: max -[x^2 + y^2 + z^2] + lambd_0 * (x + 2y - 3) + lambd_1 * (z + y - 1)
        #  min x² + y² + z² - λ₀(x + 2y - 3) - λ₁(z + y - 1)
        #  = min x² - λ₀x + y² - (2λ₀ + λ₁)y + z² - λ₁z - λ₁y + 3λ₀ + λ₁
        #  = min (x - λ₀/2)² + (y - (2λ₀ + λ₁))² + (z - λ₁)² - 5λ₀²/4 + 3λ₀ - λ₁²/4 + 3λ₁
        # The optimal solution can be found analytically:
        x_star = max(0, lambd[0] / 2) #x_star >= 0
        y_star = max(0, lambd[0] + 0.5*lambd[1]) #y_star >= 0
        z_star = max(0, lambd[1]/2) #z_star >= 0
        subgradient = -np.array([x_star + 2*y_star - 3, z_star + y_star - 1])
        function_value = (x_star**2 + y_star**2 + z_star**2) - lambd[0] * (x_star + 2*y_star - 3) - lambd[1] * (z_star + y_star - 1)
        f_star = subgradient @ lambd - function_value
        # print(f"    Oracle3: lambd = {lambd}, x* = {x_star:.2f}, y* = {y_star:.2f}, z* = {z_star:.2f}, function_value = {function_value:.2f}, subgradient = {subgradient}, f_star = {f_star:.2f}")
        self.subgradient = -subgradient
        self.function_value = -function_value
        self.f_star = -f_star
        self.has_changed = False
    

class TestOracle4(Oracle):
    # solving min f(lambd) = max_x,y -[x^2 + y^2 + z^2] + lambd_0 * (x + 2y - 3) + lambd_1 * (z + y - 1), x >= 0, y >= 0, z >= 0
    #                    max min_x,y  [x^2 + y^2 + z^2] - lambd_0 * (x + 2y - 3) - lambd_1 * (z + y - 1), x >= 0, y >= 0, z >= 0
    #  lamda_0 >= 0, lambda_1 >= 0
    def __init__(self):
        self.lambd_size = 2
        self.lambd_lb = np.array([0.0, 0.0])  # lower bound for lambda
        # solve problem by cvxpy
        import cvxpy as cp
        x = cp.Variable()
        y = cp.Variable()
        z = cp.Variable()
        constraints = [x + 2*y - 3 >= 0, z + y - 1 >= 0, x >= 0, y >= 0, z >= 0]
        problem = cp.Problem(cp.Minimize(x**2 + y**2 + z**2), constraints)
        problem.solve()
        print(f"    Oracle4: optimal solution = ({x.value:.2f}, {y.value:.2f}, {z.value:.2f}), optimal value = {problem.value:.2f}")
        

    def solve_subproblem(self):
        lambd = self.lambd
        # subgradient = [x* + 2y* - 3, z* + y* - 1], where (x*, y*, z*) is the optimal solution of
        # the inner problem 
        # optimize: max -[x^2 + y^2 + z^2] + lambd_0 * (x + 2y - 3) + lambd_1 * (z + y - 1)
        #  min x² + y² + z² - λ₀(x + 2y - 3) - λ₁(z + y - 1)
        #  = min x² - λ₀x + y² - (2λ₀ + λ₁)y + z² - λ₁z - λ₁y + 3λ₀ + λ₁
        #  = min (x - λ₀/2)² + (y - (2λ₀ + λ₁))² + (z - λ₁)² - 5λ₀²/4 + 3λ₀ - λ₁²/4 + 3λ₁
        # The optimal solution can be found analytically:
        x_star = max(0, lambd[0] / 2) #x_star >= 0
        y_star = max(0, lambd[0] + 0.5*lambd[1]) #y_star >= 0
        z_star = max(0, lambd[1]/2) #z_star >= 0
        subgradient = -np.array([x_star + 2*y_star - 3, z_star + y_star - 1])
        function_value = (x_star**2 + y_star**2 + z_star**2) -   lambd[0] * (x_star + 2*y_star - 3) - lambd[1] * (z_star + y_star - 1)
        f_star = subgradient @ lambd - function_value
        print(f"    Oracle4: lambd = {lambd}, x* = {x_star:.2f}, y* = {y_star:.2f}, z* = {z_star:.2f}, function_value = {function_value:.2f}, subgradient = {subgradient}, f_star = {f_star:.2f}")
        
        self.subgradient = -subgradient
        self.function_value = -function_value
        self.f_star = -f_star
        self.has_changed = False

class TestOracle5(Oracle):
    # solving max min_x,y  [X + 4y - z] - lambd_0 * (x + 2y - 3) - lambd_1 * (z + y - 1), x >= 0, y >= 0, z >= 0
    #  lamda_0 >= 0, lambda_1 >= 0
    def __init__(self):
        self.lambd_size = 2
        self.lambd_lb = np.array([0.0, 0.0])  # lower bound for lambda
        # solve problem by cvxpy
        import cvxpy as cp
        x = cp.Variable()
        y = cp.Variable()
        z = cp.Variable()
        constraints = [x + 2*y - 3 >= 0, z + y - 1 >= 0, x >= 0, y >= 0, z >= 0, x <= 1, y <= 1, z <= 1]
        problem = cp.Problem(cp.Minimize(x + 4*y - z), constraints)
        problem.solve()
        print(f"    Oracle5: optimal solution = ({x.value:.2f}, {y.value:.2f}, {z.value:.2f}), optimal value = {problem.value:.2f}")
        

    def solve_subproblem(self):
        lambd = self.lambd
        # subgradient = [x* + 2y* - 3, z* + y* - 1], where (x*, y*, z*) is the optimal solution of
        # the inner problem 
        # optimize: max -[x + 4y - z] + lambd_0 * (x + 2y - 3) + lambd_1 * (z + y - 1)
        #  min x + 4y - z - λ₀(x + 2y - 3) - λ₁(z + y - 1)
        #  = min (1 - λ₀)x + (4 - 2λ₀ - λ₁)y + (-1 - λ₁)z + 3λ₀ + λ₁
        x_star = 0 if (1 - lambd[0]) >= 0 else 1 #x_star >= 0
        y_star = 0 if (4 - 2*lambd[0] - lambd[1]) >= 0 else 1 #y_star >= 0
        z_star = 0 if (-1 - lambd[1]) >= 0 else 1 #z_star >= 0
        subgradient = -np.array([x_star + 2*y_star - 3, z_star + y_star - 1])
        function_value = (x_star + 4*y_star - z_star) - lambd[0] * (x_star + 2*y_star - 3) + lambd[1] * (z_star + y_star - 1)
        f_star = subgradient @ lambd - function_value
        print(f"    Oracle5: lambd = {lambd}, x* = {x_star:.2f}, y* = {y_star:.2f}, z* = {z_star:.2f}, function_value = {function_value:.2f}, subgradient = {subgradient}, f_star = {f_star:.2f}")
        
        self.subgradient = -subgradient
        self.function_value = -function_value
        self.f_star = -f_star
        self.has_changed = False

