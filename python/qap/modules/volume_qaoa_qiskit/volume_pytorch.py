from dataclasses import dataclass

@dataclass
class VolumeParams:
    lambdainit: float = 0.1
    alphainit: float = 0.01
    alphamin: float = 1e-3
    alphafactor: float = 0.5
    ubinit: float = float("inf")

    primal_abs_precision: float = 0.02
    gap_abs_precision: float = 0.0
    gap_rel_precision: float = 1e-3
    granularity: float = 0.0

    minimum_rel_ascent: float = 1e-4
    ascent_first_check: int = 500
    ascent_check_invl: int = 100

    maxsgriters: int = 2000
    heurinvl: int = 10**8
    printinvl: int = 50
    
import abc
import torch

class VolumeHooks(abc.ABC):

    @abc.abstractmethod
    def solve_subproblem(self, u: torch.Tensor, rc: torch.Tensor):
        """
        Returns:
          lcost: float
          x: torch.Tensor
          v: torch.Tensor (constraint violation)
          pcost: float
        """
        pass

    @abc.abstractmethod
    def heuristics(self, x: torch.Tensor) -> float:
        """Optional primal heuristic"""
        pass
    
def dual_step(u, v, target, lcost, lambda_, lb, ub):
    """
    All tensors live on GPU
    """
    mask = ((v > 0) & (u < ub)) | ((v < 0) & (u > lb))
    viol = torch.sum(v[mask] ** 2)

    if viol.item() == 0.0:
        return u

    step = (target - lcost) / viol * lambda_
    u_new = u + step * v
    return torch.clamp(u_new, lb, ub)

class VolumeAlgorithm:
    def __init__(self, psize, dsize, hooks: VolumeHooks,
                 params: VolumeParams,
                 device="cuda"):
        self.psize = psize
        self.dsize = dsize
        self.hooks = hooks
        self.params = params
        self.device = device

        self.alpha = params.alphainit
        self.lambda_ = params.lambdainit

        self.dual_lb = torch.full((dsize,), -torch.inf, device=device)
        self.dual_ub = torch.full((dsize,),  torch.inf, device=device)

    def solve(self):
        p = self.params

        # Dual variables
        u = torch.zeros(self.dsize, device=self.device)

        # Initial solve
        lcost, x, v, pcost = self.hooks.solve_subproblem(u)
        pstar_x = x.clone()
        pstar_v = v.clone()
        pstar_val = pcost

        best_dual = lcost
        target = max(lcost + 0.05 * abs(lcost), 10.0)

        for it in range(1, p.maxsgriters + 1):
            u_old = u.clone()

            # Dual step
            u = dual_step(u, pstar_v, target, lcost,
                          self.lambda_, self.dual_lb, self.dual_ub)

            rc = self.hooks.compute_rc(u)
            lcost, x, v, pcost = self.hooks.solve_subproblem(u, rc)

            if lcost > best_dual:
                best_dual = lcost

            # Convex combination (volume update)
            alpha_fb = self._alpha_feedback(v, pstar_v, u)
            pstar_x = torch.lerp(pstar_x, x, alpha_fb)
            pstar_v = torch.lerp(pstar_v, v, alpha_fb)
            pstar_val = alpha_fb * pcost + (1 - alpha_fb) * pstar_val

            # Termination checks
            viol = torch.max(torch.abs(pstar_v)).item()
            gap = pstar_val - best_dual

            if viol < p.primal_abs_precision and (
                abs(gap) < p.gap_abs_precision or
                abs(gap / best_dual) < p.gap_rel_precision
            ):
                break

            if it % p.printinvl == 0:
                print(f"[{it:5d}] L={lcost:.6f} P={pstar_val:.6f} viol={viol:.3e}")

        return {
            "dual_value": best_dual,
            "primal_x": pstar_x,
            "primal_value": pstar_val,
            "dual_u": u
        }

    def _alpha_feedback(self, v, vstar, u):
        """
        Power heuristic (VOL_problem::power_heur)
        """
        vv = torch.dot(v, v)
        vh = torch.dot(v, vstar)
        hh = torch.dot(vstar, vstar)

        denom = vv + hh - 2 * vh
        if denom <= 0:
            return self.alpha

        alpha_fb = (hh - vh) / denom
        alpha_fb = torch.clamp(
            torch.tensor(alpha_fb, device=self.device),
            0.0, self.alpha
        ).item()
        return alpha_fb