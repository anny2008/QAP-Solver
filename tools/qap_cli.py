
#!/usr/bin/env python3
"""Unified build/run interface for QAP-Solver modules.

Supports:
- C++ binaries: sfd_scip, rlt1_scip, local_search (2-opt, ils, tabu)
- Python solvers: rlt1_cplex, rlt1_scip_py

Examples:
    # Build C++ modules
    python tools/qap_cli.py build --module sfd_scip rlt1_scip local_search

    # Run Local Search (tabu search) with config
    python tools/qap_cli.py run --module local_search --config configs/local_search.json

    # Run SFD (SCIP) with config
    python tools/qap_cli.py run --module sfd_scip --config configs/sfd_scip.json

    # Run RLT1 (SCIP C++) with instance and warmstart
    python tools/qap_cli.py run --module rlt1_scip --instance data/chr12a.dat --warmstart data/chr12a.sln --threads 8 --time-limit 300

    # Run RLT1 (CPLEX Python) using config and write output JSON
    python tools/qap_cli.py run --module rlt1_cplex --instance data/chr12a.dat --config configs/rlt1_cplex.json --output results.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent

# Ensure local Python packages are importable (add repo's python/ to sys.path)
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))


@dataclass
class ModuleSpec:
    name: str
    kind: str  # "cpp" or "python"
    workdir: Path
    binary: Optional[Path] = None
    build_cmd: Optional[list[str]] = None
    default_config: Optional[Path] = None
    runner: Optional[Callable[["Args", "ModuleSpec"], int]] = None


# Runner implementations -----------------------------------------------------

def run_sfd_scip(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for sfd_scip")
    if not args.instance and not args.config:
        raise ValueError("SFD SCIP requires --instance or --config")

    cmd = [str(spec.binary)]

    if args.config:
        cmd += ["--config", str(_resolve_path(args.config))]
    else:
        cmd.append(str(_resolve_path(args.instance)))
        if args.warmstart:
            cmd += ["--warmstart", str(_resolve_path(args.warmstart))]
        if args.decomposition:
            cmd += ["--decomposition", args.decomposition]
        if args.time_limit is not None:
            cmd += ["--time", str(args.time_limit)]
        if args.threads is not None:
            cmd += ["--threads", str(args.threads)]
        if args.relax:
            cmd.append("--relax")
        if args.log:
            cmd.append("--log")

    return _run_subprocess(cmd, spec.workdir)


def run_rlt1_scip_cpp(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for rlt1_scip")
    
    # Load config if provided
    config = _load_config(args.config, spec.default_config)
    
    # Determine instance path (CLI arg > config > error)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("RLT1 SCIP (C++) requires --instance or instance in config")

    cmd = [str(spec.binary), str(_resolve_path(instance))]
    
    # Warmstart: CLI > config
    warmstart = args.warmstart or config.get("warmstart")
    if warmstart:
        cmd += ["--warmstart", str(_resolve_path(warmstart))]
    
    if args.output:
        cmd += ["--output", str(_resolve_path(args.output))]
    
    formulation = config.get("formulation")
    if formulation:
        cmd += ["--formulation", str(formulation)]

    # Time limit: CLI > config
    time_limit = args.time_limit or config.get("time_limit")
    if time_limit is not None:
        cmd += ["--time", str(time_limit)]
    
    # Threads: CLI > config
    threads = args.threads or config.get("threads")
    if threads is not None:
        cmd += ["--threads", str(threads)]
    
    # Log: CLI > config
    if args.log or config.get("log_output", False):
        cmd.append("--log")
        
    # Relaxation mode (config only for now): "default" or "volume"
    relax_mode = config.get("relaxation")
    if relax_mode in {"default", "volume"}:
        cmd += ["--relaxation", relax_mode]

    # Relaxation info (extra handler output)
    relax_info = config.get("relaxation_info")
    if relax_info is True or relax_info == 1 or relax_info == "true":
        cmd.append("--relaxation-info")

    return _run_subprocess(cmd, spec.workdir)

def run_rlt1_gurobi_reduction_IV_cpp(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for rlt1_gurobi_reduction_IV")
    # Load config if provided
    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("RLT1 Gurobi Reduction IV (C++) requires --instance or instance in config")
    cmd = [str(spec.binary), str(_resolve_path(instance))]
    # Always pass config file (default or specified)
    config_path = args.config if args.config else str(spec.default_config)
    if config_path:
        cmd += ["--config", str(_resolve_path(config_path))]
    return _run_subprocess(cmd, spec.workdir)

def run_rlt1_gurobi_reduction_IV_cpp_2(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for rlt1_gurobi_reduction_IV_cpp_2")
    # Load config if provided
    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("RLT1 Gurobi Reduction IV (C++) requires --instance or instance in config")
    cmd = [str(spec.binary), str(_resolve_path(instance))]
    # Always pass config file (default or specified)
    config_path = args.config if args.config else str(spec.default_config)
    if config_path:
        cmd += ["--config", str(_resolve_path(config_path))]
    return _run_subprocess(cmd, spec.workdir)

def run_rlt1_gurobi_cpp(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for rlt1_gurobi")
    # Load config if provided
    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("RLT1 Gurobi (C++) requires --instance or instance in config")
    cmd = [str(spec.binary), str(_resolve_path(instance))]
    # Always pass config file (default or specified)
    config_path = args.config if args.config else str(spec.default_config)
    if config_path:
        cmd += ["--config", str(_resolve_path(config_path))]
    return _run_subprocess(cmd, spec.workdir)

def run_KBXY_gurobi_cpp(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for KBXY_gurobi")
    # Load config if provided
    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("KBXY Gurobi (C++) requires --instance or instance in config")
    cmd = [str(spec.binary), str(_resolve_path(instance))]
    # Always pass config file (default or specified)
    config_path = args.config if args.config else str(spec.default_config)
    if config_path:
        cmd += ["--config", str(_resolve_path(config_path))]
    return _run_subprocess(cmd, spec.workdir)

def run_RLT1_cg(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for rlt1_cg_cplex")
    # Load config if provided
    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("RLT1 Column Generation CPLEX (C++) requires --instance or instance in config")
    cmd = [str(spec.binary), str(_resolve_path(instance))]
    # Always pass config file (default or specified)
    config_path = args.config if args.config else str(spec.default_config)
    if config_path:
        cmd += ["--config", str(_resolve_path(config_path))]
    return _run_subprocess(cmd, spec.workdir)

def run_RLT1_cg_gurobi(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for rlt1_cg_cplex")
    # Load config if provided
    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("RLT1 Column Generation Gurobi (C++) requires --instance or instance in config")
    cmd = [str(spec.binary), str(_resolve_path(instance))]
    # Always pass config file (default or specified)
    config_path = args.config if args.config else str(spec.default_config)
    if config_path:
        cmd += ["--config", str(_resolve_path(config_path))]
    return _run_subprocess(cmd, spec.workdir)

def run_sfd_volume(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for sfd_volume")
    # Load config if provided
    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("RLT1 Volume requires --instance or instance in config")
    else:
        print(f"[info] using instance: {instance}")

    cmd = [str(spec.binary), str(_resolve_path(instance))]

    # Time limit: CLI > config > default
    time_limit = args.time_limit or config.get("time_limit")
    if time_limit is not None:
        cmd += ["--time", str(time_limit)]
    
    # Threads: CLI > config > default
    threads = args.threads or config.get("threads")
    if threads is not None:
        cmd += ["--threads", str(threads)]
    
    # Log: CLI > config
    if args.log or config.get("log_output", False):
        cmd.append("--log")
    
    if args.output:
        cmd += ["--output", str(_resolve_path(args.output))]
    
    # Dual vector save/load from config
    save_dual = config.get("save_dual", "")
    if save_dual:
        cmd += ["--save-dual", str(_resolve_path(save_dual))]
    
    load_dual = config.get("load_dual", "")
    if load_dual:
        cmd += ["--load-dual", str(_resolve_path(load_dual))]

    # Fixed variables (file) from CLI or config key "fixed"
    fixed_path = args.fixed or config.get("fixed")
    if fixed_path:
        cmd += ["--fixed", str(_resolve_path(fixed_path))]

    return _run_subprocess(cmd, spec.workdir)

def run_rlt1_volume(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for rlt1_volume")
    
    # Load config if provided
    config = _load_config(args.config, spec.default_config)
    
    # Determine instance path (CLI arg > config > error)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("RLT1 Volume requires --instance or instance in config")

    cmd = [str(spec.binary), str(_resolve_path(instance))]
    
    # Time limit: CLI > config > default
    time_limit = args.time_limit or config.get("time_limit")
    if time_limit is not None:
        cmd += ["--time", str(time_limit)]
    
    # Threads: CLI > config > default
    threads = args.threads or config.get("threads")
    if threads is not None:
        cmd += ["--threads", str(threads)]
    
    # Log: CLI > config
    if args.log or config.get("log_output", False):
        cmd.append("--log")
    
    output_path = config.get("output_path")
    if output_path:
        cmd += ["--output_path", str(_resolve_path(output_path))]

    # Dual vector save/load from config
    save_dual = config.get("save_dual", "")
    if save_dual:
        cmd += ["--save-dual", str(_resolve_path(save_dual))]
    
    load_dual = config.get("load_dual", "")
    if load_dual:
        cmd += ["--load-dual", str(_resolve_path(load_dual))]
        
        
    formulation = config.get("formulation", "")
    if formulation:
        cmd += ["--formulation", str(formulation)]

    # Fixed variables (file) from CLI or config key "fixed"
    fixed_path = args.fixed or config.get("fixed")
    if fixed_path:
        cmd += ["--fixed", str(_resolve_path(fixed_path))]

    return _run_subprocess(cmd, spec.workdir)

def run_rlt1_bundle(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for rlt1_bundle")
    
    # Load config if provided
    config = _load_config(args.config, spec.default_config)
    
    # Determine instance path (CLI arg > config > error)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("RLT1 Bundle requires --instance or instance in config")

    cmd = [str(spec.binary), str(_resolve_path(instance))]
    
    # Time limit: CLI > config > default
    time_limit = args.time_limit or config.get("time_limit")
    if time_limit is not None:
        cmd += ["--time", str(time_limit)]
    
    iteration_limit = config.get("iteration_limit")
    if iteration_limit is not None:
        cmd += ["--iteration-limit", str(iteration_limit)]
    
    # Threads: CLI > config > default
    threads = args.threads or config.get("threads")
    if threads is not None:
        cmd += ["--threads", str(threads)]
    
    # Log: CLI > config
    if args.log or config.get("log_output", False):
        cmd.append("--log")
    
    output_path = config.get("output_path")
    if output_path:
        cmd += ["--output_path", str(_resolve_path(output_path))]

    # Dual vector save/load from config
    save_dual = config.get("save_dual", "")
    if save_dual:
        cmd += ["--save-dual", str(_resolve_path(save_dual))]
    
    load_dual = config.get("load_dual", "")
    if load_dual:
        cmd += ["--load-dual", str(_resolve_path(load_dual))]
        
        
    formulation = config.get("formulation", "")
    if formulation:
        cmd += ["--formulation", str(formulation)]

    # Fixed variables (file) from CLI or config key "fixed"
    fixed_path = args.fixed or config.get("fixed")
    if fixed_path:
        cmd += ["--fixed", str(_resolve_path(fixed_path))]

    return _run_subprocess(cmd, spec.workdir)

def run_sdp_bundle(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for sdp_bundle")
    
    # Load config if provided
    config = _load_config(args.config, spec.default_config)
    
    # Determine instance path (CLI arg > config > error)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("SDP Bundle requires --instance or instance in config")

    cmd = [str(spec.binary), str(_resolve_path(instance))]
    
    # Time limit: CLI > config > default
    time_limit = args.time_limit or config.get("time_limit")
    if time_limit is not None:
        cmd += ["--time", str(time_limit)]
    
    iteration_limit = config.get("iteration_limit")
    if iteration_limit is not None:
        cmd += ["--iteration-limit", str(iteration_limit)]

    mSS = config.get("mSS")
    if mSS is not None:
        cmd += ["--mSS", str(mSS)]
        
    # Threads: CLI > config > default
    threads = args.threads or config.get("threads")
    if threads is not None:
        cmd += ["--threads", str(threads)]
    
    # Log: CLI > config
    if args.log or config.get("log_output", False):
        cmd.append("--log")
    
    output_path = config.get("output_path")
    if output_path:
        cmd += ["--output_path", str(_resolve_path(output_path))]

    # Dual vector save/load from config
    save_dual = config.get("save_dual", "")
    if save_dual:
        cmd += ["--save-dual", str(_resolve_path(save_dual))]
    
    load_dual = config.get("load_dual", "")
    if load_dual:
        cmd += ["--load-dual", str(_resolve_path(load_dual))]
        
        
    formulation = args.formulation or config.get("formulation", "")
    if formulation:
        cmd += ["--formulation", str(formulation)]

    # Fixed variables (file) from CLI or config key "fixed"
    fixed_path = args.fixed or config.get("fixed")
    if fixed_path:
        cmd += ["--fixed", str(_resolve_path(fixed_path))]

    return _run_subprocess(cmd, spec.workdir)

def run_sdp_volume(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for sdp_volume")
    
    # Load config if provided
    config = _load_config(args.config, spec.default_config)
    
    # Determine instance path (CLI arg > config > error)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("SDP Bundle requires --instance or instance in config")

    cmd = [str(spec.binary), str(_resolve_path(instance))]
    
    # Time limit: CLI > config > default
    time_limit = args.time_limit or config.get("time_limit")
    if time_limit is not None:
        cmd += ["--time", str(time_limit)]
    
    iteration_limit = config.get("iteration_limit")
    if iteration_limit is not None:
        cmd += ["--iteration-limit", str(iteration_limit)]
        
    lambda_init = config.get("lambda_init")
    if lambda_init is not None:
        cmd += ["--lambda-init", str(lambda_init)]
        
    alpha_init = config.get("alpha_init")
    if alpha_init is not None:
        cmd += ["--alpha-init", str(alpha_init)]
        
    ncv_init = config.get("ncv_init")
    if ncv_init is not None:
        cmd += ["--ncv-init", str(ncv_init)]
        
    tol_init = config.get("tol_init")
    if tol_init is not None:
        cmd += ["--tol-init", str(tol_init)]

    eigen_solver = config.get("eigen_solver")
    if eigen_solver:
        cmd += ["--eigen-solver", str(eigen_solver)]
        
    C = config.get("C")
    if C is not None:
        cmd += ["--C", str(C)]
        
    # Threads: CLI > config > default
    threads = args.threads or config.get("threads")
    if threads is not None:
        cmd += ["--threads", str(threads)]
    
    # Log: CLI > config
    if args.log or config.get("log_output", False):
        cmd.append("--log")
    
    output_path = config.get("output_path")
    if output_path:
        cmd += ["--output_path", str(_resolve_path(output_path))]

    # Dual vector save/load from config
    save_dual = config.get("save_dual", "")
    if save_dual:
        cmd += ["--save-dual", str(_resolve_path(save_dual))]
    
    load_dual = config.get("load_dual", "")
    if load_dual:
        cmd += ["--load-dual", str(_resolve_path(load_dual))]
        
        
    formulation = args.formulation or config.get("formulation", "")
    if formulation:
        cmd += ["--formulation", str(formulation)]

    # Fixed variables (file) from CLI or config key "fixed"
    fixed_path = args.fixed or config.get("fixed")
    if fixed_path:
        cmd += ["--fixed", str(_resolve_path(fixed_path))]

    return _run_subprocess(cmd, spec.workdir)

def run_sdp_volume_2(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for sdp_volume")
    
    # Load config if provided
    config = _load_config(args.config, spec.default_config)
    
    # Determine instance path (CLI arg > config > error)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("SDP Bundle requires --instance or instance in config")

    cmd = [str(spec.binary), str(_resolve_path(instance))]
    
    # Time limit: CLI > config > default
    time_limit = args.time_limit or config.get("time_limit")
    if time_limit is not None:
        cmd += ["--time", str(time_limit)]
    
    iteration_limit = config.get("iteration_limit")
    if iteration_limit is not None:
        cmd += ["--iteration-limit", str(iteration_limit)]
        
    lambda_init = config.get("lambda_init")
    if lambda_init is not None:
        cmd += ["--lambda-init", str(lambda_init)]
        
    alpha_init = config.get("alpha_init")
    if alpha_init is not None:
        cmd += ["--alpha-init", str(alpha_init)]
        
    eigen_solver = config.get("eigen_solver")
    if eigen_solver:
        cmd += ["--eigen-solver", str(eigen_solver)]
        
    C = config.get("C")
    if C is not None:
        cmd += ["--C", str(C)]
        
    # Threads: CLI > config > default
    threads = args.threads or config.get("threads")
    if threads is not None:
        cmd += ["--threads", str(threads)]
    
    # Log: CLI > config
    if args.log or config.get("log_output", False):
        cmd.append("--log")
    
    output_path = config.get("output_path")
    if output_path:
        cmd += ["--output_path", str(_resolve_path(output_path))]

    # Dual vector save/load from config
    save_dual = config.get("save_dual", "")
    if save_dual:
        cmd += ["--save-dual", str(_resolve_path(save_dual))]
    
    load_dual = config.get("load_dual", "")
    if load_dual:
        cmd += ["--load-dual", str(_resolve_path(load_dual))]
        
        
    formulation = args.formulation or config.get("formulation", "")
    if formulation:
        cmd += ["--formulation", str(formulation)]

    # Fixed variables (file) from CLI or config key "fixed"
    fixed_path = args.fixed or config.get("fixed")
    if fixed_path:
        cmd += ["--fixed", str(_resolve_path(fixed_path))]

    return _run_subprocess(cmd, spec.workdir)

def run_m4_volume(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for m4_volume")
    
    # Load config if provided
    config = _load_config(args.config, spec.default_config)
    
    # Determine instance path (CLI arg > config > error)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("M4 Volume requires --instance or instance in config")

    cmd = [str(spec.binary), str(_resolve_path(instance))]
    
    # Time limit: CLI > config > default
    time_limit = args.time_limit or config.get("time_limit")
    if time_limit is not None:
        cmd += ["--time", str(time_limit)]
    
    # Threads: CLI > config > default
    threads = args.threads or config.get("threads")
    if threads is not None:
        cmd += ["--threads", str(threads)]
    
    # Log: CLI > config
    if args.log or config.get("log_output", False):
        cmd.append("--log")

    # Dual vector save/load from config
    save_dual = config.get("save_dual", "")
    if save_dual:
        cmd += ["--save-dual", str(_resolve_path(save_dual))]
    
    load_dual = config.get("load_dual", "")
    if load_dual:
        cmd += ["--load-dual", str(_resolve_path(load_dual))]
        
        
    formulation = config.get("formulation", "")
    if formulation:
        cmd += ["--formulation", str(formulation)]

    # Fixed variables (file) from CLI or config key "fixed"
    fixed_path = args.fixed or config.get("fixed")
    if fixed_path:
        cmd += ["--fixed", str(_resolve_path(fixed_path))]

    return _run_subprocess(cmd, spec.workdir)


def run_rlt1_cplex(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.rlt1_cplex.solver import RLT1CPLEXSolver
    except ImportError as exc:
        print("RLT1 CPLEX solver missing dependencies (docplex/cplex)", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("RLT1 CPLEX requires --instance")
    solver = RLT1CPLEXSolver(config)

    # Warm-start: prefer CLI arg, else use config key 'warmstart' if provided
    warmstart_path = None
    if args.warmstart:
        warmstart_path = str(_resolve_path(args.warmstart))
    elif "warmstart" in config and config["warmstart"]:
        warmstart_path = str(_resolve_path(config["warmstart"]))

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=warmstart_path,
    )

    _print_solution_summary(solution)
    return 0

def run_rlt1_cplex_reduction_IV(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.rlt1_cplex_reduction_IV.solver import RLT1CPLEXSolver
    except ImportError as exc:
        print("RLT1 CPLEX solver missing dependencies (docplex/cplex)", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("RLT1 CPLEX requires --instance")
    solver = RLT1CPLEXSolver(config)

    # Warm-start: prefer CLI arg, else use config key 'warmstart' if provided
    warmstart_path = None
    if args.warmstart:
        warmstart_path = str(_resolve_path(args.warmstart))
    elif "warmstart" in config and config["warmstart"]:
        warmstart_path = str(_resolve_path(config["warmstart"]))

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=warmstart_path,
    )

    _print_solution_summary(solution)
    return 0

def run_rlt1_sdp(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.rlt1_sdp_cvxpy.solver import RLT1SDPSolver
    except ImportError as exc:
        print("RLT1 SDP solver missing dependencies (cvxpy)", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("RLT1 SDP requires --instance")
    solver = RLT1SDPSolver(config)

    # Warm-start: prefer CLI arg, else use config key 'warmstart' if provided
    warmstart_path = None
    if args.warmstart:
        warmstart_path = str(_resolve_path(args.warmstart))
    elif "warmstart" in config and config["warmstart"]:
        warmstart_path = str(_resolve_path(config["warmstart"]))

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=warmstart_path,
    )

    _print_solution_summary(solution)
    return 0

def run_rlt1_gurobi(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.rlt1_gurobi.solver import RLT1GurobiSolver
    except ImportError as exc:
        print("RLT1 Gurobi solver missing", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("RLT1 Gurobi requires --instance")
    solver = RLT1GurobiSolver(config)

    # Warm-start: prefer CLI arg, else use config key 'warmstart' if provided
    warmstart_path = None
    if args.warmstart:
        warmstart_path = str(_resolve_path(args.warmstart))
    elif "warmstart" in config and config["warmstart"]:
        warmstart_path = str(_resolve_path(config["warmstart"]))

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=warmstart_path,
    )

    _print_solution_summary(solution)
    return 0

def run_rlt1_gurobi_reduction_IV(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.rlt1_gurobi_reduction_IV.solver import RLT1GurobiSolver
    except ImportError as exc:
        print("RLT1 Gurobi solver missing", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("RLT1 Gurobi requires --instance")
    solver = RLT1GurobiSolver(config)

    # Warm-start: prefer CLI arg, else use config key 'warmstart' if provided
    warmstart_path = None
    if args.warmstart:
        warmstart_path = str(_resolve_path(args.warmstart))
    elif "warmstart" in config and config["warmstart"]:
        warmstart_path = str(_resolve_path(config["warmstart"]))

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=warmstart_path,
    )

    _print_solution_summary(solution)
    return 0

def run_m5_cplex(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.m5_cplex.solver import M5CPLEXSolver
    except ImportError as exc:
        print("M5 CPLEX solver missing dependencies (docplex/cplex)", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("M5 CPLEX requires --instance")
    solver = M5CPLEXSolver(config)

    # Warm-start: prefer CLI arg, else use config key 'warmstart' if provided
    warmstart_path = None
    if args.warmstart:
        warmstart_path = str(_resolve_path(args.warmstart))
    elif "warmstart" in config and config["warmstart"]:
        warmstart_path = str(_resolve_path(config["warmstart"]))

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=warmstart_path,
    )

    _print_solution_summary(solution)
    return 0

def run_local_search_python(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.local_search.solver import LocalSearchSolver
    except ImportError as exc:
        print("Local Search Python solver missing dependencies", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("Local Search Python requires --instance")
    solver = LocalSearchSolver(config)

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=str(_resolve_path(args.warmstart)) if args.warmstart else None,
    )

    _print_solution_summary(solution)
    return 0

def run_qaoa_qiskit(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.qaoa_qiskit.solver import QAOAQiskitSolver
    except ImportError as exc:
        print("QAOA Qiskit solver missing dependencies (qiskit)", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("QAOA Qiskit requires --instance")
    solver = QAOAQiskitSolver(config)

    # Warm-start: prefer CLI arg, else use config key 'warmstart' if provided
    warmstart_path = None
    if args.warmstart:
        warmstart_path = str(_resolve_path(args.warmstart))
    elif "warmstart" in config and config["warmstart"]:
        warmstart_path = str(_resolve_path(config["warmstart"]))

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=warmstart_path,
    )

    _print_solution_summary(solution)
    return 0

def run_sfdcg_cplex(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.sfd_cg_cplex.solver import BranchAndPriceSFD
    except ImportError as exc:
        print("SFDCG CPLEX solver missing dependencies (docplex/cplex)", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("SFDCG CPLEX requires --instance")
    solver = BranchAndPriceSFD(config)

    # Warm-start: prefer CLI arg, else use config key 'warmstart' if provided
    warmstart_path = None
    if args.warmstart:
        warmstart_path = str(_resolve_path(args.warmstart))
    elif "warmstart" in config and config["warmstart"]:
        warmstart_path = str(_resolve_path(config["warmstart"]))

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=warmstart_path,
    )

    _print_solution_summary(solution)
    return 0

def run_m5_cg_cplex_cpp(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for m5_cg_cplex")
    
    # Load config if provided
    config = _load_config(args.config, spec.default_config)
    
    # Determine instance path (CLI arg > config > error)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("M5 Column Generation CPLEX (C++) requires --instance")

    cmd = [str(spec.binary), str(_resolve_path(instance))]
    
    config_path = args.config if args.config else str(spec.default_config)
    config = _load_config(config_path, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("Gilmore-Lawler requires --instance or instance in config")
    cmd = [str(spec.binary), str(_resolve_path(instance))]
    # Always pass config file (default or specified)
    if args.config or spec.default_config:
        cmd += ["--config", str(_resolve_path(config_path))]
        

    return _run_subprocess(cmd, spec.workdir)
    

def run_volume_qaoa_qiskit(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.volume_qaoa_qiskit.solver import VolumeQAOAQiskitSolver
    except ImportError as exc:
        print("Volume QAOA Qiskit solver missing dependencies (qiskit)", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("Volume QAOA Qiskit requires --instance")
    solver = VolumeQAOAQiskitSolver(config)

    # Warm-start: prefer CLI arg, else use config key 'warmstart' if provided
    warmstart_path = None
    if args.warmstart:
        warmstart_path = str(_resolve_path(args.warmstart))
    elif "warmstart" in config and config["warmstart"]:
        warmstart_path = str(_resolve_path(config["warmstart"]))

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=warmstart_path,
    )

    _print_solution_summary(solution)
    return 0

def run_m4_cplex(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.m4.solver import M4CPLEXSolver
    except ImportError as exc:
        print("M4 CPLEX solver missing dependencies (docplex/cplex)", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("M4 CPLEX requires --instance")
    solver = M4CPLEXSolver(config)

    # Warm-start: prefer CLI arg, else use config key 'warmstart' if provided
    warmstart_path = None
    if args.warmstart:
        warmstart_path = str(_resolve_path(args.warmstart))
    elif "warmstart" in config and config["warmstart"]:
        warmstart_path = str(_resolve_path(config["warmstart"]))

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=warmstart_path,
    )

    _print_solution_summary(solution)
    return 0

def run_cg_cplex(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.column_generation.solver import ColumnGenerationCPLEXSolver
    except ImportError as exc:
        print("Column Generation CPLEX solver missing dependencies (docplex/cplex)", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("Column Generation CPLEX requires --instance")
    solver = ColumnGenerationCPLEXSolver(config)

    # Warm-start: prefer CLI arg, else use config key 'warmstart' if provided
    warmstart_path = None
    if args.warmstart:
        warmstart_path = str(_resolve_path(args.warmstart))
    elif "warmstart" in config and config["warmstart"]:
        warmstart_path = str(_resolve_path(config["warmstart"]))

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=warmstart_path,
    )

    _print_solution_summary(solution)
    return 0

def run_cg_cplex_cpp(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for column generation CPLEX")
    
    # Load config if provided
    config = _load_config(args.config, spec.default_config)
    
    # Determine instance path (CLI arg > config > error)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("Column Generation CPLEX (C++) requires --instance")

    cmd = [str(spec.binary), str(_resolve_path(instance))]
    
    config_path = args.config if args.config else str(spec.default_config)
    config = _load_config(config_path, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("Gilmore-Lawler requires --instance or instance in config")
    cmd = [str(spec.binary), str(_resolve_path(instance))]
    # Always pass config file (default or specified)
    if args.config or spec.default_config:
        cmd += ["--config", str(_resolve_path(config_path))]
        

    return _run_subprocess(cmd, spec.workdir)

def run_qubo_volume(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for qubo_volume")
    # Load config if provided
    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("QUBO Volume requires --instance or instance in config")
    else:
        print(f"[info] using instance: {instance}")

    cmd = [str(spec.binary), str(_resolve_path(instance))]

    # Time limit: CLI > config > default
    time_limit = args.time_limit or config.get("time_limit")
    if time_limit is not None:
        cmd += ["--time", str(time_limit)]
    
    # Threads: CLI > config > default
    threads = args.threads or config.get("threads")
    if threads is not None:
        cmd += ["--threads", str(threads)]
    
    # Log: CLI > config
    if args.log or config.get("log_output", False):
        cmd.append("--log")
    
    if args.output:
        cmd += ["--output", str(_resolve_path(args.output))]
    
    return _run_subprocess(cmd, spec.workdir)

def run_cg_va(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for column generation volume algorithm")
    # Load config if provided
    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("Column Generation Volume Algorithm requires --instance or instance in config")
    else:
        print(f"[info] using instance: {instance}")

    cmd = [str(spec.binary), str(_resolve_path(instance))]

    # Always forward config file so the C++ binary reads cg_va settings
    config_path = args.config if args.config else str(spec.default_config)
    if config_path:
        cmd += ["--config", str(_resolve_path(config_path))]

    # Time limit: CLI > config > default
    time_limit = args.time_limit or config.get("time_limit")
    if time_limit is not None:
        cmd += ["--time", str(time_limit)]
    
    # Threads: CLI > config > default
    threads = args.threads or config.get("threads")
    if threads is not None:
        cmd += ["--threads", str(threads)]
    
    # Log flag kept for CLI consistency (binary currently relies on config)
    if args.log or config.get("log_output", False):
        cmd.append("--log")
    
    if args.output:
        cmd += ["--output", str(_resolve_path(args.output))]
    
    return _run_subprocess(cmd, spec.workdir)

def run_rlt1_scip_py(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.rlt1_scip.solver import RLT1SCIPSolver
    except ImportError as exc:
        print("RLT1 SCIP Python solver missing dependencies (PySCIPOpt)", file=sys.stderr)
        raise exc

    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("RLT1 SCIP Python requires --instance")

    config = _load_config(args.config, spec.default_config)
    solver = RLT1SCIPSolver(config)
    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=str(_resolve_path(args.warmstart)) if args.warmstart else None,
    )

    _print_solution_summary(solution)
    return 0


def run_sfd_cplex(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.sfd_cplex.solver import SFDSolver
    except ImportError as exc:
        print("SFD CPLEX solver missing dependencies (docplex/cplex)", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("SFD CPLEX requires --instance")
    solver = SFDSolver(config)

    # Warm-start: prefer CLI arg, else use config key 'warmstart' if provided
    warmstart_path = None
    if args.warmstart:
        warmstart_path = str(_resolve_path(args.warmstart))
    elif "warmstart" in config and config["warmstart"]:
        warmstart_path = str(_resolve_path(config["warmstart"]))

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=warmstart_path,
    )

    _print_solution_summary(solution)
    return 0

def run_sfd_gurobi(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.sfd_gurobi.solver import SFDGurobiSolver
    except ImportError as exc:
        print("SFD Gurobi solver missing dependencies (gurobipy)", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("SFD Gurobi requires --instance")
    solver = SFDGurobiSolver(config)

    # Warm-start: prefer CLI arg, else use config key 'warmstart' if provided
    warmstart_path = None
    if args.warmstart:
        warmstart_path = str(_resolve_path(args.warmstart))
    elif "warmstart" in config and config["warmstart"]:
        warmstart_path = str(_resolve_path(config["warmstart"]))

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=warmstart_path,
    )

    _print_solution_summary(solution)
    return 0

def run_sfd_sdp(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.sfd_sdp_cvxpy.solver import SFDSDPSolver
    except ImportError as exc:
        print("SFD SDP solver missing dependencies (cvxpy)", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("SFD SDP requires --instance")
    solver = SFDSDPSolver(config)

    # Warm-start: prefer CLI arg, else use config key 'warmstart' if provided
    warmstart_path = None
    if args.warmstart:
        warmstart_path = str(_resolve_path(args.warmstart))
    elif "warmstart" in config and config["warmstart"]:
        warmstart_path = str(_resolve_path(config["warmstart"]))

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=warmstart_path,
    )

    _print_solution_summary(solution)
    return 0

def run_sfd_gurobi_cpp(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for sfd_gurobi")
    # Load config if provided
    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("SFD Gurobi (C++) requires --instance or instance in config")
    cmd = [str(spec.binary), str(_resolve_path(instance))]
    # Always pass config file (default or specified)
    config_path = args.config if args.config else str(spec.default_config)
    if config_path:
        cmd += ["--config", str(_resolve_path(config_path))]
    return _run_subprocess(cmd, spec.workdir)

def run_sfb_gurobi(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.sfb_gurobi.solver import SFBGurobiSolver
    except ImportError as exc:
        print("SFB Gurobi solver missing dependencies (gurobipy)", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("SFB Gurobi requires --instance")
    solver = SFBGurobiSolver(config)

    # Warm-start: prefer CLI arg, else use config key 'warmstart' if provided
    warmstart_path = None
    if args.warmstart:
        warmstart_path = str(_resolve_path(args.warmstart))
    elif "warmstart" in config and config["warmstart"]:
        warmstart_path = str(_resolve_path(config["warmstart"]))

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=warmstart_path,
    )

    _print_solution_summary(solution)
    return 0

def run_admm(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.admm.solver import ADMMSolver
    except ImportError as exc:
        print("ADMM solver missing dependencies (numpy, scipy)", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("ADMM requires --instance")
    solver = ADMMSolver(config)

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=str(_resolve_path(args.warmstart)) if args.warmstart else None,
    )

    _print_solution_summary(solution)
    return 0

def run_csdp_cvxpy(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.csdp_cvxpy.solver import CSDPSolver
    except ImportError as exc:
        print("C-SDP solver missing dependencies (cvxpy)", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("C-SDP requires --instance")
    solver = CSDPSolver(config)

    # Warm-start: prefer CLI arg, else use config key 'warmstart' if provided
    warmstart_path = None
    if args.warmstart:
        warmstart_path = str(_resolve_path(args.warmstart))
    elif "warmstart" in config and config["warmstart"]:
        warmstart_path = str(_resolve_path(config["warmstart"]))

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=warmstart_path,
    )

    _print_solution_summary(solution)
    return 0

def run_csdp(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.c_sdp_admm.solver import CSDPSolver
    except ImportError as exc:
        print("CSDP solver missing dependencies (numpy, scipy)", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("CSDP requires --instance")
    solver = CSDPSolver(config)

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=str(_resolve_path(args.warmstart)) if args.warmstart else None,
    )

    _print_solution_summary(solution)
    return 0

def run_usbs(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.usbs.solver import USBSSolver
    except ImportError as exc:
        print("USBS solver missing dependencies (numpy, scipy)", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("USBS requires --instance")
    solver = USBSSolver(config)

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=str(_resolve_path(args.warmstart)) if args.warmstart else None,
    )

    _print_solution_summary(solution)
    return 0

def run_sdp_pytorch(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.sdp_pytorch.solver import SDPPytorchSolver
    except ImportError as exc:
        print("SDP PyTorch solver missing dependencies (torch)", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("SDP PyTorch requires --instance")
    solver = SDPPytorchSolver(config)

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=str(_resolve_path(args.warmstart)) if args.warmstart else None,
    )

    _print_solution_summary(solution)
    return 0

def run_local_search(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for local_search")
    # Use default config if not specified
    config_path = args.config if args.config else str(spec.default_config)
    config = _load_config(config_path, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("Local search requires --instance or instance in config")
    cmd = [str(spec.binary), str(_resolve_path(instance))]
    # Always pass config file (default or specified)
    cmd += ["--config", str(_resolve_path(config_path))]
    # Output file
    if args.output:
        cmd += ["--output", str(_resolve_path(args.output))]
    # Max iterations
    max_iter = config.get("max_iterations")
    if max_iter is not None:
        cmd += ["--max-iterations", str(max_iter)]
    # Tabu tenure (for tabu search)
    tabu_tenure = config.get("tabu_tenure")
    if tabu_tenure is not None:
        cmd += ["--tabu-tenure", str(tabu_tenure)]
    # Initial solution type
    init_sol = config.get("initial_solution")
    if init_sol:
        cmd += ["--initial-solution", init_sol]
    # Seed
    seed = config.get("seed")
    if seed is not None:
        cmd += ["--seed", str(seed)]
    # Time limit
    time_limit = args.time_limit or config.get("time_limit")
    if time_limit is not None:
        cmd += ["--time", str(time_limit)]
    # Log
    if args.log or config.get("log_output", False):
        cmd.append("--log")
    print(f"[run] {' '.join(cmd)} (cwd={spec.workdir})")
    
    # Frequency penalty weight (for robust tabu search)
    freq_penalty_weight = config.get("freq_penalty_weight")
    if freq_penalty_weight is not None:
        cmd += ["--freq-penalty-weight", str(freq_penalty_weight)]
    # Assignment penalty weight (for robust tabu search)
    assign_penalty_weight = config.get("assign_penalty_weight")
    if assign_penalty_weight is not None:
        cmd += ["--assign-penalty-weight", str(assign_penalty_weight)]
    result = subprocess.run(cmd, cwd=spec.workdir)
    return result.returncode

def run_local_search_modify(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for local_search_modify")
    # Use default config if not specified
    config_path = args.config if args.config else str(spec.default_config)
    config = _load_config(config_path, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("Local search requires --instance or instance in config")
    cmd = [str(spec.binary), str(_resolve_path(instance))]
    # Always pass config file (default or specified)
    cmd += ["--config", str(_resolve_path(config_path))]
    # Output file
    if args.output:
        cmd += ["--output", str(_resolve_path(args.output))]
    # Max iterations
    max_iter = config.get("max_iterations")
    if max_iter is not None:
        cmd += ["--max-iterations", str(max_iter)]
    # Tabu tenure (for tabu search)
    tabu_tenure = config.get("tabu_tenure")
    if tabu_tenure is not None:
        cmd += ["--tabu-tenure", str(tabu_tenure)]
    # Initial solution type
    init_sol = config.get("initial_solution")
    if init_sol:
        cmd += ["--initial-solution", init_sol]
    # Seed
    seed = config.get("seed")
    if seed is not None:
        cmd += ["--seed", str(seed)]
    # Time limit
    time_limit = args.time_limit or config.get("time_limit")
    if time_limit is not None:
        cmd += ["--time", str(time_limit)]
    # Log
    if args.log or config.get("log_output", False):
        cmd.append("--log")
    print(f"[run] {' '.join(cmd)} (cwd={spec.workdir})")
    
    # Frequency penalty weight (for robust tabu search)
    freq_penalty_weight = config.get("freq_penalty_weight")
    if freq_penalty_weight is not None:
        cmd += ["--freq-penalty-weight", str(freq_penalty_weight)]
    # Assignment penalty weight (for robust tabu search)
    assign_penalty_weight = config.get("assign_penalty_weight")
    if assign_penalty_weight is not None:
        cmd += ["--assign-penalty-weight", str(assign_penalty_weight)]
    result = subprocess.run(cmd, cwd=spec.workdir)
    return result.returncode

# VNS runner implementation
def run_vns(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for vns")
    # Load config if provided
    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance") or config.get("input_file")
    if not instance:
        raise ValueError("VNS requires --instance or instance/input_file in config")
    cmd = [str(spec.binary), str(_resolve_path(instance))]
    # Pass config file if specified
    if args.config:
        cmd += ["--config", str(_resolve_path(args.config))]
    # Output file
    if args.output:
        cmd += ["--output", str(_resolve_path(args.output))]
    # Max iterations
    max_iter = args.time_limit or config.get("max_iterations")
    if max_iter is not None:
        cmd += ["--max_iters", str(max_iter)]
    # Seed
    seed = config.get("seed")
    if seed is not None:
        cmd += ["--seed", str(seed)]
    # Log
    if args.log or config.get("log_output", False):
        cmd.append("--log")
    print(f"[run] {' '.join(cmd)} (cwd={spec.workdir})")
    result = subprocess.run(cmd, cwd=spec.workdir)
    return result.returncode

# Gilmore-Lawler runner implementation
def run_gilmore_lawler(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for gilmore_lawler")
    config_path = args.config if args.config else str(spec.default_config)
    config = _load_config(config_path, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("Gilmore-Lawler requires --instance or instance in config")
    cmd = [str(spec.binary), str(_resolve_path(instance))]
    # Always pass config file (default or specified)
    if args.config or spec.default_config:
        cmd += ["--config", str(_resolve_path(config_path))]
    print(f"[run] {' '.join(cmd)} (cwd={spec.workdir})")
    result = subprocess.run(cmd, cwd=spec.workdir)
    return result.returncode

def run_qap_cplex(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.qap_cplex.solver import QAPCplexSolver
    except ImportError as exc:
        print("QAP CPLEX solver missing dependencies (docplex/cplex)", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("QAP CPLEX requires --instance")
    solver = QAPCplexSolver(config)

    # Warm-start: prefer CLI arg, else use config key 'warmstart' if provided
    warmstart_path = None
    if args.warmstart:
        warmstart_path = str(_resolve_path(args.warmstart))
    elif "warmstart" in config and config["warmstart"]:
        warmstart_path = str(_resolve_path(config["warmstart"]))

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=warmstart_path,
    )

    _print_solution_summary(solution)
    return 0

def run_qap_gurobi(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.qap_gurobi.solver import QAPGurobiSolver
    except ImportError as exc:
        print("QAP Gurobi solver missing dependencies (gurobipy)", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("QAP Gurobi requires --instance")
    solver = QAPGurobiSolver(config)

    # Warm-start: prefer CLI arg, else use config key 'warmstart' if provided
    warmstart_path = None
    if args.warmstart:
        warmstart_path = str(_resolve_path(args.warmstart))
    elif "warmstart" in config and config["warmstart"]:
        warmstart_path = str(_resolve_path(config["warmstart"]))

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=warmstart_path,
    )

    _print_solution_summary(solution)
    return 0

def run_sfdcg_cplex_cpp(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for sfdcg_cplex")
    
    # Load config if provided
    config = _load_config(args.config, spec.default_config)
    
    # Determine instance path (CLI arg > config > error)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("SFDCG CPLEX (C++) requires --instance")

    cmd = [str(spec.binary), str(_resolve_path(instance))]
    
    config_path = args.config if args.config else str(spec.default_config)
    config = _load_config(config_path, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("Gilmore-Lawler requires --instance or instance in config")
    cmd = [str(spec.binary), str(_resolve_path(instance))]
    # Always pass config file (default or specified)
    if args.config or spec.default_config:
        cmd += ["--config", str(_resolve_path(config_path))]
        

    return _run_subprocess(cmd, spec.workdir)

# M4 Formulation runner implementation
def run_m4_formulation(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.m4_formulation.solver import M4FormulationSolver
    except ImportError as exc:
        print("M4FormulationSolver missing dependencies (docplex/cplex or others)", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("M4Formulation requires --instance")
    solver = M4FormulationSolver(config)

    # Warm-start: prefer CLI arg, else use config key 'warmstart' if provided
    warmstart_path = None
    if args.warmstart:
        warmstart_path = str(_resolve_path(args.warmstart))
    elif "warmstart" in config and config["warmstart"]:
        warmstart_path = str(_resolve_path(config["warmstart"]))

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=warmstart_path,
    )

    if solution is not None:
        _print_solution_summary(solution)
    return 0

# Helpers --------------------------------------------------------------------

def _resolve_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def _load_config(config_path: Optional[str], default_path: Optional[Path]) -> Dict:
    path = None
    if config_path:
        path = _resolve_path(config_path)
    elif default_path:
        path = default_path

    if path is None:
        return {}

    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _run_subprocess(cmd: list[str], cwd: Path) -> int:
    print(f"[run] {' '.join(cmd)} (cwd={cwd})")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def _build_module(spec: ModuleSpec, clean: bool = False) -> int:
    if spec.kind != "cpp":
        print(f"[skip] {spec.name} is python-only; no build needed")
        return 0
    if not spec.build_cmd:
        print(f"[skip] {spec.name} has no build command configured")
        return 0

    cmd = list(spec.build_cmd)
    if clean and "make" in spec.build_cmd[0]:
        cmd = [spec.build_cmd[0], "clean"]
        print(f"[clean] {' '.join(cmd)} (cwd={spec.workdir})")
        clean_rc = subprocess.run(cmd, cwd=spec.workdir).returncode
        if clean_rc != 0:
            return clean_rc
        cmd = list(spec.build_cmd)

    print(f"[build] {' '.join(cmd)} (cwd={spec.workdir})")
    return subprocess.run(cmd, cwd=spec.workdir).returncode


def _print_solution_summary(solution) -> None:
    try:
        gap = solution.gap if hasattr(solution, "gap") else None
    except Exception:
        gap = None

    print("Instance:", solution.instance)
    print("Solver:", solution.solver)
    print("Objective:", solution.objective)
    print("Lower bound:", solution.lower_bound)
    if gap is not None:
        print("Gap (%):", gap)
    print("Time (s):", solution.time)


# Module registry ------------------------------------------------------------

def run_ga(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for ga")
    # Use default config if not specified
    config_path = args.config if args.config else str(spec.default_config)
    config = _load_config(config_path, spec.default_config)
    instance = args.instance or config.get("instance") or config.get("input_file")
    if not instance:
        raise ValueError("GA requires --instance or instance/input_file in config")
    cmd = [str(spec.binary), str(_resolve_path(instance))]
    # Always pass config file (default or specified)
    cmd += ["--config", str(_resolve_path(config_path))]
    # Output file
    if args.output:
        cmd += ["--output", str(_resolve_path(args.output))]
    # Seed
    seed = config.get("seed")
    if seed is not None:
        cmd += ["--seed", str(seed)]
    # Log
    if args.log or config.get("log_output", False):
        cmd.append("--log")
    print(f"[run] {' '.join(cmd)} (cwd={spec.workdir})")
    result = subprocess.run(cmd, cwd=spec.workdir)
    return result.returncode

MODULES: Dict[str, ModuleSpec] = {
    "gilmore_lawler": ModuleSpec(
        name="gilmore_lawler",
        kind="cpp",
        workdir=ROOT / "cpp/modules/gilmore_lawler",
        binary=ROOT / "cpp/modules/gilmore_lawler/gilmore_lawler_solver",
        build_cmd=["make"],
        default_config=ROOT / "configs/gilmore_lawler.json",
        runner=run_gilmore_lawler,
    ),
    "ga": ModuleSpec(
        name="ga",
        kind="cpp",
        workdir=ROOT / "cpp/modules/ga",
        binary=ROOT / "cpp/modules/ga/ga_solver",
        build_cmd=["make"],
        default_config=ROOT / "configs/ga.json",
        runner=run_ga,
    ),
    "vns": ModuleSpec(
        name="vns",
        kind="cpp",
        workdir=ROOT / "cpp/modules/vns",
        binary=ROOT / "cpp/modules/vns/vns_solver",
        build_cmd=["make"],
        default_config=ROOT / "configs/vns.json",
        runner=run_vns,
    ),
    "rlt1_gurobi_reduction_IV_cpp": ModuleSpec(
        name="rlt1_gurobi_reduction_IV_cpp",
        kind="cpp",
        workdir=ROOT / "cpp/modules/rlt1_gurobi_reduction_IV",
        binary=ROOT / "cpp/modules/rlt1_gurobi_reduction_IV/rlt1_gurobi_reduction_IV_solver",
        build_cmd=["make"],
        default_config=ROOT / "configs/rlt1_gurobi_reduction_IV_cpp.json",
        runner=run_rlt1_gurobi_reduction_IV_cpp,
    ),
    "sfd_scip": ModuleSpec(
        name="sfd_scip",
        kind="cpp",
        workdir=ROOT / "cpp/modules/sfd_scip",
        binary=ROOT / "cpp/modules/sfd_scip/sfd_solver",
        build_cmd=["make"],
        default_config=ROOT / "configs/sfd_scip.json",
        runner=run_sfd_scip,
    ),
    "sfd_volume": ModuleSpec(
        name="sfd_volume",
        kind="cpp",
        workdir=ROOT / "cpp/modules/sfd_volume",
        binary=ROOT / "cpp/modules/sfd_volume/sfd_volume_solver",
        build_cmd=["make"],
        default_config=ROOT / "configs/sfd_volume.json",
        runner=run_sfd_volume,
    ),
    "rlt1_scip": ModuleSpec(
        name="rlt1_scip",
        kind="cpp",
        workdir=ROOT / "cpp/modules/rlt1_scip",
        binary=ROOT / "cpp/modules/rlt1_scip/rlt1_solver",
        build_cmd=["make"],
        default_config=ROOT / "configs/rlt1_scip.json",
        runner=run_rlt1_scip_cpp,
    ),
    "rlt1_volume": ModuleSpec(
        name="rlt1_volume",
        kind="cpp",
        workdir=ROOT / "cpp/modules/rlt1_volume",
        binary=ROOT / "cpp/modules/rlt1_volume/rlt1_volume_solver",
        build_cmd=["make"],
        default_config=ROOT / "configs/rlt1_volume.json",
        runner=run_rlt1_volume,
    ),
    "rlt1_bundle": ModuleSpec(
        name="rlt1_bundle",
        kind="cpp",
        workdir=ROOT / "cpp/modules/rlt1_bundle",
        binary=ROOT / "cpp/modules/rlt1_bundle/rlt1_bundle_solver",
        build_cmd=["make"],
        default_config=ROOT / "configs/rlt1_bundle.json",
        runner=run_rlt1_bundle,
    ),
    "rlt1_cg": ModuleSpec(
        name="rlt1_cg",
        kind="cpp",
        workdir=ROOT / "cpp/modules/rlt1_cg",
        binary=ROOT / "cpp/modules/rlt1_cg/rlt1_cg",
        build_cmd=["make"],
        default_config=ROOT / "configs/rlt1_cg.json",
        runner=run_RLT1_cg,
    ),
    "rlt1_cg_gurobi": ModuleSpec(
        name="rlt1_cg_gurobi",
        kind="cpp",
        workdir=ROOT / "cpp/modules/rlt1_cg_gurobi",
        binary=ROOT / "cpp/modules/rlt1_cg_gurobi/rlt1_cg",
        build_cmd=["make"],
        default_config=ROOT / "configs/rlt1_cg_gurobi.json",
        runner=run_RLT1_cg_gurobi,
    ),
    "m4_volume": ModuleSpec(
        name="m4_volume",
        kind="cpp",
        workdir=ROOT / "cpp/modules/m4_volume",
        binary=ROOT / "cpp/modules/m4_volume/m4_volume_solver",
        build_cmd=["make"],
        default_config=ROOT / "configs/m4_volume.json",
        runner=run_m4_volume,
    ),
    "rlt1_cplex": ModuleSpec(
        name="rlt1_cplex",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/rlt1_cplex.json",
        runner=run_rlt1_cplex,
    ),
    "rlt1_cplex_reduction_IV": ModuleSpec(
        name="rlt1_cplex_reduction_IV",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/rlt1_cplex_reduction_IV.json",
        runner=run_rlt1_cplex_reduction_IV,
    ),
    "rlt1_gurobi": ModuleSpec(
        name="rlt1_gurobi",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/rlt1_gurobi.json",
        runner=run_rlt1_gurobi,
    ),
    "rlt1_gurobi_cpp": ModuleSpec(
        name="rlt1_gurobi_cpp",
        kind="cpp",
        workdir=ROOT / "cpp/modules/rlt1_gurobi",
        binary=ROOT / "cpp/modules/rlt1_gurobi/rlt1_gurobi_solver",
        build_cmd=["make"],
        default_config=ROOT / "configs/rlt1_gurobi_cpp.json",
        runner=run_rlt1_gurobi_cpp,
    ),
    "rlt1_gurobi_reduction_IV": ModuleSpec(
        name="rlt1_gurobi_reduction_IV",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/rlt1_gurobi_reduction_IV.json",
        runner=run_rlt1_gurobi_reduction_IV,
    ),
    "rlt1_gurobi_reduction_IV_cpp_2": ModuleSpec(
        name="rlt1_gurobi_reduction_IV_cpp_2",
        kind="cpp",
        workdir=ROOT / "cpp/modules/rlt1_gurobi_reduction_IV_2",
        binary=ROOT / "cpp/modules/rlt1_gurobi_reduction_IV_2/rlt1_gurobi_reduction_IV_solver",
        build_cmd=["make"],
        default_config=ROOT / "configs/rlt1_gurobi_reduction_IV_cpp_2.json",
        runner=run_rlt1_gurobi_reduction_IV_cpp_2,
    ),
    "rlt1_sdp": ModuleSpec(
        name="rlt1_sdp",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/rlt1_sdp.json",
        runner=run_rlt1_sdp,
    ),
    "m5_cplex": ModuleSpec(
        name="m5_cplex",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/m5_cplex.json",
        runner=run_m5_cplex,
    ),
    "cg_cplex": ModuleSpec(
        name="cg_cplex",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/cg_cplex.json",
        runner=run_cg_cplex,
    ),
    "cg_cplex_cpp": ModuleSpec(
        name="cg_cplex_cpp",
        kind="cpp",
        workdir=ROOT / "cpp/modules/cg_cplex",
        binary=ROOT / "cpp/modules/cg_cplex/cg_cplex",
        build_cmd=["make"],
        default_config=ROOT / "configs/cg_cplex.json",
        runner=run_cg_cplex_cpp,
    ),
    "sfdcg_cplex": ModuleSpec(
        name="sfdcg_cplex",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/sfdcg_cplex.json",
        runner=run_sfdcg_cplex,
    ),
    "sfdcg_cplex_cpp": ModuleSpec(
        name="sfdcg_cplex_cpp",
        kind="cpp",
        workdir=ROOT / "cpp/modules/sfdcg_cplex",
        binary=ROOT / "cpp/modules/sfdcg_cplex/sfdcg_cplex",
        build_cmd=["make"],
        default_config=ROOT / "configs/sfdcg_cplex.json",
        runner=run_sfdcg_cplex_cpp,
    ),
    "qaoa_qiskit": ModuleSpec(
        name="qaoa_qiskit",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/qaoa_qiskit.json",
        runner=run_qaoa_qiskit,
    ),
    "volume_qaoa_qiskit": ModuleSpec(
        name="volume_qaoa_qiskit",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/volume_qaoa_qiskit.json",
        runner=run_volume_qaoa_qiskit,
    ),
    "m4_cplex": ModuleSpec(
        name="m4_cplex",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/m4.json",
        runner=run_m4_cplex,
    ),
    "rlt1_scip_py": ModuleSpec(
        name="rlt1_scip_py",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/rlt1_scip.json",
        runner=run_rlt1_scip_py,
    ),
    "sfd_cplex": ModuleSpec(
        name="sfd_cplex",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/sfd.json",
        runner=run_sfd_cplex,
    ),
    "sfd_gurobi": ModuleSpec(
        name="sfd_gurobi",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/sfd_gurobi.json",
        runner=run_sfd_gurobi,
    ),
    "sfd_gurobi_cpp": ModuleSpec(
        name="sfd_gurobi_cpp",
        kind="cpp",
        workdir=ROOT / "cpp/modules/sfd_gurobi",
        binary=ROOT / "cpp/modules/sfd_gurobi/sfd_gurobi_solver",
        build_cmd=["make"],
        default_config=ROOT / "configs/sfd_gurobi_cpp.json",
        runner=run_sfd_gurobi_cpp,
    ),
    "sfd_sdp": ModuleSpec(
        name="sfd_sdp",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/sfd_sdp.json",
        runner=run_sfd_sdp,
    ),
    "sfb_gurobi": ModuleSpec(
        name="sfb_gurobi",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/sfb_gurobi.json",
        runner=run_sfb_gurobi,
    ),
    "KBXY_gurobi_cpp": ModuleSpec(
        name="KBXY_gurobi_cpp",
        kind="cpp",
        workdir=ROOT / "cpp/modules/KBXY_gurobi",
        binary=ROOT / "cpp/modules/KBXY_gurobi/KBXY_gurobi_solver",
        build_cmd=["make"],
        default_config=ROOT / "configs/KBXY_gurobi_cpp.json",
        runner=run_KBXY_gurobi_cpp,
    ),
    "local_search": ModuleSpec(
        name="local_search",
        kind="cpp",
        workdir=ROOT / "cpp/modules/local_search",
        binary=ROOT / "cpp/modules/local_search/local_search_solver",
        build_cmd=["make"],
        default_config=ROOT / "configs/local_search.json",
        runner=run_local_search,
    ),
    "local_search_modify": ModuleSpec(
        name="local_search_modify",
        kind="cpp",
        workdir=ROOT / "cpp/modules/local_search_modify",
        binary=ROOT / "cpp/modules/local_search_modify/local_search_solver",
        build_cmd=["make"],
        default_config=ROOT / "configs/local_search_modify.json",
        runner=run_local_search_modify,
    ),
    "local_search_python": ModuleSpec(
        name="local_search_python",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/local_search_python.json",
        runner=run_local_search_python,
    ),
    "m4_formulation": ModuleSpec(
        name="m4_formulation",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/m4_formulation.json",
        runner=lambda args, spec: run_m4_formulation(args, spec),
    ),
    "qubo_volume": ModuleSpec(
        name="qubo_volume",
        kind="cpp",
        workdir=ROOT / "cpp/modules/qubo_volume",
        binary=ROOT / "cpp/modules/qubo_volume/qubo_volume_solver",
        build_cmd=["make"],
        default_config=ROOT / "configs/qubo_volume.json",
        runner=run_qubo_volume,
    ),
    "qap_cplex": ModuleSpec(
        name="qap_cplex",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/qap_cplex.json",
        runner=run_qap_cplex,  # Reuse qap_cplex runner since it's a similar CPLEX-based solver
    ),
    "qap_gurobi": ModuleSpec(
        name="qap_gurobi",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/qap_gurobi.json",
        runner=run_qap_gurobi,  # Reuse qap_gurobi runner since it's a similar Gurobi-based solver
    ),
    "cg_va": ModuleSpec(
        name="cg_va",
        kind="cpp",
        workdir=ROOT / "cpp/modules/cg_va",
        binary=ROOT / "cpp/modules/cg_va/cg_va",
        build_cmd=["make"],
        default_config=ROOT / "configs/cg_va.json",
        runner=run_cg_va,
    ),
    "m5_cg_cplex_cpp": ModuleSpec(
        name="m5_cg_cplex_cpp",
        kind="cpp",
        workdir=ROOT / "cpp/modules/m5_cg_cplex",
        binary=ROOT / "cpp/modules/m5_cg_cplex/m5_cg_cplex",
        build_cmd=["make"],
        default_config=ROOT / "configs/m5_cg_cplex.json",
        runner=run_m5_cg_cplex_cpp,
    ),
    "admm": ModuleSpec(
        name="admm",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/admm.json",
        runner=run_admm,
    ),
    "csdp": ModuleSpec(
        name="c_sdp_admm",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/csdp.json",
        runner=run_csdp,
    ),
    "csdp_cvxpy": ModuleSpec(
        name="c_sdp_cvxpy",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/csdp_cvxpy.json",
        runner=run_csdp_cvxpy,
    ),
    "usbs": ModuleSpec(
        name="usbs",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/usbs.json",
        runner=run_usbs,
    ),
    "sdp_pytorch": ModuleSpec(
        name="sdp_pytorch",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/sdp_pytorch.json",
        runner=run_sdp_pytorch,
    ),
    "sdp_bundle": ModuleSpec(
        name="sdp_bundle",
        kind="cpp",
        workdir=ROOT / "cpp/modules/sdp_bundle",
        binary=ROOT / "cpp/modules/sdp_bundle/sdp_bundle_solver",
        build_cmd=["make"],
        default_config=ROOT / "configs/sdp_bundle.json",
        runner=run_sdp_bundle,
    ),
    "sdp_volume": ModuleSpec(
        name="sdp_volume",
        kind="cpp",
        workdir=ROOT / "cpp/modules/sdp_volume",
        binary=ROOT / "cpp/modules/sdp_volume/sdp_volume_solver",
        build_cmd=["make"],
        default_config=ROOT / "configs/sdp_volume.json",
        runner=run_sdp_volume,
    ),
    "sdp_volume_2": ModuleSpec(
        name="sdp_volume_2",
        kind="cpp",
        workdir=ROOT / "cpp/modules/sdp_volume_2",
        binary=ROOT / "cpp/modules/sdp_volume_2/sdp_volume_solver",
        build_cmd=["make"],
        default_config=ROOT / "configs/sdp_volume_2.json",
        runner=run_sdp_volume_2,
    ),

}

# CLI ------------------------------------------------------------------------

def build_command(subparsers) -> None:
    parser = subparsers.add_parser("build", help="Build a module")
    parser.add_argument(
        "--module",
        "-m",
        nargs="+",
        choices=MODULES.keys(),
        help="Modules to build (default: all C++ modules)",
    )
    parser.add_argument("--clean", action="store_true", help="Run clean before build when supported")


def run_command(subparsers) -> None:
    parser = subparsers.add_parser("run", help="Run a module")
    parser.add_argument("--module", "-m", required=True, choices=MODULES.keys())
    parser.add_argument("--instance", help="QAP instance file (.dat)")
    parser.add_argument("--config", help="Config JSON path")
    parser.add_argument("--warmstart", help="Warm-start solution file (.sln)")
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument("--time-limit", type=int, dest="time_limit", help="Time limit seconds (where supported)")
    parser.add_argument("--threads", type=int, help="Thread count (where supported)")
    parser.add_argument("--decomposition", choices=["value_layer", "value_only"], help="SFD decomposition type")
    parser.add_argument("--relax", action="store_true", help="Use relaxed constraints where supported")
    parser.add_argument("--log", action="store_true", help="Enable solver logging")
    parser.add_argument("--build-first", action="store_true", help="Build before running")
    parser.add_argument("--fixed", help="Fixed variables file (module-specific)")
    parser.add_argument("--formulation", help="Formulation to use (module-specific)")


def parse_args() -> "Args":
    parser = argparse.ArgumentParser(description="Unified interface for QAP-Solver modules")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_command(subparsers)
    run_command(subparsers)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.command == "build":
        targets = args.module or [name for name, spec in MODULES.items() if spec.kind == "cpp"]
        for name in targets:
            spec = MODULES[name]
            rc = _build_module(spec, clean=args.clean)
            if rc != 0:
                print(f"[error] build failed for {name}")
                return rc
        return 0

    # run
    spec = MODULES[args.module]
    if args.build_first:
        rc = _build_module(spec)
        if rc != 0:
            print(f"[error] build failed for {spec.name}")
            return rc

    if spec.runner is None:
        raise ValueError(f"No runner configured for module {spec.name}")

    return spec.runner(args, spec)


if __name__ == "__main__":
    sys.exit(main())
