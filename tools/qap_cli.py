
#!/usr/bin/env python3
"""Unified build/run interface for QAP-Solver modules.

Supports:
- C++ binaries: sfd_scip, rtl1_scip, local_search (2-opt, ils, tabu)
- Python solvers: rtl1_cplex, rtl1_scip_py

Examples:
    # Build C++ modules
    python tools/qap_cli.py build --module sfd_scip rtl1_scip local_search

    # Run Local Search (tabu search) with config
    python tools/qap_cli.py run --module local_search --config configs/local_search.json

    # Run SFD (SCIP) with config
    python tools/qap_cli.py run --module sfd_scip --config configs/sfd_scip.json

    # Run RTL1 (SCIP C++) with instance and warmstart
    python tools/qap_cli.py run --module rtl1_scip --instance data/chr12a.dat --warmstart data/chr12a.sln --threads 8 --time-limit 300

    # Run RTL1 (CPLEX Python) using config and write output JSON
    python tools/qap_cli.py run --module rtl1_cplex --instance data/chr12a.dat --config configs/rtl1_cplex.json --output results.json
"""

from __future__ import annotations

import argparse
import json
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


def run_rtl1_scip_cpp(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for rtl1_scip")
    
    # Load config if provided
    config = _load_config(args.config, spec.default_config)
    
    # Determine instance path (CLI arg > config > error)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("RTL1 SCIP (C++) requires --instance or instance in config")

    cmd = [str(spec.binary), str(_resolve_path(instance))]
    
    # Warmstart: CLI > config
    warmstart = args.warmstart or config.get("warmstart")
    if warmstart:
        cmd += ["--warmstart", str(_resolve_path(warmstart))]
    
    if args.output:
        cmd += ["--output", str(_resolve_path(args.output))]
    
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

def run_sfd_volume(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for sfd_volume")
    # Load config if provided
    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("RTL1 Volume requires --instance or instance in config")
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

def run_rtl1_volume(args: "Args", spec: ModuleSpec) -> int:
    if not spec.binary:
        raise ValueError("Missing binary path for rtl1_volume")
    
    # Load config if provided
    config = _load_config(args.config, spec.default_config)
    
    # Determine instance path (CLI arg > config > error)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("RTL1 Volume requires --instance or instance in config")

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
        
        
    formulation = config.get("formulation", "")
    if formulation:
        cmd += ["--formulation", str(formulation)]

    # Fixed variables (file) from CLI or config key "fixed"
    fixed_path = args.fixed or config.get("fixed")
    if fixed_path:
        cmd += ["--fixed", str(_resolve_path(fixed_path))]

    return _run_subprocess(cmd, spec.workdir)


def run_rtl1_cplex(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.rtl1_cplex.solver import RTL1CPLEXSolver
    except ImportError as exc:
        print("RTL1 CPLEX solver missing dependencies (docplex/cplex)", file=sys.stderr)
        raise exc

    config = _load_config(args.config, spec.default_config)
    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("RTL1 CPLEX requires --instance")
    solver = RTL1CPLEXSolver(config)

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


def run_rtl1_scip_py(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.rtl1_scip.solver import RTL1SCIPSolver
    except ImportError as exc:
        print("RTL1 SCIP Python solver missing dependencies (PySCIPOpt)", file=sys.stderr)
        raise exc

    instance = args.instance or config.get("instance")
    if not instance:
        raise ValueError("RTL1 SCIP Python requires --instance")

    config = _load_config(args.config, spec.default_config)
    solver = RTL1SCIPSolver(config)
    solution = solver.solve_instance(
        instance_path=str(_resolve_path(instance)),
        output_path=str(_resolve_path(args.output)) if args.output else None,
        warmstart_path=str(_resolve_path(args.warmstart)) if args.warmstart else None,
    )

    _print_solution_summary(solution)
    return 0


def run_sfd_cplex(args: "Args", spec: ModuleSpec) -> int:
    # The SFD CPLEX implementation requires subgraph decomposition logic that
    # is not wired in this repository. Provide a clear error until integrated.
    raise NotImplementedError(
        "SFD CPLEX runner not yet integrated: subgraph decomposition pipeline is missing."
    )


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
    "rtl1_scip": ModuleSpec(
        name="rtl1_scip",
        kind="cpp",
        workdir=ROOT / "cpp/modules/rtl1_scip",
        binary=ROOT / "cpp/modules/rtl1_scip/rtl1_solver",
        build_cmd=["make"],
        default_config=ROOT / "configs/rtl1_scip.json",
        runner=run_rtl1_scip_cpp,
    ),
    "rtl1_volume": ModuleSpec(
        name="rtl1_volume",
        kind="cpp",
        workdir=ROOT / "cpp/modules/rtl1_volume",
        binary=ROOT / "cpp/modules/rtl1_volume/rtl1_volume_solver",
        build_cmd=["make"],
        default_config=ROOT / "configs/rtl1_volume.json",
        runner=run_rtl1_volume,
    ),
    "rtl1_cplex": ModuleSpec(
        name="rtl1_cplex",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/rtl1_cplex.json",
        runner=run_rtl1_cplex,
    ),
    "rtl1_scip_py": ModuleSpec(
        name="rtl1_scip_py",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/rtl1_scip.json",
        runner=run_rtl1_scip_py,
    ),
    "sfd_cplex": ModuleSpec(
        name="sfd_cplex",
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "configs/sfd.json",
        runner=run_sfd_cplex,
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
