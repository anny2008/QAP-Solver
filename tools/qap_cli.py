#!/usr/bin/env python3
"""Unified build/run interface for QAP-Solver modules.

Supports:
- C++ binaries: sfd_scip, rtl1_scip
- Python solvers: rtl1_cplex, rtl1_scip_py

Examples:
  # Build C++ modules
  python tools/qap_cli.py build --module sfd_scip rtl1_scip

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
from typing import Callable, Dict, Optional

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
    if not args.instance:
        raise ValueError("RTL1 SCIP (C++) requires --instance")

    cmd = [str(spec.binary), str(_resolve_path(args.instance))]
    if args.warmstart:
        cmd += ["--warmstart", str(_resolve_path(args.warmstart))]
    if args.output:
        cmd += ["--output", str(_resolve_path(args.output))]
    if args.time_limit is not None:
        cmd += ["--time", str(args.time_limit)]
    if args.threads is not None:
        cmd += ["--threads", str(args.threads)]
    if args.log:
        cmd.append("--log")

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

    return _run_subprocess(cmd, spec.workdir)


def run_rtl1_cplex(args: "Args", spec: ModuleSpec) -> int:
    try:
        from qap.modules.rtl1_cplex.solver import RTL1CPLEXSolver
    except ImportError as exc:
        print("RTL1 CPLEX solver missing dependencies (docplex/cplex)", file=sys.stderr)
        raise exc

    if not args.instance:
        raise ValueError("RTL1 CPLEX requires --instance")

    config = _load_config(args.config, spec.default_config)
    solver = RTL1CPLEXSolver(config)

    # Warm-start: prefer CLI arg, else use config key 'warmstart' if provided
    warmstart_path = None
    if args.warmstart:
        warmstart_path = str(_resolve_path(args.warmstart))
    elif "warmstart" in config and config["warmstart"]:
        warmstart_path = str(_resolve_path(config["warmstart"]))

    solution = solver.solve_instance(
        instance_path=str(_resolve_path(args.instance)),
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

    if not args.instance:
        raise ValueError("RTL1 SCIP Python requires --instance")

    config = _load_config(args.config, spec.default_config)
    solver = RTL1SCIPSolver(config)
    solution = solver.solve_instance(
        instance_path=str(_resolve_path(args.instance)),
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
    # Placeholder: local search heuristics are not implemented in this repo snapshot.
    raise NotImplementedError(
        "Local search runner not available: heuristic implementation is absent."
    )


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

MODULES: Dict[str, ModuleSpec] = {
    "sfd_scip": ModuleSpec(
        name="sfd_scip",
        kind="cpp",
        workdir=ROOT / "cpp/modules/sfd_scip",
        binary=ROOT / "cpp/modules/sfd_scip/sfd_solver",
        build_cmd=["make"],
        default_config=ROOT / "configs/sfd_scip.json",
        runner=run_sfd_scip,
    ),
    "rtl1_scip": ModuleSpec(
        name="rtl1_scip",
        kind="cpp",
        workdir=ROOT / "cpp/modules/rtl1_scip",
        binary=ROOT / "cpp/modules/rtl1_scip/rtl1_solver",
        build_cmd=["make"],
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
        kind="python",
        workdir=ROOT,
        default_config=ROOT / "python/qap/modules/local_search/example_config.json",
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
