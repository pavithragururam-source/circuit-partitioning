"""
OpenROAD / TritonPart adapter.

Mirrors the OpenROAD TCL command interface so the framework can
  (a) invoke TritonPart as a subprocess baseline, and
  (b) parse its output into the same result dict as the metaheuristics.

OpenROAD commands modelled here:
  triton_part_hypergraph        – partition a .hgr file
  evaluate_hypergraph_solution  – score an existing solution file
  triton_part_design            – partition the placed design (requires DB)
  evaluate_part_design_solution – score a design-level solution
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from objective import Hypergraph, PartitionConfig, evaluate


# ---------------------------------------------------------------------------
# TCL command builders
# ---------------------------------------------------------------------------

def _triton_part_hypergraph_tcl(
    hypergraph_file: str,
    num_parts: int,
    balance_constraint: float,
    seed: int = 0,
    solution_file: str = "solution.part",
    timing_aware: bool = False,
    placement_file: str = "",
    global_net_threshold: int = 1000,
) -> str:
    """Return a TCL snippet that calls triton_part_hypergraph."""
    lines = [
        "triton_part_hypergraph \\",
        f"  -hypergraph_file {hypergraph_file} \\",
        f"  -num_parts {num_parts} \\",
        f"  -balance_constraint {balance_constraint} \\",
        f"  -seed {seed} \\",
        f"  -solution_file {solution_file} \\",
        f"  -global_net_threshold {global_net_threshold}",
    ]
    if timing_aware:
        lines[-1] += " \\"
        lines.append("  -timing_aware_flag 1")
    if placement_file:
        lines[-1] += " \\"
        lines.append(f"  -placement_file {placement_file}")
    return "\n".join(lines)


def _evaluate_hypergraph_solution_tcl(
    hypergraph_file: str,
    solution_file: str,
    num_parts: int,
    balance_constraint: float,
) -> str:
    return (
        f"evaluate_hypergraph_solution "
        f"-hypergraph_file {hypergraph_file} "
        f"-solution_file {solution_file} "
        f"-num_parts {num_parts} "
        f"-balance_constraint {balance_constraint}"
    )


# ---------------------------------------------------------------------------
# Subprocess invocation (requires OpenROAD in PATH)
# ---------------------------------------------------------------------------

class OpenROADAdapter:
    """
    Thin wrapper around the OpenROAD executable for TritonPart baseline runs.

    If openroad_bin is not found the adapter falls back to a pure-Python
    evaluation using the framework's own objective.evaluate().
    """

    def __init__(self, openroad_bin: str = "openroad"):
        self.openroad_bin = openroad_bin
        self._available: Optional[bool] = None

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        if self._available is None:
            try:
                result = subprocess.run(
                    [self.openroad_bin, "-version"],
                    capture_output=True, timeout=10,
                )
                self._available = result.returncode == 0
            except (FileNotFoundError, subprocess.TimeoutExpired):
                self._available = False
        return self._available

    # ------------------------------------------------------------------
    # Partition via TritonPart
    # ------------------------------------------------------------------

    def triton_part_hypergraph(
        self,
        hgr_file: str,
        cfg: PartitionConfig,
        seed: int = 0,
    ) -> Dict:
        """
        Run TritonPart on an .hgr file. Returns a result dict compatible
        with the metaheuristic outputs.

        If OpenROAD is unavailable, raises RuntimeError.
        """
        if not self.is_available():
            raise RuntimeError(
                f"OpenROAD binary '{self.openroad_bin}' not found. "
                "Install OpenROAD or use a metaheuristic optimizer instead."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            sol_file = os.path.join(tmpdir, "solution.part")
            tcl_script = os.path.join(tmpdir, "run.tcl")

            tcl = _triton_part_hypergraph_tcl(
                hypergraph_file=hgr_file,
                num_parts=cfg.num_parts,
                balance_constraint=cfg.balance_constraint,
                seed=seed,
                solution_file=sol_file,
                timing_aware=cfg.timing_aware,
                global_net_threshold=cfg.global_net_threshold,
            ) + "\nexit\n"

            Path(tcl_script).write_text(tcl)

            result = subprocess.run(
                [self.openroad_bin, tcl_script],
                capture_output=True, text=True, timeout=600,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"OpenROAD exited with code {result.returncode}:\n{result.stderr}"
                )

            partition = _read_solution_file(sol_file)
            cutsize, feasible = _parse_openroad_stdout(result.stdout)

        return {
            "partition": partition,
            "cutsize": cutsize,
            "feasible": feasible,
            "stdout": result.stdout,
            "source": "TritonPart",
        }

    # ------------------------------------------------------------------
    # Pure-Python fallback evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_solution(
        hg: Hypergraph,
        partition: np.ndarray,
        cfg: PartitionConfig,
    ) -> Dict:
        """
        Python-only equivalent of evaluate_hypergraph_solution.
        Always available regardless of OpenROAD installation.
        """
        return evaluate(hg, partition, cfg)


# ---------------------------------------------------------------------------
# Solution file I/O
# ---------------------------------------------------------------------------

def write_solution_file(partition: np.ndarray, path: str) -> None:
    """Write a partition to a TritonPart-compatible solution file (one label per line)."""
    Path(path).write_text("\n".join(str(int(p)) for p in partition) + "\n")


def read_solution_file(path: str) -> np.ndarray:
    return _read_solution_file(path)


def _read_solution_file(path: str) -> np.ndarray:
    text = Path(path).read_text()
    labels = [int(l.strip()) for l in text.splitlines() if l.strip()]
    return np.array(labels, dtype=int)


# ---------------------------------------------------------------------------
# Output parsing helpers
# ---------------------------------------------------------------------------

def _parse_openroad_stdout(stdout: str) -> tuple:
    """Extract cutsize and feasibility from TritonPart stdout."""
    cutsize = None
    feasible = False

    m_cut = re.search(r"cutsize\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)", stdout, re.IGNORECASE)
    if m_cut:
        cutsize = float(m_cut.group(1))

    if re.search(r"partition is (feasible|balanced)", stdout, re.IGNORECASE):
        feasible = True

    return cutsize, feasible


# ---------------------------------------------------------------------------
# Convenience: generate the TCL script for documentation / batch runs
# ---------------------------------------------------------------------------

def generate_tcl_script(
    hgr_file: str,
    cfg: PartitionConfig,
    seed: int = 0,
    solution_file: str = "solution.part",
    output_path: Optional[str] = None,
) -> str:
    """
    Return (and optionally write) the TCL script that reproduces a run
    in an actual OpenROAD environment.
    """
    tcl = _triton_part_hypergraph_tcl(
        hypergraph_file=hgr_file,
        num_parts=cfg.num_parts,
        balance_constraint=cfg.balance_constraint,
        seed=seed,
        solution_file=solution_file,
        timing_aware=cfg.timing_aware,
        global_net_threshold=cfg.global_net_threshold,
    )
    eval_tcl = _evaluate_hypergraph_solution_tcl(
        hypergraph_file=hgr_file,
        solution_file=solution_file,
        num_parts=cfg.num_parts,
        balance_constraint=cfg.balance_constraint,
    )
    full_script = tcl + "\n" + eval_tcl + "\nexit\n"

    if output_path:
        Path(output_path).write_text(full_script)

    return full_script
