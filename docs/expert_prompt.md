# Expert Prompt — Metaheuristic VLSI Circuit Partitioning Framework

## Context

This repository implements and benchmarks eight bio-inspired metaheuristic
algorithms for VLSI circuit hypergraph partitioning, aligned to the
OpenROAD/TritonPart interface so results are directly comparable with the
industrial baseline.

**Algorithms:** ABC · KH · MBO · EWA · EHO · MS · SMA · HHO

**Benchmark families:**
- ISPD98 — 18 hypergraph partitioning circuits (`.hgr`)
- ISCAS'85 — 11 combinational circuits (`c17` … `c7552`, `.bench`)
- ISCAS'89 — 30 sequential circuits (`s27` … `s38584`, `.bench` / `.v`)

---

## OpenROAD / TritonPart alignment

| Framework field | Meaning | OpenROAD concept |
|---|---|---|
| `num_parts` | k-way partition count | `-num_parts` |
| `balance_constraint` | allowed imbalance | `-balance_constraint` |
| `hypergraph_file` | HGR input | `-hypergraph_file` |
| `placement_file` | placement embedding | `-placement_file` |
| `timing_aware` | timing-driven flag | `-timing_aware_flag` |
| `solution_file` | partition labels | `-solution_file` |
| `global_net_threshold` | ignore large nets | `-global_net_threshold` |
| `evaluate()` | cut/balance check | `evaluate_hypergraph_solution` |

TritonPart uses multilevel coarsening → ILP-seeded initial partitioning
→ FM-style refinement → COCP → V-cycle refinement.
The metaheuristics operate directly on the flat hypergraph and are therefore
most effective on small-to-medium circuits; for very large circuits
(s35932, s38584) consider coarsening as a preprocessing step.

---

## Objective function

```
F = w_c · cutsize
  + w_b · max(0, imbalance − balance_constraint)
  + w_t · timingPenalty
  + w_p · placementPenalty
```

Default weights: `w_c=1.0`, `w_b=10.0`, `w_t=0.0`, `w_p=0.0`.
Set `w_t > 0` only with `--timing-aware`; set `w_p > 0` only with
`--placement-file`.

---

## Quick start

```bash
# 1. Build Docker image
docker build -t vlsi-partition -f docker/Dockerfile .

# 2. Fetch benchmarks
docker run --rm -v $(pwd)/benchmarks:/app/benchmarks vlsi-partition \
    bash scripts/fetch_benchmarks.sh

# 3. Run all algorithms on all downloaded benchmarks
docker run --rm \
    -v $(pwd)/benchmarks:/app/benchmarks \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/docs:/app/docs \
    vlsi-partition \
    bash scripts/run_all.sh --k 2 --seeds "0 1 2" --iter 200

# 4. Single run (example)
docker run --rm \
    -v $(pwd)/benchmarks:/app/benchmarks \
    -v $(pwd)/results:/app/results \
    vlsi-partition \
    python src/main.py \
        --benchmark benchmarks/circuits/c432.bench \
        --algorithm ABC HHO SMA \
        --num-parts 2 --balance-constraint 0.1 \
        --seeds 0 1 2 --max-iter 300 --verbose
```

---

## Result schema (`results/cut_results_template.csv`)

| Column | Description |
|---|---|
| `benchmark` | Circuit name |
| `family` | ISPD98 / ISCAS85 / ISCAS89 |
| `algorithm` | ABC / KH / MBO / EWA / EHO / MS / SMA / HHO |
| `num_parts` | k-way partition count |
| `balance_constraint` | allowed imbalance |
| `seed` | random seed used |
| `cutsize` | hyperedge cut metric |
| `runtime_sec` | wall-clock time |
| `feasible` | balance constraint satisfied? |
| `notes` | free text |

---

## Figure generation rules

All figures must:
- Have a **white background** (`facecolor="white"`).
- Contain **no text overlap** (use `tight_layout` and rotated x-labels).
- Use **orthogonal arrows** in pipeline/architecture diagrams.
- Export as **SVG first**, then PNG for manuscripts.

```python
from reporting import plot_convergence, plot_cut_comparison

plot_convergence(
    {"ABC": abc_hist, "HHO": hho_hist},
    title="c432 — k=2 convergence",
    output_path="docs/figures/convergence_c432.svg",
)

plot_cut_comparison(
    {"c432": {"ABC": 12.0, "HHO": 9.5}},
    title="ISCAS85 cut comparison",
    output_path="docs/figures/cut_comparison_iscas85.svg",
)
```

---

## Adding a new algorithm

1. Create `src/optimizers/my_algo.py` with a class inheriting `BaseOptimizer`.
2. Implement `optimize(self, hg, cfg, seed) -> (partition, history)`.
3. Register it in `src/optimizers/__init__.py`:
   ```python
   from .my_algo import MyAlgoOptimizer
   REGISTRY["MYALGO"] = MyAlgoOptimizer
   ```
4. It is now selectable via `--algorithm MYALGO`.

---

## Repository layout

```
vlsi-partition-framework/
├── docker/Dockerfile
├── configs/default.yaml
├── benchmarks/
│   ├── benchmark_manifest.csv
│   └── circuits/          ← populated by fetch_benchmarks.sh
├── docs/
│   ├── figures/
│   └── expert_prompt.md
├── results/
│   └── cut_results_template.csv
├── scripts/
│   ├── fetch_benchmarks.sh
│   └── run_all.sh
└── src/
    ├── main.py
    ├── objective.py
    ├── benchmark_io.py
    ├── openroad_adapter.py
    ├── reporting.py
    └── optimizers/
        ├── __init__.py
        ├── base.py
        ├── abc.py  kh.py  mbo.py  ewa.py
        └── eho.py  ms.py  sma.py  hho.py
```
