# Metaheuristic VLSI Circuit Partitioning Framework

A Docker-ready, OpenROAD/TritonPart-aligned research framework for benchmarking
eight bio-inspired metaheuristic algorithms on ISPD98 and ISCAS benchmark families,
with expert-grade EDA modules: FM local refinement, Static Timing Analysis, and
Heavy Edge Matching multilevel coarsening.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Architecture

![Architecture](docs/figures/architecture_diagram.png)

---

## Execution Flow

![Flow](docs/figures/flow_diagram.png)

The full pipeline follows industrial multilevel practice:

```
Benchmark (.bench / .hgr)
        │
        ▼
  parse_bench / parse_hgr           ← area weights, gate types, fixed vertices
        │
        ▼
  apply_global_net_threshold        ← remove large global nets (OpenROAD flag)
        │
        ▼
  [STA annotation]                  ← AT/RAT/slack → net criticality (--timing-aware)
        │
        ▼
  [HEM coarsening]                  ← Heavy Edge Matching levels (--multilevel)
        │
        ▼
  Metaheuristic optimize()          ← 8 bio-inspired algorithms
  (ABC / KH / MBO / EWA / EHO      ← spectral seed + random population
   / MS / SMA / HHO)
        │
        ▼
  [FM refinement per level]         ← k-way Fiduccia-Mattheyses (--fm-refinement)
        │
        ▼
  evaluate()                        ← SOED / cutsize / imbalance / timing / HPWL
        │
        ▼
  results/run.csv  +  .part files
```

---

## Algorithms

| ID | Algorithm | Biological Metaphor |
|---|---|---|
| **ABC** | Artificial Bee Colony | Foraging behaviour of honeybees |
| **KH** | Krill Herd | Swarm motion of Antarctic krill |
| **MBO** | Monarch Butterfly Optimization | Seasonal migration of monarch butterflies |
| **EWA** | Earthworm Optimization Algorithm | Reproduction and Cauchy mutation of earthworms |
| **EHO** | Elephant Herding Optimization | Clan-based movement of elephants |
| **MS** | Moth Search | Levy-flight spiral navigation toward light |
| **SMA** | Slime Mould Algorithm | Vein-network propagation of *Physarum polycephalum* |
| **HHO** | Harris Hawks Optimization | Cooperative hunting strategies of Harris hawks |

All algorithms share the same interface and are transparently wrapped by the
multilevel + FM pipeline:

```python
# Low-level: metaheuristic only
partition, history = optimizer.optimize(hg, cfg, seed=42)

# High-level: multilevel + FM wrapping (recommended)
partition, history = optimizer.run(hg, cfg, seed=42)
```

---

## Expert EDA Modules

### Objective: SOED + Timing-Weighted Cut + HPWL

```
F = w_cut  · SOED(partition)                        ← Sum of External Degrees
  + w_bal  · max(0, imbalance − balance_constraint) · (W/k)
  + w_tim  · Σ_n w(n)·(1 + crit(n))  if net n cut  ← timing-weighted cut
  + w_plc  · HPWL(partition)                        ← Half-Perimeter Wirelength
```

**SOED** (Sum of External Degrees) is the canonical k-way cut metric:
`SOED = Σ_n w(n)·(|S(n)|−1)` where `S(n)` is the set of partitions spanned by net n.
It equals binary cut-size for k=2 and penalises nets spanning many partitions for k>2.

Default weights: `w_cut=1.0`, `w_bal=10.0`, `w_tim=1.0` (active only when `--timing-aware`), `w_plc=0.0`

---

### FM Local Refinement (Fiduccia-Mattheyses)

Activated by default (`--fm-refinement`, disable with `--no-fm-refinement`).

**Algorithm (one pass):**
1. Compute initial SOED gain table `gain_to[v][q]` — O(Σ|net|·k)
2. Insert all movable vertices into a lazy-deletion max-heap
3. Greedily extract and execute the best balance-feasible move
4. Update gains of affected neighbours with exact incremental SOED formulas
5. Track the move prefix that minimises cumulative SOED
6. Roll back moves after the best prefix; repeat passes until no improvement

**Gain formula** for moving v from partition p to q:
```
gain(v, p→q) = Σ_n [w(n) if pc[n][p]==1]   ← uncut from p
             − Σ_n [w(n) if pc[n][q]==0]   ← new span added to q
```

Fixed vertices (`hg.fixed_vertices`) are never moved.
Complexity: O(Σ|net|·k·log n) per pass with lazy-deletion heap.

---

### Static Timing Analysis (STA)

Activated by `--timing-aware`. For ISCAS BENCH netlists.

**Gate delay model** (normalised FO4 units, 28 nm-style):

| Gate type | Delay |
|---|---|
| INV / BUF | 0.5 |
| NAND / NOR | 1.0 |
| AND / OR | 1.5 |
| XOR / XNOR | 2.0 |
| DFF | 0.0 (register boundary) |

**Pipeline:**
1. Topological sort (Kahn's algorithm) — cuts DFF arcs (clock boundary)
2. Forward AT propagation: `AT[sig] = max_input_AT + gate_delay`
3. Backward RAT propagation: `RAT[sig] = min_consumer(RAT[c] − delay[c])`
4. Per-signal slack: `slack = RAT − AT`
5. Per-net criticality: `crit(n) = (max_slack − min_slack_in_net) / max_slack ∈ [0, 1]`

Nets on the critical path receive `crit ≈ 1.0`; slack-rich nets receive `crit ≈ 0.0`.
Heuristic fallback (fanout-based) is used when gate definitions are unavailable.

```bash
python src/main.py --benchmark c880.bench --timing-aware --w-timing 1.0
```

---

### Multilevel Coarsening (Heavy Edge Matching)

Activated by `--multilevel`. Recommended for circuits with > 5 000 cells.

**Three-phase multilevel framework:**

```
COARSENING:    HEM → cluster map → coarse hypergraph   (repeated until |V| ≤ coarsen_to)
INITIAL PART:  metaheuristic on coarsest hypergraph
UNCOARSENING:  project partition back → FM refinement at every level
```

**HEM matching:**
- Weighted adjacency: contribution `w(n)/(|n|−1)` from every net containing both u and v
- Random vertex order (avoids systematic bias)
- Each vertex is matched with its heaviest unmatched, non-fixed neighbour
- Fixed vertices form singleton clusters (never merged)
- Stop when coarsening ratio < 5% (cannot compress further)

```bash
python src/main.py --benchmark c7552.bench --multilevel --coarsen-to 200
```

---

### Spectral Initialisation

`BaseOptimizer._init_population()` automatically seeds the first population
slot from the **Fiedler eigenvector** embedding for graphs ≤ 2 000 vertices:

1. Build clique-expansion weighted adjacency (same as HEM adjacency)
2. Form graph Laplacian L = D − A
3. Compute k smallest non-trivial eigenvectors (spectral embedding)
4. Run k-means (20 iterations) on the embedding → balanced initial partition

The remaining pop_size−1 slots are uniformly random, balance-repaired.
This biases the search towards a topologically sound starting point.

---

## Results

### Cut Size Comparison — ISCAS'85

![ISCAS85 Cut Comparison](docs/figures/cut_comparison_iscas85.png)

### Cut Size Comparison — ISCAS'89

![ISCAS89 Cut Comparison](docs/figures/cut_comparison_iscas89.png)

### Convergence Curves

![Convergence](docs/figures/convergence_example.png)

### Algorithm Performance Radar

![Radar](docs/figures/algorithm_radar.png)

### Runtime Scaling

![Runtime](docs/figures/runtime_scaling.png)

### Objective Weight Sensitivity

![Objective Weights](docs/figures/objective_weights.png)

---

## OpenROAD / TritonPart Interface Alignment

| Framework field | Meaning | OpenROAD concept |
|---|---|---|
| `num_parts` | Number of partitions | `-num_parts` |
| `balance_constraint` | Allowed imbalance ratio | `-balance_constraint` |
| `hypergraph_file` | HGR input | `-hypergraph_file` |
| `placement_file` | Placement embedding | `-placement_file` |
| `timing_aware` | Timing-driven mode flag | `-timing_aware_flag` |
| `solution_file` | Saved partition labels | `-solution_file` |
| `global_net_threshold` | Skip nets larger than N | `-global_net_threshold` |
| `evaluate()` | Cut/balance validation | `evaluate_hypergraph_solution` |

---

## Benchmarks

| Family | Circuits | Format |
|---|---|---|
| ISPD98 | 18 hypergraph circuits | `.hgr` |
| ISCAS'85 | c17, c432, c499, c880, c1355, c1908, c2670, c3540, c5315, c6288, c7552 | `.bench` |
| ISCAS'89 | s27, s298, s344 … s38584 (30 circuits) | `.bench` / `.v` |

Vertex weights use **NAND-equivalent gate area** so balance is area-driven:

| Gate | Area (NAND eq.) |
|---|---|
| INPUT / OUTPUT | 0.0 |
| INV / BUF | 0.67 |
| NAND / NOR | 1.0 |
| AND / OR | 1.33 |
| XOR / XNOR | 2.67 |
| DFF | 6.0 |

---

## Repository Layout

```
vlsi-partition-framework/
├── docker/
│   └── Dockerfile
├── configs/
│   └── default.yaml              ← partitioning, refinement, coarsening, timing config
├── benchmarks/
│   └── benchmark_manifest.csv
├── docs/
│   ├── expert_prompt.md
│   └── figures/                  ← SVG + PNG publication figures
│       ├── architecture_diagram.{svg,png}
│       ├── flow_diagram.{svg,png}
│       ├── convergence_example.{svg,png}
│       ├── cut_comparison_iscas85.{svg,png}
│       ├── cut_comparison_iscas89.{svg,png}
│       ├── algorithm_radar.{svg,png}
│       ├── objective_weights.{svg,png}
│       └── runtime_scaling.{svg,png}
├── requirements.txt
├── results/
│   └── cut_results_template.csv
├── scripts/
│   ├── fetch_benchmarks.sh
│   ├── generate_figures.py
│   └── run_all.sh
└── src/
    ├── main.py                   ← CLI entry point
    ├── objective.py              ← Hypergraph, PartitionConfig, SOED, HPWL, evaluate
    ├── benchmark_io.py           ← BENCH / HGR parsers, gate-area weights, fixed vertices
    ├── fm_refine.py              ← k-way FM refinement (SOED gain, lazy-deletion heap)
    ├── timing.py                 ← STA: topological sort, AT/RAT, net criticality
    ├── coarsening.py             ← HEM multilevel: CoarseningHierarchy, project+uncoarsen
    ├── openroad_adapter.py       ← TritonPart subprocess wrapper
    ├── reporting.py              ← CSV append, summary tables, figure generation
    └── optimizers/
        ├── __init__.py
        ├── base.py               ← BaseOptimizer: run(), spectral init, FM integration
        ├── abc.py  kh.py  mbo.py  ewa.py
        └── eho.py  ms.py  sma.py  hho.py
```

---

## Quick Start

### With Docker

```bash
# Build
docker build -t vlsi-partition -f docker/Dockerfile .

# Download benchmarks
docker run --rm -v $(pwd)/benchmarks:/app/benchmarks vlsi-partition \
    bash scripts/fetch_benchmarks.sh

# Run all algorithms on all benchmarks (k=2, 3 seeds, 200 iterations, FM on)
docker run --rm \
    -v $(pwd)/benchmarks:/app/benchmarks \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/docs:/app/docs \
    vlsi-partition \
    bash scripts/run_all.sh --k 2 --seeds "0 1 2" --iter 200
```

### Without Docker

```bash
pip install -r requirements.txt

# Download benchmarks
bash scripts/fetch_benchmarks.sh

# Single circuit — HHO with FM refinement (default on), 3 seeds
python src/main.py \
    --benchmark benchmarks/circuits/c432.bench \
    --algorithm HHO \
    --num-parts 2 --balance-constraint 0.1 \
    --seeds 0 1 2 --max-iter 300 --verbose

# Timing-aware run (STA + timing-weighted cut)
python src/main.py \
    --benchmark benchmarks/circuits/c880.bench \
    --timing-aware --w-timing 1.0 \
    --algorithm HHO SMA --seeds 0 1 2

# Multilevel run (recommended for large circuits > 5k cells)
python src/main.py \
    --benchmark benchmarks/circuits/c7552.bench \
    --multilevel --coarsen-to 200 \
    --algorithm HHO --seeds 0 1 2

# All 8 algorithms, all circuits
bash scripts/run_all.sh --k 2 --seeds "0 1 2" --iter 200

# Regenerate publication-ready figures
python scripts/generate_figures.py
```

---

## CLI Reference

```
python src/main.py [OPTIONS]

Input
  --benchmark FILE         .bench / .v / .hgr benchmark file
  --manifest CSV           run every circuit listed in benchmark_manifest.csv
  --family NAME            override family tag in result rows

Partitioning (OpenROAD-aligned)
  --num-parts K            k-way partition count                (default: 2)
  --balance-constraint B   allowed imbalance ratio              (default: 0.10)
  --global-net-threshold T skip nets larger than T pins         (default: 1000)

Timing
  --timing-aware           enable STA and timing-weighted cut
  --clock-period T         STA clock period in FO4 units (default: auto-detect)
  --w-timing FLOAT         timing penalty weight                (default: 1.0)

Placement
  --placement-file FILE    enable HPWL placement-aware mode
  --w-placement FLOAT      HPWL penalty weight                  (default: 0.0)

Objective weights
  --w-cut FLOAT            SOED / cut weight                    (default: 1.0)
  --w-balance FLOAT        imbalance penalty weight             (default: 10.0)

FM refinement
  --fm-refinement          apply FM after metaheuristic         (default: ON)
  --no-fm-refinement       disable FM refinement
  --fm-max-passes P        maximum FM passes per level          (default: 10)

Multilevel coarsening
  --multilevel             enable HEM multilevel coarsening     (default: OFF)
  --no-multilevel          disable multilevel
  --coarsen-to N           stop coarsening at |V| ≤ N           (default: 200)

Optimiser
  --algorithm ALG [...]    ABC KH MBO EWA EHO MS SMA HHO  (or ALL)
  --pop-size INT           population size                      (default: 30)
  --max-iter INT           iteration count                      (default: 200)
  --seeds S [...]          random seeds — one run per seed      (default: [0])

Output
  --results-csv CSV        append results to this CSV file
  --solution-dir DIR       write .part solution files here
  --figures-dir DIR        save generated figures here
  --no-figures             skip figure generation
  --openroad-baseline      also run TritonPart (requires OpenROAD in PATH)
  --verbose / -v           print per-run metrics
```

---

## Result Schema

`results/run.csv` columns:

| Column | Description |
|---|---|
| `benchmark` | Circuit name (stem of file) |
| `family` | ISPD98 / ISCAS85 / ISCAS89 |
| `algorithm` | ABC / KH / MBO / EWA / EHO / MS / SMA / HHO / TritonPart |
| `num_parts` | k-way partition count |
| `balance_constraint` | allowed imbalance ratio |
| `seed` | random seed |
| `cutsize` | binary hyperedge cut (nets spanning ≥ 2 partitions) |
| `soed` | Sum of External Degrees — canonical k-way cut metric |
| `imbalance` | max fractional deviation from ideal weight W/k |
| `timing_penalty` | timing-weighted cut (0 unless `--timing-aware`) |
| `runtime_sec` | wall-clock run time |
| `feasible` | balance constraint satisfied? |
| `multilevel` | multilevel coarsening enabled? |
| `fm_refinement` | FM post-refinement enabled? |
| `notes` | free-form annotation |

---

## Adding a New Algorithm

1. Create `src/optimizers/my_algo.py` inheriting `BaseOptimizer`.
2. Implement `optimize(self, hg, cfg, seed) → (partition, history)`.
   - Use `self._init_population(hg, cfg, rng)` for the initial population
     (includes spectral seeding automatically).
   - Use `self._obj(hg, part, cfg)` for objective evaluation.
   - Use `self._repair(part, hg, cfg, rng)` after every solution update.
3. Register in `src/optimizers/__init__.py`:
   ```python
   from .my_algo import MyAlgoOptimizer
   REGISTRY["MYALGO"] = MyAlgoOptimizer
   ```
4. Run with `--algorithm MYALGO`.

FM refinement and multilevel coarsening are applied automatically by
`BaseOptimizer.run()` — no changes needed in your algorithm.

---

## Figure Generation Rules

All figures follow publication standards:

- White background (`facecolor="white"`)
- No text overlap (`tight_layout` + rotated labels)
- Non-overlapping legend entries
- Orthogonal connector arrows in block diagrams
- Exported as **SVG** (vector) and **PNG** (300 dpi raster)

```bash
python scripts/generate_figures.py
```

---

## Dependencies

```
numpy>=1.24
matplotlib>=3.7
pyyaml>=6.0
```

---

## References

### Metaheuristic algorithms
- Karaboga & Basturk (2007) — Artificial Bee Colony
- Gandomi & Alavi (2012) — Krill Herd
- Wang et al. (2019) — Monarch Butterfly Optimization
- Wang et al. (2018) — Earthworm Optimization Algorithm
- Wang et al. (2016) — Elephant Herding Optimization
- Wang (2018) — Moth Search Algorithm
- Li et al. (2020) — Slime Mould Algorithm
- Heidari et al. (2019) — Harris Hawks Optimization

### EDA / partitioning theory
- Fiduccia & Mattheyses (1982) — FM bisection refinement, DAC
- Sanchis (1989) — k-way FM extension, IEEE Trans. CAD
- Karypis & Kumar (1998) — Multilevel k-way partitioning (METIS), SIAM J. Sci. Comput.
- Alpert & Kahng (1995) — SOED metric, ISPD
- Alpert, Huang & Kahng (1997) — Multilevel circuit partitioning, ICCAD

### Benchmarks
- Alpert & Kahng (1995) — ISPD98 benchmark suite
- Brglez & Fujiwara (1985) — ISCAS'85 combinational benchmarks
- Brglez, Bryan & Kozminski (1989) — ISCAS'89 sequential benchmarks
- OpenROAD TritonPart — multilevel hypergraph partitioner (reference baseline)
