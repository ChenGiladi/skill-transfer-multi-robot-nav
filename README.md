# Environment-Specific Skill Transfer for Decentralized Multi-Robot Navigation

Reproducible code, data, and trained model for the article

> **Environment-Specific Skill Transfer for Decentralized Multi-Robot Navigation via
> Hybrid RRT and Behavior Cloning in Grid-Based Industrial Environments**
> Yovel Atia and Chen Giladi.

Everything needed to regenerate every reported number, table, and figure is in this
repository. The simulators are deterministic and seeded: re-running any episode
reproduces the committed results exactly (wall-clock runtime is the only
hardware-dependent column).

## Repository layout

```
.
├── reproducible/              # deterministic, seeded study (run everything from here)
│   ├── sim_rrtbc.py           #   hybrid RRT + Behavior-Cloning simulator (+ optional sharing)
│   ├── sim_rrt_online.py      #   online-RRT baseline
│   ├── scenario.py            #   builds scenarios; filters each library against its native map
│   ├── run_experiments_*.py   #   experiment campaigns -> results_*.csv
│   ├── analyze_rrtbc.py       #   statistics + performance figures + LaTeX tables
│   ├── make_heatmaps_rrtbc.py #   spatial / schematic figures + route-diagnostics table
│   ├── pubstyle.py            #   shared publication figure style
│   ├── results_*.csv          #   committed raw per-episode results
│   ├── README.md              #   full reproduction manifest (which script makes which output)
│   ├── CITATION.cff, LICENSE, .zenodo.json
│   └── stats_summary_rrtbc.md
├── dataset/                   # offline RRT route libraries
│   ├── original_data.csv      #   Map1-generated library  ("naive" on Map2)
│   └── alternative_data.csv   #   Map2-generated library  ("adapted")
├── env/                       # the two 50x50 grid maps
│   ├── map1.txt
│   └── map2.txt
├── trained_model_NN.pth       # trained Behavior-Cloning network (used verbatim)
├── Functions_code.py          # original per-step navigation + BC inference (verbatim)
├── Functions_code_RRT.py      # original RRT planner (verbatim)
├── agent_navigation.py        # original single-agent navigation helper
├── Main_iter.py, Main_time.py # original student entry points (kept for provenance)
├── requirements.txt
├── LICENSE                    # MIT
└── CITATION.cff
```

## Installation

**Prerequisites:** Python **3.12** and `git`. A GPU is *not* required — everything runs on CPU.

```bash
# 1. Clone the repository
git clone https://github.com/ChenGiladi/skill-transfer-multi-robot-nav.git
cd skill-transfer-multi-robot-nav

# 2. Create and activate an isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate            # Linux / macOS
# .venv\Scripts\activate             # Windows (PowerShell)

# 3. Install the pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify the installation
python -c "import numpy, scipy, pandas, torch, matplotlib, seaborn, sklearn, joblib; print('environment OK')"
```

Dependencies (pinned in `requirements.txt`): `numpy` 2.3, `scipy` 1.16, `pandas` 2.3,
`torch` 2.9, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`. The deterministic result
columns reproduce on nearby versions too; only wall-clock runtime is hardware-dependent.

## Reproduce the results

All commands run from the `reproducible/` directory:

```bash
cd reproducible
```

**Fast path (seconds, no experiments re-run).** The raw per-episode `results_*.csv` are
committed, so the analysis and figure scripts reproduce every reported statistic, table,
and figure directly:

```bash
python3 analyze_rrtbc.py            # statistics + LaTeX tables + performance figures
python3 make_heatmaps_rrtbc.py      # spatial / schematic figures + route-diagnostics table
```

**Full path (re-runs the deterministic, seeded experiment campaigns).** Re-generates the
`results_*.csv` from scratch, then analyses them — output is byte-identical to the
committed CSVs except the wall-clock runtime column:

```bash
python3 run_experiments_rrtbc.py --reps 15        # skill transfer + collision-history sharing
python3 run_experiments_rrt_online.py --reps 15   # online-RRT baseline
python3 run_experiments_rrtbc.py --throughput     # throughput-dependence sweep
python3 analyze_rrtbc.py
python3 make_heatmaps_rrtbc.py
```

> **Figures.** The figure files themselves are *not* committed to this repository; the
> scripts above regenerate them locally from the committed results. See
[`reproducible/README.md`](reproducible/README.md) for the full output-by-output
reproduction manifest and the verified-determinism notes.

## Data

| Asset | Description |
|---|---|
| `dataset/original_data.csv` | Offline RRT routes generated on Map1 (the Map2-*naive* library). |
| `dataset/alternative_data.csv` | Offline RRT routes generated on Map2 (the Map2-*adapted* library). |
| `env/map1.txt`, `env/map2.txt` | The two 50×50 occupancy grids (tab-separated). |
| `trained_model_NN.pth` | Trained Behavior-Cloning policy, used unchanged. |
| `reproducible/results_*.csv` | Raw per-episode results behind every table and figure. |

## Environment

Tested with **Python 3.12** (`numpy` 2.3, `scipy` 1.16, `pandas` 2.3, `torch` 2.9). All
result columns reproduce exactly across machines; only wall-clock runtime varies.

## License and citation

Released under the MIT License (see [`LICENSE`](LICENSE)). If you use this code or data,
please cite the article and this repository — citation metadata is in
[`CITATION.cff`](CITATION.cff).
