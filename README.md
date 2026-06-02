# Environment-Specific Skill Transfer for Decentralized Multi-Robot Navigation

Code, data, and the trained model for the article:

> **Environment-Specific Skill Transfer for Decentralized Multi-Robot Navigation via
> Hybrid RRT and Behavior Cloning in Grid-Based Industrial Environments**
> Yovel Atia and Chen Giladi.

This repository accompanies the article: it contains the hybrid RRT + Behavior-Cloning
navigation engine, the two grid environments and offline route libraries, the trained
behavior-cloning model, and the experiment and analysis scripts that produce the reported
tables and figures. The simulators are seeded, so a given run yields the same numbers on
any machine (wall-clock runtime aside).

## What's included

```
.
├── src/                       # navigation engine
│   ├── Functions_code.py      #   per-step navigation + behavior-cloning inference
│   ├── Functions_code_RRT.py  #   RRT planner
│   └── agent_navigation.py    #   single-agent navigation helper
├── experiments/               # the study — run everything from here
│   ├── sim_rrtbc.py           #   hybrid RRT + Behavior-Cloning simulator (+ optional sharing)
│   ├── sim_rrt_online.py      #   online-RRT baseline
│   ├── scenario.py            #   builds scenarios; matches each library to its map
│   ├── run_experiments_*.py   #   experiment campaigns → results_*.csv
│   ├── analyze_rrtbc.py       #   statistics, tables, and performance figures
│   ├── make_heatmaps_rrtbc.py #   spatial / schematic figures + route-diagnostics table
│   ├── pubstyle.py            #   shared figure style
│   ├── results_*.csv          #   the raw per-episode results behind the article
│   └── README.md              #   which script produces which output
├── data/                      # offline RRT route libraries
│   ├── original_data.csv      #   Map 1 library  (the "naive" library on Map 2)
│   └── alternative_data.csv   #   Map 2 library  (the "adapted" library)
├── maps/                      # the two 50×50 grid environments
│   ├── map1.txt
│   └── map2.txt
├── model/
│   └── trained_model_NN.pth   # trained behavior-cloning model (used unchanged)
├── legacy/                    # original entry-point scripts (kept for provenance)
├── docs/                      # development notes
├── requirements.txt
├── LICENSE                    # MIT
└── CITATION.cff
```

## Getting started

**Requirements:** Python **3.12** and `git`. No GPU is needed — everything runs on CPU.

```bash
git clone https://github.com/ChenGiladi/skill-transfer-multi-robot-nav.git
cd skill-transfer-multi-robot-nav

python3 -m venv .venv
source .venv/bin/activate            # Linux / macOS
# .venv\Scripts\activate             # Windows (PowerShell)

python -m pip install --upgrade pip
pip install -r requirements.txt

# quick check that the environment is ready
python -c "import numpy, scipy, pandas, torch, matplotlib, seaborn, sklearn, joblib; print('environment OK')"
```

Pinned dependencies (`requirements.txt`): `numpy` 2.3, `scipy` 1.16, `pandas` 2.3,
`torch` 2.9, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`.

## How to use

All commands run from the `experiments/` directory:

```bash
cd experiments
```

**Generate the tables and figures from the released results.** The per-episode
`results_*.csv` are included, so the analysis runs in seconds — no need to re-run the
experiments:

```bash
python3 analyze_rrtbc.py            # statistics, LaTeX tables, performance figures
python3 make_heatmaps_rrtbc.py      # spatial / schematic figures + route-diagnostics table
```

Figures and tables are written to `figures/` and `tables/` at the repository root (these
are not tracked — they are generated locally).

**Re-run the experiments from scratch (optional).** This regenerates the `results_*.csv`;
the seeded simulators give the same numbers as the included files (apart from wall-clock
runtime):

```bash
python3 run_experiments_rrtbc.py --reps 15        # skill transfer + collision-history sharing
python3 run_experiments_rrt_online.py --reps 15   # online-RRT baseline
python3 run_experiments_rrtbc.py --throughput     # throughput sweep
python3 analyze_rrtbc.py
python3 make_heatmaps_rrtbc.py
```

See [`experiments/README.md`](experiments/README.md) for a script-by-script guide to which
output each one produces.

## Data and model

| Asset | Description |
|---|---|
| `data/original_data.csv` | Offline RRT routes generated on Map 1 (the Map 2 "naive" library). |
| `data/alternative_data.csv` | Offline RRT routes generated on Map 2 (the Map 2 "adapted" library). |
| `maps/map1.txt`, `maps/map2.txt` | The two 50×50 occupancy grids (tab-separated). |
| `model/trained_model_NN.pth` | Trained behavior-cloning model, used unchanged. |
| `experiments/results_*.csv` | Raw per-episode results behind the tables and figures. |

## How to cite

Released under the MIT License (see [`LICENSE`](LICENSE)). If you use this code or data,
please cite the article and this repository; citation metadata is in
[`CITATION.cff`](CITATION.cff).
