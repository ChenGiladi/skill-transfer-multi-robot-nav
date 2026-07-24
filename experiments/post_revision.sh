#!/usr/bin/env bash
# Relaunch after the sys.path fix: revision experiments, then their analysis.
cd "$(dirname "$0")"
echo "[rev] start $(date)"
python3 run_experiments_revision.py --reps 30 --procs 13
echo "[rev] revision campaign done rc=$? $(date)"
python3 analyze_revision.py
echo "[rev] analyze_revision done rc=$? $(date)"
echo "[rev] REVISION PIPELINE ALL DONE $(date)"
