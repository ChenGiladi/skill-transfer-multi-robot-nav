#!/usr/bin/env bash
cd "$(dirname "$0")"
echo "[campaign] start $(date)"
python3 run_experiments_rrtbc.py --reps 30 --procs 14
echo "[campaign] main done rc=$? $(date)"
python3 run_experiments_rrt_online.py --reps 30 --procs 14
echo "[campaign] online done rc=$? $(date)"
python3 run_experiments_rrtbc.py --throughput --reps 30 --procs 14
echo "[campaign] ALL DONE rc=$? $(date)"
