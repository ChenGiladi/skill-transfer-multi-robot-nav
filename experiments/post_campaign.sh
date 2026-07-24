#!/usr/bin/env bash
# Chained pipeline: wait for campaign_all.sh, then revision experiments, then analysis.
cd "$(dirname "$0")"
echo "[post] start $(date)"

# 1. Wait for the main/online/throughput campaigns to finish
while ! grep -q "ALL DONE" campaign_all.log; do
  sleep 60
done
echo "[post] campaign_all finished $(date)"

# 2. Revision experiment campaign (lib variants, scripted, prioritized, online caps, map3)
python3 run_experiments_revision.py --reps 30 --procs 14
echo "[post] revision campaign done rc=$? $(date)"

# 3. Analysis + figures
python3 analyze_rrtbc.py
echo "[post] analyze_rrtbc done rc=$? $(date)"
python3 analyze_revision.py
echo "[post] analyze_revision done rc=$? $(date)"
python3 make_heatmaps_rrtbc.py
echo "[post] heatmaps done rc=$? $(date)"

echo "[post] PIPELINE ALL DONE $(date)"
