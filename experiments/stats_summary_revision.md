# Revision-1 campaign — statistics

Source: results_revision.csv (1500 episodes, 30 seeds per condition), reference conditions from results_rrtbc.csv (T0, 30 seeds) and results_rrt_online.csv (paper cap 3000). All episodes: 2000 scheduler steps.


## R1 library variants on Map2 (offline filter / repair / regen of the naive library)
- 2 agents: naive coll 279.07 ± 36.92; filter coll 39.63 ± 19.48 (+86% vs naive, p=1.5e-11, p_perm=0.0001); repair coll 39.37 ± 17.40 (+86% vs naive, p=1.49e-11, p_perm=0.0001); regen coll 31.63 ± 9.83 (+89% vs naive, p=1.49e-11, p_perm=0.0001).
- 4 agents: naive coll 718.50 ± 61.57; filter coll 203.73 ± 31.98 (+72% vs naive, p=1.5e-11, p_perm=0.0001); repair coll 221.80 ± 43.79 (+69% vs naive, p=1.51e-11, p_perm=0.0001); regen coll 196.07 ± 33.41 (+73% vs naive, p=1.51e-11, p_perm=0.0001).
- 6 agents: naive coll 1272.13 ± 94.33; filter coll 543.13 ± 62.89 (+57% vs naive, p=1.51e-11, p_perm=0.0001); repair coll 555.43 ± 59.67 (+56% vs naive, p=1.51e-11, p_perm=0.0001); regen coll 469.27 ± 46.08 (+63% vs naive, p=1.51e-11, p_perm=0.0001).
- 8 agents: naive coll 1942.27 ± 124.38; filter coll 985.77 ± 83.58 (+49% vs naive, p=1.51e-11, p_perm=0.0001); repair coll 1016.43 ± 85.41 (+48% vs naive, p=1.51e-11, p_perm=0.0001); regen coll 873.90 ± 87.55 (+55% vs naive, p=1.51e-11, p_perm=0.0001).
- 10 agents: naive coll 2688.93 ± 113.24; filter coll 1521.07 ± 116.72 (+43% vs naive, p=1.51e-11, p_perm=0.0001); repair coll 1640.80 ± 111.37 (+39% vs naive, p=1.51e-11, p_perm=0.0001); regen coll 1416.83 ± 113.48 (+47% vs naive, p=1.51e-11, p_perm=0.0001).
- Library sizes after transform (from lib_cache): filter: 1387 routes; repair: 2862 routes; regen: 3238 routes.

## R2 scripted connector vs trained BC connector (Map2, adapted library)
- 2 agents: scripted coll 33.37 ± 14.89 vs BC coll 41.20 ± 17.97; MWU scripted>BC p=0.953, BC>scripted p=0.048.
- 4 agents: scripted coll 193.63 ± 36.12 vs BC coll 210.03 ± 33.93; MWU scripted>BC p=0.975, BC>scripted p=0.0255.
- 6 agents: scripted coll 474.70 ± 74.20 vs BC coll 573.37 ± 68.98; MWU scripted>BC p=1, BC>scripted p=3.03e-06.
- 8 agents: scripted coll 891.20 ± 66.36 vs BC coll 1045.27 ± 87.37; MWU scripted>BC p=1, BC>scripted p=3.7e-09.
- 10 agents: scripted coll 1486.13 ± 112.39 vs BC coll 1653.73 ± 104.81; MWU scripted>BC p=1, BC>scripted p=1.9e-07.
- Naive-library check: scripted (original) coll 1208.97 ± 787.55 vs BC naive hybrid 1380.18 ± 866.80 pooled over N (per-N table rows use the adapted pair above).
- Verdict: the trained BC connector does not outperform the scripted greedy controller on mean collisions at 0/5 fleet sizes.

## R3+R4 compute-budget frontier (Map2): runtime, coll/task, tasks per wall-clock second
- N=4:
  - Map2-naive hybrid: runtime 60.7 ± 6.5 s | coll/task 4.08 | tasks/s 2.92
  - Map2-adapted hybrid: runtime 49.5 ± 2.9 s | coll/task 1.11 | tasks/s 3.83
  - online RRT, cap 3000: runtime 262.0 ± 158.0 s | coll/task 0.81 | tasks/s 0.84
  - online RRT, cap 1000: runtime 23.7 ± 10.1 s | coll/task 0.81 | tasks/s 9.70
  - online RRT, cap 300: runtime 17.7 ± 8.0 s | coll/task 0.78 | tasks/s 16.39
  - prioritized planning: runtime 3.2 ± 1.7 s | coll/task 0.05 | tasks/s 108.90
- N=10:
  - Map2-naive hybrid: runtime 210.0 ± 19.6 s | coll/task 6.75 | tasks/s 1.90
  - Map2-adapted hybrid: runtime 184.6 ± 12.1 s | coll/task 3.87 | tasks/s 2.31
  - online RRT, cap 3000: runtime 921.9 ± 470.9 s | coll/task 2.73 | tasks/s 0.51
  - online RRT, cap 1000: runtime 88.1 ± 30.4 s | coll/task 2.65 | tasks/s 5.79
  - online RRT, cap 300: runtime 52.7 ± 17.6 s | coll/task 2.35 | tasks/s 13.07
  - prioritized planning: runtime 11.0 ± 5.0 s | coll/task 0.14 | tasks/s 78.79

## R5 transfer on Map3 (corridor/bottleneck; Map1 library vs Map3 library)
- 2 agents: naive coll 982.63 ± 62.93 (per-task 30.47); adapted coll 44.13 ± 16.37 (per-task 0.49); adaptation cuts collisions 96% (MWU naive>adapted p=1.5e-11, p_perm=0.0001).
- 4 agents: naive coll 2086.03 ± 77.24 (per-task 32.95); adapted coll 284.67 ± 41.73 (per-task 1.61); adaptation cuts collisions 86% (MWU naive>adapted p=1.51e-11, p_perm=0.0001).
- 6 agents: naive coll 3286.17 ± 99.65 (per-task 36.27); adapted coll 673.20 ± 71.17 (per-task 2.56); adaptation cuts collisions 80% (MWU naive>adapted p=1.51e-11, p_perm=0.0001).
- 8 agents: naive coll 4475.87 ± 102.22 (per-task 35.40); adapted coll 1276.67 ± 106.77 (per-task 3.79); adaptation cuts collisions 71% (MWU naive>adapted p=1.51e-11, p_perm=0.0001).
- 10 agents: naive coll 5862.10 ± 138.40 (per-task 38.02); adapted coll 2047.17 ± 119.75 (per-task 4.90); adaptation cuts collisions 65% (MWU naive>adapted p=1.51e-11, p_perm=0.0001).

## Holm within families
- R1 family (15 tests: 3 variants x fleet sizes): 15/15 survive Holm at 0.05.
  - filter N=2: nominal p=1.5e-11, Holm p=2.24e-10, sig after correction.
  - repair N=2: nominal p=1.49e-11, Holm p=2.24e-10, sig after correction.
  - regen N=2: nominal p=1.49e-11, Holm p=2.24e-10, sig after correction.
  - filter N=4: nominal p=1.5e-11, Holm p=2.24e-10, sig after correction.
  - repair N=4: nominal p=1.51e-11, Holm p=2.24e-10, sig after correction.
  - regen N=4: nominal p=1.51e-11, Holm p=2.24e-10, sig after correction.
  - filter N=6: nominal p=1.51e-11, Holm p=2.24e-10, sig after correction.
  - repair N=6: nominal p=1.51e-11, Holm p=2.24e-10, sig after correction.
  - regen N=6: nominal p=1.51e-11, Holm p=2.24e-10, sig after correction.
  - filter N=8: nominal p=1.51e-11, Holm p=2.24e-10, sig after correction.
  - repair N=8: nominal p=1.51e-11, Holm p=2.24e-10, sig after correction.
  - regen N=8: nominal p=1.51e-11, Holm p=2.24e-10, sig after correction.
  - filter N=10: nominal p=1.51e-11, Holm p=2.24e-10, sig after correction.
  - repair N=10: nominal p=1.51e-11, Holm p=2.24e-10, sig after correction.
  - regen N=10: nominal p=1.51e-11, Holm p=2.24e-10, sig after correction.
- R5 family (5 tests, one per fleet size): 5/5 survive Holm at 0.05.
  - N=2: nominal p=1.5e-11, Holm p=7.5e-11, sig after correction.
  - N=4: nominal p=1.51e-11, Holm p=7.5e-11, sig after correction.
  - N=6: nominal p=1.51e-11, Holm p=7.5e-11, sig after correction.
  - N=8: nominal p=1.51e-11, Holm p=7.5e-11, sig after correction.
  - N=10: nominal p=1.51e-11, Holm p=7.5e-11, sig after correction.
