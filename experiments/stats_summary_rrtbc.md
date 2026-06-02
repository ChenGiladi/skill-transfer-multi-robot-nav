# RRT + Behavior-Cloning campaign — statistics

Source: results_rrtbc.csv (480 episodes). Faithful method: RRT expert routes + trained BC local policy + route reuse. Collisions = blocked-move (3-step-lookahead-into-obstacle) events. Episodes run to steps=2000 (~245 completed tasks at 5 agents), so the shared collision database fills and comparisons have power.


## Skill transfer on Map2 (hybrid; Map2-naive 'original' vs Map2-adapted 'alternative')
- 2 agents: original coll 281.33 ± 28.10 (per-task 3.05, fail-rate 0.20); alternative coll 41.13 ± 15.96 (per-task 0.42, fail-rate 0.06); adaptation cuts collisions 85% (MWU original>alt p=1.68e-06).
- 4 agents: original coll 703.20 ± 60.72 (per-task 4.00, fail-rate 0.23); alternative coll 205.53 ± 36.38 (per-task 1.08, fail-rate 0.09); adaptation cuts collisions 71% (MWU original>alt p=1.69e-06).
- 6 agents: original coll 1263.07 ± 78.10 (per-task 4.98, fail-rate 0.26); alternative coll 575.87 ± 85.05 (per-task 2.09, fail-rate 0.12); adaptation cuts collisions 54% (MWU original>alt p=1.7e-06).
- 8 agents: original coll 1944.20 ± 139.88 (per-task 5.97, fail-rate 0.29); alternative coll 1028.33 ± 107.43 (per-task 2.92, fail-rate 0.16); adaptation cuts collisions 47% (MWU original>alt p=1.69e-06).
- 10 agents: original coll 2686.93 ± 120.39 (per-task 6.83, fail-rate 0.32); alternative coll 1698.47 ± 114.26 (per-task 3.99, fail-rate 0.18); adaptation cuts collisions 37% (MWU original>alt p=1.7e-06).

## Collision-history sharing — ablation on map1 (5 agents)
- none: collisions 362.20 ± 65.66 | per-task 1.42 | tasks 254.73 ± 11.69
- selective 0.6: collisions 346.33 ± 62.03 | per-task 1.41 | tasks 247.67 ± 12.02
- full 1.0: collisions 316.33 ± 53.54 | per-task 1.29 | tasks 246.47 ± 8.59
- selective reduces collisions 4% vs none (MWU none>selective p=0.324).
- full reduces collisions 13% vs none (MWU none>full p=0.0232).
- full>selective one-sided p=0.938 (no migration).

## Collision-history sharing — ablation on map2 (5 agents)
- none: collisions 386.13 ± 59.79 | per-task 1.68 | tasks 231.47 ± 7.85
- selective 0.6: collisions 377.47 ± 38.40 | per-task 1.64 | tasks 231.20 ± 10.90
- full 1.0: collisions 379.40 ± 60.73 | per-task 1.63 | tasks 233.07 ± 8.67
- selective reduces collisions 2% vs none (MWU none>selective p=0.434).
- full reduces collisions 2% vs none (MWU none>full p=0.37).
- full>selective one-sided p=0.541 (selective better (migration under full)).

## Access-fraction sweep (Map1, 5 agents)
- p=0.0: collisions 362.20 ± 65.66 | per-task 1.42
- p=0.2: collisions 325.93 ± 51.56 | per-task 1.28
- p=0.4: collisions 345.07 ± 50.79 | per-task 1.38
- p=0.6: collisions 346.33 ± 62.03 | per-task 1.41
- p=0.8: collisions 336.40 ± 40.43 | per-task 1.36
- p=1.0: collisions 316.33 ± 53.54 | per-task 1.29
- Spearman ρ(p, collisions-per-task) = -0.09 (p=0.422); minimum at p=0.2.

## Sharing scalability (Map1, none vs selective 0.6)
- no sharing: ρ(fleet, tasks)=0.98 (p=6.1e-53); coll-per-task 0.11..4.11
- selective ($q_{\mathrm{share}}=0.6$): ρ(fleet, tasks)=0.98 (p=6e-53); coll-per-task 0.16..3.59

## Sharing multiple-comparison correction (Holm across the 4 ablation tests)
- map1 share=0.6: nominal p=0.324, Holm p=0.972, n.s. after correction.
- map1 share=1.0: nominal p=0.0232, Holm p=0.0929, n.s. after correction.
- map2 share=0.6: nominal p=0.434, Holm p=0.972, n.s. after correction.
- map2 share=1.0: nominal p=0.37, Holm p=0.972, n.s. after correction.
