# PROGRESS — AI OHT Fab

## Status
All phases in PLAN.md are done. Tests: `pytest backend/tests` 27 passed, `node --test frontend/tests` 4 passed.
Real-WebGL rendering was only checked headlessly with a stubbed Three.js in the cloud workspace (no JS errors, panels fill in, mode switch works). Check it on the Mac after `npm install`.

## Simulation sanity (rule_basic, normal, 30 min)
About 906 moves/h, average delivery 167 s, p95 267 s, fleet utilization about 68%. No deadlocks or gap violations in any scenario.

## Models
| run | how | note |
|---|---|---|
| oht_bc | BC + 1 DAgger round from rule_smart, mix of scenarios | imitation accuracy about 98% |
| oht_ppo_a | PPO fine-tune from oht_bc, lr 1e-4, clip 0.1, ent 0.001, 600k steps | **recommended**, best eval at 123k steps |
| oht_ppo_b | PPO fine-tune from oht_bc, lr 3e-4, clip 0.2, ent 0.003 | became unstable later (eval fell to −283 by 600k); best snapshot kept from 61k |

## Evaluation (7 scenarios × 5 held-out seeds, same seeds for every policy)
Full tables: [docs/eval_report.md](docs/eval_report.md), chart: docs/eval_comparison.png

| policy | mean reward | moves/h | p95 delivery s | wait for vehicle s | hot on-time |
|---|---|---|---|---|---|
| ppo(oht_ppo_a) | **1161** | 894 | 396 | 82.3 | 99.8% |
| ppo(oht_bc) | 1125 | 891 | 397 | 84.7 | 99.7% |
| rule_smart | 1123 | 889 | 396 | 82.8 | 99.7% |
| rule_basic | 1109 | 890 | 399 | 85.8 | 99.8% |
| ppo(oht_ppo_b) | 1080 | 887 | 402 | 85.5 | 99.7% |
| random | −1738 | 716 | 668 | 205.6 | 84.9% |

oht_ppo_a minus rule_smart, per scenario (mean reward difference):

| scenario | difference |
|---|---|
| mixed | +111 |
| high_load | +102 |
| bay_surge | +38 |
| fleet_short | +29 |
| normal | +24 |
| hot_lots | −4 |
| track_block | −34 |

### Honest reading
- oht_ppo_a is ahead of rule_smart by **+38 reward on average (standard error ±25)**. It wins 19 of 35 paired episodes. The gap is about 1.5 standard errors, so it is **not a statistically clear win**: the model is roughly as good as the smart rule, slightly better under heavy or mixed load, and slightly worse under track_block.
- PPO from scratch (no BC) was not trained or evaluated in this session; based on the conveyor project, a BC warm start is the recommended path.
- The rule baselines are already near the fab's capacity: high_load and fleet_short saturate the fleet whatever the dispatcher does (wait for vehicle is about 200 s). Bigger AI gains would need more decision freedom, for example dispatching to vehicles that are still busy, re-dispatch, or per-vehicle speed control. These are possible next steps.

## Possible next steps
- Longer or wider PPO search (more seeds), and reward shaping on wait for vehicle.
- Pre-assign jobs to vehicles that are about to become free; dynamic re-dispatch.
- Vehicle breakdowns, multi-floor lifters, and a larger fab (e.g. 16 bays, 150 vehicles).
