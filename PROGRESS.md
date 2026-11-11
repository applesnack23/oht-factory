# PROGRESS — AI OHT Fab

## Status
All phases in PLAN.md are done. Tests: `pytest backend/tests` 30 passed, `npm test` (frontend) 4 passed.
Real-WebGL rendering was only checked headlessly with a stubbed Three.js in the cloud workspace (no JS errors, panels fill in, mode switch works). Check it on the Mac after `npm install`.

## Simulation sanity (rule_basic, normal, 30 min)
About 906 moves/h, average delivery 167 s, p95 267 s, fleet utilization about 68%. No deadlocks or gap violations in any scenario.

## v2 — pre-assignment (current)
v1 analysis showed that about 49% of vehicle travel was empty and that dispatch could only use idle vehicles. v2 adds **pre-assignment**: a DISPATCH candidate may be a vehicle that is still driving to, or hoisting at, its drop port (time to free ≤ 90 s). The job is reserved for that vehicle, and the vehicle heads straight to the job when the drop finishes.
- Observation 111 → 127 (+2 features per candidate: busy, time to free). v1 models are incompatible and were removed.
- The classic rules (rule_basic, rule_smart) still use idle vehicles only. The new `rule_preassign` = smart + pre-assignment (a busy vehicle must be more than 10 s faster than the best idle one).
- "Wait for vehicle" is now measured until a vehicle actually heads to the job, not until it is reserved, so pre-assignment does not get credit just for booking early.

### Models
| run | how | note |
|---|---|---|
| oht_bc2 | BC + 1 DAgger round from rule_preassign, mix of scenarios | imitation accuracy 98.3% |
| oht_ppo2_a | PPO fine-tune from oht_bc2, lr 1e-4, clip 0.1, ent 0.001, 600k steps | |
| oht_ppo2_b | PPO fine-tune from oht_bc2, lr 5e-5, clip 0.1, ent 0, 600k steps | **recommended** |

### Evaluation (7 scenarios × 5 held-out seeds)
Full tables: [docs/eval_report.md](docs/eval_report.md), chart: docs/eval_comparison.png

| policy | mean reward | moves/h | avg delivery s | p95 s | wait for vehicle s | empty travel | hot on-time |
|---|---|---|---|---|---|---|---|
| ppo(oht_ppo2_b) | **2134** | 951 | 180.8 | 309 | 43.2 | 40.0% | 97.5% |
| ppo(oht_bc2) | 2132 | 951 | 180.0 | 308 | 41.5 | 40.4% | 97.4% |
| ppo(oht_ppo2_a) | 2124 | 951 | 180.7 | 308 | 43.5 | 40.1% | 96.9% |
| rule_preassign | 2063 | 945 | 183.2 | 314 | 44.9 | 40.4% | 97.5% |
| rule_smart | 1123 | 889 | 233.5 | 396 | 82.8 | 49.0% | 99.7% |
| rule_basic | 1109 | 890 | 235.0 | 399 | 85.8 | 49.3% | 99.8% |
| random | −1898 | 721 | 372.8 | 673 | 198.9 | 51.1% | 56.4% |

Paired comparison (same scenario and seed):
| | difference in reward | wins |
|---|---|---|
| oht_ppo2_b vs rule_smart | +1010 (SE ±99) | 35/35 |
| oht_ppo2_b vs rule_preassign (its teacher) | +71 (SE ±32) | 22/35 |

### Honest reading
- Against the classic controllers the AI is clearly better in every scenario: delivery time −23%, wait for vehicle −48%, moves/h +7%, empty travel 49% → 40%.
- **Most of that gain comes from the pre-assignment capability itself.** The hand-written rule_preassign gets about 93% of it. On top of its teacher, the AI adds +71 reward (about 2.2 SE, a modest but consistent edge). The edge is biggest in bay_surge (+188), high_load (+157) and mixed (+136), and roughly even in normal, hot_lots and track_block.
- PPO fine-tuning added little over BC (oht_bc2 ≈ oht_ppo2_b). The eval reward during training peaked early and then drifted.
- Hot-lot on-time rate fell from 99.7% to about 97.5% with pre-assignment, because a reserved vehicle cannot be taken back for a later hot lot. A possible next step is allowing a reservation to be cancelled for a hot lot.
- No deadlocks in any of the 245 evaluation episodes.

## v1 (historical, models removed)
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
