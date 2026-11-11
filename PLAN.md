# PLAN — AI OHT Fab

Goal: a medium-scale (≈130 m × 110 m, 8 bays, 50 vehicles) OHT virtual fab where an AI makes three kinds of decisions:
1. **DISPATCH** — which idle vehicle takes a new transport job (or wait).
2. **ROUTE** — at the two shortcut decision points, take the shortcut or stay on the interbay ring.
3. **REPOSITION** — where an idle vehicle parks (which bay/stocker) to anticipate demand.

| Phase | Content | Status |
|---|---|---|
| 1 | Rail graph, layout generator, routing tables | done |
| 2 | Vehicle kinematics, min-gap, merge locks, diverge safety, deadlock detection | done |
| 3 | Demand model, scenarios (7), KPIs | done |
| 4 | Gym env (event-driven decisions, K=8 candidates + default, masks) | done |
| 5 | Rule baselines (basic NVF / smart / random) | done |
| 6 | BC + DAgger warm start, PPO fine-tune, evaluation | done (see PROGRESS) |
| 7 | API + WebSocket + Three.js 3D twin + dashboard + watch-training | done |
| 8 | Docs, delivery | done |

Not in scope (possible next steps): multi-floor/lifter, vehicle failures, real MES data import, multi-agent per-vehicle control.
