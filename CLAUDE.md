# CLAUDE.md — AI OHT Fab

Semiconductor fab AMHS (OHT) digital twin + RL dispatcher. Sister project of `../ai-conveyor-factory` (same stack, separate ports).

## Layout
- `config/` fab.json (geometry, speeds, fleet), demand.json, reward.json, scenarios.json — all tunables live here.
- `backend/sim/` pure-python simulation: `track.py` rail graph + Floyd routing, `layout.py` fab generator (ring, 2 shortcuts, 8 intrabay loops, 2 stockers with 2 IN + 2 OUT ports, 120 ports), `fab.py` vehicles/jobs/decisions, `lotflow.py` closed-loop lots (64 tools with EFEM load ports, one FOUP per LP, deliveries only to reserved free LPs, stocker queues), `stocker.py` crane, `demand.py` Job record.
- `backend/ai/` `oht_env.py` (Gymnasium env, one step = one decision), `train_ppo.py` (SB3 or NumpyPPO fallback), `pretrain_bc.py` (BC + DAgger from PreassignRulePolicy by default), `evaluate.py`, `inference.py`.
- `backend/policies/rules.py` rule baselines (basic NVF, smart, preassign, random) — they read the observation only.
- `backend/api/` FabService + routes; `backend/main.py` FastAPI (port 8001); `backend/api/devserver.py` stdlib fallback server.
- `frontend/` Vite + Three.js (port 5174, proxies /api and /ws to 8001).

## Rules
- Run tests before claiming anything works: `python -m pytest -q backend/tests`, `cd frontend && npm test`.
- Report RL results against rule_smart and rule_preassign honestly (eval seeds ≥ 1_000_000 are held out from training).
- Observation/action sizes are fixed (obs 127, MultiDiscrete([9])); changing them invalidates all saved models.
- Transport requests come from lots finishing at tools (lotflow); per-lot randomness is keyed by (seed, lot, step) so policies face the same lots. test_lotflow.py guards one-FOUP-per-LP and constant WIP.
- Rail is one-way; safety invariants (min gap, merge FIFO locks, diverge sibling check) are covered by test_sim.py — keep them green.
