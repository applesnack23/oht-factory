# CLAUDE.md — AI OHT Fab

Semiconductor fab AMHS (OHT) digital twin + RL dispatcher. Sister project of `../ai-conveyor-factory` (same stack, separate ports).

## Layout
- `config/` fab.json (geometry, speeds, fleet), demand.json, reward.json, scenarios.json — all tunables live here.
- `backend/sim/` pure-python simulation: `track.py` rail graph + Floyd routing, `layout.py` fab generator (ring, 2 shortcuts, 8 intrabay loops, 2 stockers, 120 ports), `fab.py` vehicles/jobs/decisions, `demand.py`.
- `backend/ai/` `oht_env.py` (Gymnasium env, one step = one decision), `train_ppo.py` (SB3 or NumpyPPO fallback), `pretrain_bc.py` (BC + DAgger from PreassignRulePolicy by default), `evaluate.py`, `inference.py`.
- `backend/policies/rules.py` rule baselines (basic NVF, smart, preassign, random) — they read the observation only.
- `backend/api/` FabService + routes; `backend/main.py` FastAPI (port 8001); `backend/api/devserver.py` stdlib fallback server.
- `frontend/` Vite + Three.js (port 5174, proxies /api and /ws to 8001).

## Rules
- Run tests before claiming anything works: `python -m pytest -q backend/tests`, `cd frontend && npm test`.
- Report RL results against rule_smart and rule_preassign honestly (eval seeds ≥ 1_000_000 are held out from training).
- Observation/action sizes are fixed (obs 127, MultiDiscrete([9])); changing them invalidates all saved models.
- Rail is one-way; safety invariants (min gap, merge FIFO locks, diverge sibling check) are covered by test_sim.py — keep them green.
