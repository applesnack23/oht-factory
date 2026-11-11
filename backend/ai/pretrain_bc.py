"""Imitation warm start: clone a rule-based OHT controller (with DAgger), then fine-tune with PPO.

    python -m backend.ai.pretrain_bc --level mix --run oht_bc --episodes 60 --dagger 1
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from backend.ai.inference import MODELS_DIR, PPOPolicy
from backend.ai.numpy_ppo import Adam, NumpyPPO, RunningMeanStd, softmax_heads
from backend.ai.oht_env import OHTEnv, eval_seeds
from backend.ai.rollout import evaluate_policy
from backend.policies.rules import SmartRulePolicy


def collect(level, n, seed0, learner=None, beta=1.0, rng=None):
    env = OHTEnv(level)
    teacher = SmartRulePolicy(env)
    X, Y, M, R, D = [], [], [], [], []
    for ep in range(n):
        o, _ = env.reset(seed=seed0 + ep)
        if learner is not None:
            learner.reset(env)
        done = False
        while not done:
            m = env.action_masks()
            lab = int(teacher.act(o)[0])
            a = np.array([lab]) if (learner is None or rng.random() < beta) else learner.act(o)
            X.append(o.astype(np.float32)); Y.append([lab]); M.append(m)
            o, r, te, tr, _ = env.step(a)
            R.append(r); done = te or tr; D.append(done)
    return np.array(X), np.array(Y, dtype=np.int64), np.array(M), np.array(R), np.array(D)


def value_targets(R, D, gamma):
    rms, run, seq = RunningMeanStd(), 0.0, np.zeros_like(R)
    for t in range(len(R)):
        run = run * gamma + R[t]; seq[t] = run
        if D[t]:
            run = 0.0
    rms.update(seq)
    scale = float(np.sqrt(rms.var + 1e-8))
    G, acc = np.zeros_like(R), 0.0
    for t in reversed(range(len(R))):
        if D[t]:
            acc = 0.0
        acc = np.clip(R[t] / scale, -10, 10) + gamma * acc
        G[t] = acc
    return G, rms


def fit(model, X, Y, M, G, epochs, log, lr=1e-3, batch=512, seed=0):
    rng = np.random.default_rng(seed)
    op, ov = Adam(model.pi.params, lr=lr), Adam(model.vf.params, lr=lr)
    for ep in range(epochs):
        idx = rng.permutation(len(X)); ce = acc = vm = 0.0; k = 0
        for s in range(0, len(X), batch):
            mb = idx[s:s + batch]; x = X[mb].astype(np.float64); y = Y[mb]; B = len(mb)
            logits = model.pi.forward(x, cache=True)
            probs, logps = softmax_heads(logits, model.splits, M[mb])
            p, lp = probs[0], logps[0]
            d = np.zeros_like(logits); oh = np.zeros_like(p); oh[np.arange(B), y[:, 0]] = 1
            d[:, model.splits[0]] = (p - oh) / B
            op.step(model.pi.backward(d))
            v = model.vf.forward(x, cache=True)[:, 0]
            ov.step(model.vf.backward(((v - G[mb]) / B)[:, None]))
            ce -= lp[np.arange(B), y[:, 0]].mean(); acc += (p.argmax(1) == y[:, 0]).mean(); vm += ((v - G[mb]) ** 2).mean(); k += 1
        log(f"  bc epoch {ep + 1}/{epochs}: ce={ce / k:.4f} action-acc={acc / k:.4f} value-mse={vm / k:.3f}")
    model.opt = Adam(model.pi.params + model.vf.params, lr=model.cfg["learning_rate"])
    return model


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--level", default="mix")
    p.add_argument("--run", default="oht_bc")
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--dagger", type=int, default=1)
    p.add_argument("--dagger-episodes", type=int, default=30)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--ppo-lr", type=float, default=1e-4)
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    env = OHTEnv(a.level)
    rng = np.random.default_rng(a.seed)
    seed0 = 500_000 + a.seed * 10_000
    log(f"collecting {a.episodes} smart-rule episodes ({a.level})")
    X, Y, M, R, D = collect(a.level, a.episodes, seed0)
    G, rms = value_targets(R, D, 0.99)
    model = NumpyPPO(env.observation_space.shape[0], env.action_space.nvec, hidden=tuple(a.hidden),
                     learning_rate=a.ppo_lr, seed=a.seed)
    fit(model, X, Y, M, G, a.epochs, log); model.ret_rms = rms
    for it in range(a.dagger):
        beta = 0.5 / (it + 1)
        log(f"DAgger {it + 1}/{a.dagger}: learner drives with p={1 - beta:.2f}")
        X2, Y2, M2, R2, D2 = collect(a.level, a.dagger_episodes, seed0 + 100_000 * (it + 1), PPOPolicy(model), beta, rng)
        G2, _ = value_targets(R2, D2, 0.99)
        X, Y, M, G = np.concatenate([X, X2]), np.concatenate([Y, Y2]), np.concatenate([M, M2]), np.concatenate([G, G2])
        fit(model, X, Y, M, G, a.epochs, log); model.ret_rms = rms
    ev = OHTEnv(a.level, seed_mode="eval")
    seeds = eval_seeds(a.eval_episodes)
    res = evaluate_policy(ev, PPOPolicy(model), seeds)
    rule = evaluate_policy(ev, SmartRulePolicy(ev), seeds)
    out = MODELS_DIR / a.run
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "best_model"); model.save(out / "final_model")
    meta = {"run": a.run, "level": a.level, "algo": "bc_numpy", "timesteps": 0, "best_eval": res["mean_reward"],
            "samples": int(len(X)), "best_metrics": {k: v for k, v in res.items() if k != "episodes"},
            "teacher_eval": {k: v for k, v in rule.items() if k != "episodes"}}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    log(f"BC {a.run}: eval {res['mean_reward']:.1f} (teacher {rule['mean_reward']:.1f}) avgDel {res['avgDeliveryTime']:.1f}s "
        f"(teacher {rule['avgDeliveryTime']:.1f}s) thr/h {res['throughputPerHour']:.0f} (teacher {rule['throughputPerHour']:.0f})")
    return meta


if __name__ == "__main__":
    main()
