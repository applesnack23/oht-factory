"""PPO for MultiDiscrete action spaces implemented with NumPy only.

Used when Stable-Baselines3/PyTorch are unavailable. The algorithm follows
SB3's PPO (clipped surrogate objective, GAE(lambda), value loss, entropy
bonus, global gradient-norm clipping, per-minibatch advantage normalization,
running return-based reward scaling like VecNormalize(norm_reward=True)).
Gradients are computed analytically (verified by finite differences in
backend/tests/test_ppo.py).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


# ----------------------------------------------------------------------------
class MLP:
    """tanh MLP with linear output; manual forward/backward."""

    def __init__(self, sizes: Sequence[int], rng: np.random.Generator, out_gain: float = 1.0):
        self.sizes = list(sizes)
        self.params: List[np.ndarray] = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            gain = out_gain if i == len(sizes) - 2 else np.sqrt(2.0)
            self.params.append(self._orthogonal((a, b), gain, rng))
            self.params.append(np.zeros(b, dtype=np.float64))

    @staticmethod
    def _orthogonal(shape, gain, rng):
        a = rng.standard_normal(shape)
        u, _, vt = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == shape else vt
        return (gain * q.reshape(shape)).astype(np.float64)

    def forward(self, x: np.ndarray, cache: bool = False):
        hs = [x]
        h = x
        n = len(self.params) // 2
        for i in range(n):
            w, b = self.params[2 * i], self.params[2 * i + 1]
            h = h @ w + b
            if i < n - 1:
                h = np.tanh(h)
            hs.append(h)
        if cache:
            self._hs = hs
        return h

    def backward(self, dout: np.ndarray) -> List[np.ndarray]:
        hs = self._hs
        n = len(self.params) // 2
        grads: List[np.ndarray] = [None] * len(self.params)  # type: ignore
        d = dout
        for i in reversed(range(n)):
            w = self.params[2 * i]
            if i < n - 1:
                d = d * (1.0 - hs[i + 1] ** 2)  # tanh'
            grads[2 * i] = hs[i].T @ d
            grads[2 * i + 1] = d.sum(axis=0)
            if i > 0:
                d = d @ w.T
        return grads


class Adam:
    def __init__(self, params: List[np.ndarray], lr: float = 3e-4, b1=0.9, b2=0.999, eps=1e-5):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: List[np.ndarray]) -> None:
        self.t += 1
        b1t = 1 - self.b1 ** self.t
        b2t = 1 - self.b2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= self.lr * (m / b1t) / (np.sqrt(v / b2t) + self.eps)


class RunningMeanStd:
    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        bm, bv, bc = x.mean(), x.var(), x.size
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot


# ----------------------------------------------------------------------------
def softmax_heads(logits: np.ndarray, splits: List[slice], mask: Optional[np.ndarray] = None):
    probs, logps = [], []
    if mask is not None:
        logits = np.where(mask, logits, -1e9)
    for s in splits:
        z = logits[:, s]
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        probs.append(p)
        logps.append(np.log(np.clip(p, 1e-12, 1.0)))
    return probs, logps


DEFAULTS = dict(learning_rate=3e-4, n_steps=512, batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, hidden=(64, 64),
                normalize_reward=True, seed=0)


class NumpyPPO:
    algo = "numpy_ppo"

    def __init__(self, obs_dim: int, nvec: Sequence[int], **kw):
        self.cfg = {**DEFAULTS, **kw}
        self.cfg["hidden"] = tuple(self.cfg["hidden"])
        self.obs_dim = int(obs_dim)
        self.nvec = [int(n) for n in nvec]
        self.rng = np.random.default_rng(self.cfg["seed"])
        offs = np.cumsum([0] + self.nvec)
        self.splits = [slice(int(a), int(b)) for a, b in zip(offs[:-1], offs[1:])]
        h = list(self.cfg["hidden"])
        self.pi = MLP([self.obs_dim] + h + [int(offs[-1])], self.rng, out_gain=0.01)
        self.vf = MLP([self.obs_dim] + h + [1], self.rng, out_gain=1.0)
        self.opt = Adam(self.pi.params + self.vf.params, lr=self.cfg["learning_rate"])
        self.ret_rms = RunningMeanStd()
        self.num_timesteps = 0
        self.n_updates = 0
        self.train_log: Dict[str, float] = {}

    # ------------------------------------------------------------- policy
    def _mask(self, mask, B):
        if mask is None:
            return None
        m = np.asarray(mask, dtype=bool)
        return np.broadcast_to(m, (B, m.shape[-1])) if m.ndim == 1 else m

    def _forward(self, obs: np.ndarray):
        obs = np.asarray(obs, dtype=np.float64)
        if obs.ndim == 1:
            obs = obs[None]
        logits = self.pi.forward(obs)
        value = self.vf.forward(obs)[:, 0]
        return logits, value

    def act(self, obs: np.ndarray, deterministic: bool = False, mask: Optional[np.ndarray] = None):
        logits, value = self._forward(obs)
        probs, logps = softmax_heads(logits, self.splits, self._mask(mask, logits.shape[0]))
        B = logits.shape[0]
        acts = np.zeros((B, len(self.nvec)), dtype=np.int64)
        logp = np.zeros(B)
        for h, (p, lp) in enumerate(zip(probs, logps)):
            if deterministic:
                a = p.argmax(axis=1)
            else:
                u = self.rng.random((B, 1))
                a = (p.cumsum(axis=1) < u).sum(axis=1).clip(0, p.shape[1] - 1)
            acts[:, h] = a
            logp += lp[np.arange(B), a]
        return acts, logp, value

    def predict(self, observation, state=None, episode_start=None, deterministic: bool = True, action_masks=None):
        """SB3-compatible predict() (sb3-contrib style action_masks optional)."""
        obs = np.asarray(observation)
        single = obs.ndim == 1
        acts, _, _ = self.act(obs, deterministic=deterministic, mask=action_masks)
        return (acts[0] if single else acts), None

    def evaluate(self, obs, actions, mask=None):
        logits = self.pi.forward(obs, cache=True)
        value = self.vf.forward(obs, cache=True)[:, 0]
        probs, logps = softmax_heads(logits, self.splits, mask)
        B = obs.shape[0]
        logp = np.zeros(B)
        ent = np.zeros(B)
        for h, (p, lp) in enumerate(zip(probs, logps)):
            logp += lp[np.arange(B), actions[:, h]]
            ent += -(p * np.where(p > 0, lp, 0.0)).sum(axis=1)
        return logits, probs, logps, logp, ent, value

    # --------------------------------------------------------------- loss
    def loss_and_grads(self, obs, actions, old_logp, adv, returns, mask=None):
        c = self.cfg
        B = obs.shape[0]
        logits, probs, logps, logp, ent, value = self.evaluate(obs, actions, mask)
        ratio = np.exp(logp - old_logp)
        eps = c["clip_range"]
        s1 = ratio * adv
        s2 = np.clip(ratio, 1 - eps, 1 + eps) * adv
        pg_loss = -np.minimum(s1, s2).mean()
        ent_loss = -ent.mean()
        v_loss = ((returns - value) ** 2).mean()
        loss = pg_loss + c["ent_coef"] * ent_loss + c["vf_coef"] * v_loss
        # d pg / d logp
        active = s1 <= s2  # gradient flows through the unclipped term
        dlogp = np.where(active, -adv * ratio, 0.0) / B
        dlogits = np.zeros_like(logits)
        for h, (s, p, lp) in enumerate(zip(self.splits, probs, logps)):
            onehot = np.zeros_like(p)
            onehot[np.arange(B), actions[:, h]] = 1.0
            g = dlogp[:, None] * (onehot - p)
            lpz = np.where(p > 0, lp, 0.0)
            Hh = -(p * lpz).sum(axis=1, keepdims=True)
            dH = -p * (lpz + Hh)
            g += -c["ent_coef"] / B * dH
            dlogits[:, s] = g
        gpi = self.pi.backward(dlogits)
        dv = c["vf_coef"] * 2.0 * (value - returns) / B
        gvf = self.vf.backward(dv[:, None])
        info = {"policy_loss": float(pg_loss), "value_loss": float(v_loss), "entropy": float(ent.mean()),
                "approx_kl": float(((ratio - 1) - np.log(ratio)).mean()),
                "clip_fraction": float((np.abs(ratio - 1) > eps).mean()), "loss": float(loss)}
        return loss, gpi + gvf, info

    def _apply(self, grads):
        norm = np.sqrt(sum(float((g * g).sum()) for g in grads))
        mg = self.cfg["max_grad_norm"]
        if mg and norm > mg:
            grads = [g * (mg / (norm + 1e-8)) for g in grads]
        self.opt.step(grads)
        return norm

    # -------------------------------------------------------------- learn
    def learn(self, envs: List, total_timesteps: int, callback: Optional[Callable] = None,
              reset_num_timesteps: bool = False, log_fn: Optional[Callable] = None):
        c = self.cfg
        n_env = len(envs)
        T = c["n_steps"]
        if not hasattr(self, "_obs") or self._obs is None or reset_num_timesteps:
            self._obs = np.stack([e.reset()[0] for e in envs]).astype(np.float64)
            self._ret = np.zeros(n_env)
            self._ep_rew = np.zeros(n_env)
            self._ep_len = np.zeros(n_env, dtype=int)
        self.ep_infos: List[dict] = getattr(self, "ep_infos", [])
        target = self.num_timesteps + total_timesteps
        while self.num_timesteps < target:
            t0 = time.time()
            ob_buf = np.zeros((T, n_env, self.obs_dim))
            ac_buf = np.zeros((T, n_env, len(self.nvec)), dtype=np.int64)
            lp_buf = np.zeros((T, n_env))
            v_buf = np.zeros((T, n_env))
            r_buf = np.zeros((T, n_env))
            d_buf = np.zeros((T, n_env))
            use_mask = hasattr(envs[0], "action_masks")
            m_buf = np.ones((T, n_env, int(sum(self.nvec))), dtype=bool)
            for t in range(T):
                mask = np.stack([e.action_masks() for e in envs]) if use_mask else None
                if use_mask:
                    m_buf[t] = mask
                acts, logp, val = self.act(self._obs, mask=mask)
                ob_buf[t], ac_buf[t], lp_buf[t], v_buf[t] = self._obs, acts, logp, val
                next_obs = np.zeros_like(self._obs)
                for i, e in enumerate(envs):
                    o, r, term, trunc, info = e.step(acts[i])
                    self._ep_rew[i] += r
                    self._ep_len[i] += 1
                    if c["normalize_reward"]:
                        self._ret[i] = self._ret[i] * c["gamma"] + r
                    r_buf[t, i] = r
                    if term or trunc:
                        if trunc and not term:  # bootstrap time-limit truncation
                            _, vt = self._forward(o)
                            r_buf[t, i] += c["gamma"] * float(vt[0]) * self._reward_std_or_one()
                        d_buf[t, i] = 1.0
                        ep = {"r": float(self._ep_rew[i]), "l": int(self._ep_len[i])}
                        if "metrics" in info:
                            ep["metrics"] = info["metrics"]
                        if "episode_reward_info" in info:
                            ep["reward_info"] = info["episode_reward_info"]
                        self.ep_infos.append(ep)
                        self._ep_rew[i] = 0.0
                        self._ep_len[i] = 0
                        o, _ = e.reset()
                    next_obs[i] = o
                if c["normalize_reward"]:
                    self.ret_rms.update(self._ret)
                    self._ret[d_buf[t] > 0] = 0.0
                self._obs = next_obs
                self.num_timesteps += n_env
            # reward scaling
            scale = self._reward_std_or_one()
            r_scaled = np.clip(r_buf / scale, -10.0, 10.0) if c["normalize_reward"] else r_buf
            _, last_v = self._forward(self._obs)
            adv = np.zeros((T, n_env))
            gae = np.zeros(n_env)
            for t in reversed(range(T)):
                nv = last_v if t == T - 1 else v_buf[t + 1]
                nonterm = 1.0 - d_buf[t]
                delta = r_scaled[t] + c["gamma"] * nv * nonterm - v_buf[t]
                gae = delta + c["gamma"] * c["gae_lambda"] * nonterm * gae
                adv[t] = gae
            ret = adv + v_buf
            N = T * n_env
            b_obs = ob_buf.reshape(N, -1)
            b_act = ac_buf.reshape(N, -1)
            b_lp, b_adv, b_ret = lp_buf.reshape(N), adv.reshape(N), ret.reshape(N)
            b_mask = m_buf.reshape(N, -1) if use_mask else None
            infos = []
            for _ in range(c["n_epochs"]):
                idx = self.rng.permutation(N)
                for s in range(0, N, c["batch_size"]):
                    mb = idx[s:s + c["batch_size"]]
                    a = b_adv[mb]
                    a = (a - a.mean()) / (a.std() + 1e-8)
                    _, grads, info = self.loss_and_grads(b_obs[mb], b_act[mb], b_lp[mb], a, b_ret[mb],
                                                         None if b_mask is None else b_mask[mb])
                    info["grad_norm"] = float(self._apply(grads))
                    infos.append(info)
            self.n_updates += 1
            var_y = np.var(b_ret)
            ev = float(1 - np.var(b_ret - v_buf.reshape(N)) / var_y) if var_y > 0 else 0.0
            self.train_log = {k: float(np.mean([i[k] for i in infos])) for k in infos[0]}
            self.train_log.update({"explained_variance": ev, "fps": N / max(time.time() - t0, 1e-9),
                                   "n_updates": self.n_updates, "reward_scale": float(scale)})
            if log_fn:
                log_fn(self)
            if callback is not None and callback(self) is False:
                break
        return self

    def _reward_std_or_one(self) -> float:
        if not self.cfg["normalize_reward"]:
            return 1.0
        return float(np.sqrt(self.ret_rms.var + 1e-8))

    # ------------------------------------------------------------ persist
    def save(self, path) -> str:
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"pi_{i}": p for i, p in enumerate(self.pi.params)}
        arrays.update({f"vf_{i}": p for i, p in enumerate(self.vf.params)})
        arrays.update({f"adam_m_{i}": m for i, m in enumerate(self.opt.m)})
        arrays.update({f"adam_v_{i}": v for i, v in enumerate(self.opt.v)})
        meta = {"algo": self.algo, "obs_dim": self.obs_dim, "nvec": self.nvec, "cfg": {**self.cfg, "hidden": list(self.cfg["hidden"])},
                "num_timesteps": self.num_timesteps, "n_updates": self.n_updates, "adam_t": self.opt.t,
                "ret_rms": [self.ret_rms.mean, self.ret_rms.var, self.ret_rms.count]}
        arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
        np.savez(path, **arrays)
        return str(path)

    @classmethod
    def load(cls, path, **override) -> "NumpyPPO":
        path = Path(path)
        if not path.exists() and path.with_suffix(".npz").exists():
            path = path.with_suffix(".npz")
        data = np.load(path)
        meta = json.loads(bytes(data["meta"]).decode())
        cfg = {**meta["cfg"], **override}
        model = cls(meta["obs_dim"], meta["nvec"], **cfg)
        for i in range(len(model.pi.params)):
            model.pi.params[i][...] = data[f"pi_{i}"]
        for i in range(len(model.vf.params)):
            model.vf.params[i][...] = data[f"vf_{i}"]
        for i in range(len(model.opt.m)):
            if f"adam_m_{i}" in data:
                model.opt.m[i][...] = data[f"adam_m_{i}"]
                model.opt.v[i][...] = data[f"adam_v_{i}"]
        model.opt.t = meta.get("adam_t", 0)
        model.num_timesteps = meta.get("num_timesteps", 0)
        model.n_updates = meta.get("n_updates", 0)
        m, v, c = meta.get("ret_rms", [0.0, 1.0, 1e-4])
        model.ret_rms.mean, model.ret_rms.var, model.ret_rms.count = m, v, c
        model.opt.lr = cfg["learning_rate"]
        return model
