"""Train a PPO agent for OHT dispatch/routing/repositioning (headless).

Examples
  python -m backend.ai.train_ppo --level normal --timesteps 300000 --run l1_ppo
  python -m backend.ai.train_ppo --level normal --timesteps 500000 --init-from oht_bc
  python -m backend.ai.train_ppo --level normal --smoke          # quick sanity run
  python -m backend.ai.train_ppo --resume l1_ppo --timesteps 100000

Backend: Stable-Baselines3 PPO (MlpPolicy) when stable_baselines3+torch are
installed (--algo sb3), otherwise the NumPy PPO implementation (--algo numpy).
Writes: backend/models/<run>/{best_model,final_model}, backend/checkpoints/<run>/,
backend/logs/<run>/{plan.json,progress.csv,status.json,training_curves.png},
and backend/logs/training_status.json (shared with the API server).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from backend.ai.oht_env import OHTEnv as FactoryEnv, check_env, eval_seeds
from backend.ai.inference import MODELS_DIR, PPOPolicy, have_sb3
from backend.ai.rollout import evaluate_policy
from backend.sim.config import PROJECT_ROOT

LOGS_DIR = PROJECT_ROOT / "backend" / "logs"
CKPT_DIR = PROJECT_ROOT / "backend" / "checkpoints"
SHARED_STATUS = LOGS_DIR / "training_status.json"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float))
    os.replace(tmp, path)


class TB:
    """TensorBoard writer if available (torch / tensorboardX), else no-op."""

    def __init__(self, logdir: Path):
        self.w = None
        for mod in ("torch.utils.tensorboard", "tensorboardX"):
            try:
                SummaryWriter = __import__(mod, fromlist=["SummaryWriter"]).SummaryWriter
                self.w = SummaryWriter(str(logdir))
                self.kind = mod
                break
            except Exception:
                continue

    def scalars(self, d: dict, step: int) -> None:
        if self.w is None:
            return
        for k, v in d.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                self.w.add_scalar(k, v, step)
        self.w.flush()


def benchmark_env(level: str, n: int = 200) -> float:
    env = FactoryEnv(level)
    env.reset(seed=0)
    t0 = time.time()
    for _ in range(n):
        _, _, te, tr, _ = env.step(env.action_space.sample())
        if te or tr:
            env.reset()
    return n / max(time.time() - t0, 1e-9)


def episode_stats(ep_infos) -> dict:
    if not ep_infos:
        return {}
    ms = [e["metrics"] for e in ep_infos if "metrics" in e]
    def m(k):
        vals = [x[k] for x in ms if x.get(k) is not None]
        return float(np.mean(vals)) if vals else float("nan")
    return {"rollout/ep_rew_mean": float(np.mean([e["r"] for e in ep_infos])),
            "rollout/ep_len_mean": float(np.mean([e["l"] for e in ep_infos])),
            "fab/throughput_per_hour": m("throughputPerHour"),
            "fab/avg_delivery_time": m("avgDeliveryTime"),
            "fab/p95_delivery_time": m("p95DeliveryTime"),
            "fab/avg_wait_for_vehicle": m("avgWaitForVehicle"),
            "fab/empty_travel_ratio": m("emptyTravelRatio"),
            "fab/blocked_ratio": m("blockedRatio"),
            "fab/hot_on_time_rate": m("hotOnTimeRate"),
            "fab/pending_at_end": m("pending")}


class Trainer:
    def __init__(self, args):
        self.args = args
        self.level = args.level
        self.run_name = args.run
        self.model_dir = MODELS_DIR / self.run_name
        self.ckpt_dir = CKPT_DIR / self.run_name
        self.log_dir = LOGS_DIR / self.run_name
        for d in (self.model_dir, self.ckpt_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.status_path = Path(args.status_file) if args.status_file else self.log_dir / "status.json"
        self.csv_path = self.log_dir / "progress.csv"
        self.tb = TB(self.log_dir / "tb")
        self.eval_env = FactoryEnv(self.level, seed_mode="eval")
        self.eval_seeds = eval_seeds(args.eval_episodes)
        self.best = -float("inf")
        self.best_metrics = {}
        self.log_lines = deque(maxlen=40)
        self.started = time.time()
        self.last_eval_step = 0
        self.last_ckpt_step = 0
        self.last_snapshot_step = None
        self.snapshot_count = 0
        self.last_eval = None
        self.history = []
        self.stop_file = self.log_dir / "STOP"
        if self.stop_file.exists():
            self.stop_file.unlink()

    def log(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        self.log_lines.append(line)
        print(line, flush=True)

    def status(self, state: str, step: int, extra=None) -> None:
        last = self.history[-1] if self.history else {}
        st = {"run": self.run_name, "level": self.level, "algo": self.algo, "state": state, "pid": os.getpid(),
              "step": int(step), "totalSteps": int(self.target), "progress": min(1.0, (step - self.start_step) / max(1, self.target - self.start_step)),
              "meanReward": last.get("rollout/ep_rew_mean"), "bestEvalReward": None if self.best == -float("inf") else self.best,
              "throughput": last.get("fab/throughput_per_hour"), "avgDeliveryTime": last.get("fab/avg_delivery_time"),
              "p95DeliveryTime": last.get("fab/p95_delivery_time"), "elapsed": time.time() - self.started,
              "bestMetrics": self.best_metrics, "log": list(self.log_lines), "updated": time.time(),
              "modelDir": str(self.model_dir)}
        if extra:
            st.update(extra)
        write_json(self.status_path, st)
        write_json(SHARED_STATUS, st)

    def record(self, step: int, row: dict) -> None:
        row = {"timesteps": step, "time": round(time.time() - self.started, 1), **row}
        self.history.append(row)
        new = not self.csv_path.exists()
        keys = list(row.keys())
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
        self.tb.scalars(row, step)

    def maybe_eval(self, model, step: int, save_fn) -> None:
        if step - self.last_eval_step < self.args.eval_freq and step < self.target:
            return
        self.last_eval_step = step
        res = evaluate_policy(self.eval_env, PPOPolicy(model), self.eval_seeds)
        msg = (f"eval@{step}: reward={res['mean_reward']:.1f}±{res['std_reward']:.1f} thr/h={res['throughputPerHour']:.0f} "
               f"avgDel={res['avgDeliveryTime']:.1f}s p95={res['p95DeliveryTime']:.1f}s wait={res['avgWaitForVehicle']:.1f}s")
        self.log(msg)
        self.tb.scalars({"eval/mean_reward": res["mean_reward"], "eval/avg_delivery_time": res["avgDeliveryTime"],
                         "eval/throughput": res["throughputPerHour"]}, step)
        with open(self.log_dir / "evaluations.csv", "a") as f:
            f.write(f"{step},{res['mean_reward']},{res['std_reward']},{res['avgDeliveryTime']},{res['throughputPerHour']},{res['p95DeliveryTime']},{res['avgWaitForVehicle']}\n")
        self.last_eval = {"step": int(step), "reward": res["mean_reward"], "avgDeliveryTime": res["avgDeliveryTime"],
                          "throughputPerHour": res["throughputPerHour"]}
        self.snapshot(model, step, save_fn, evaluated=True)
        if res["mean_reward"] > self.best:  # noqa: best by mean eval reward
            self.best = res["mean_reward"]
            self.best_metrics = {k: v for k, v in res.items() if k != "episodes"}
            save_fn(self.model_dir / "best_model")
            self.log(f"new best model (eval reward {self.best:.1f}) saved")

    def maybe_snapshot(self, model, step: int, save_fn) -> None:
        if self.last_snapshot_step is None or step - self.last_snapshot_step >= self.args.snapshot_freq:
            self.snapshot(model, step, save_fn)

    def snapshot(self, model, step: int, save_fn, evaluated: bool = False) -> None:
        """Write models/<run>/latest_model.{npz,zip} + latest_model.json atomically.

        The live server's "watch training" mode reloads this file whenever it
        changes, so the 3D factory shows the policy as it improves without
        slowing training down.
        """
        tmp = Path(save_fn(self.model_dir / "latest_tmp"))
        final = self.model_dir / ("latest_model" + tmp.suffix)
        os.replace(tmp, final)
        self.last_snapshot_step = step
        self.snapshot_count += 1
        write_json(self.model_dir / "latest_model.json", {
            "run": self.run_name, "level": self.level, "algo": self.algo, "step": int(step),
            "totalSteps": int(self.target), "snapshot": self.snapshot_count, "file": final.name,
            "evaluated": evaluated, "lastEval": self.last_eval, "time": time.time()})

    def maybe_ckpt(self, step: int, save_fn) -> None:
        if step - self.last_ckpt_step >= self.args.save_freq:
            self.last_ckpt_step = step
            p = save_fn(self.ckpt_dir / f"factory_ppo_{step}_steps")
            self.log(f"checkpoint {p}")

    # ---------------------------------------------------------------- numpy
    def train_numpy(self):
        from backend.ai.numpy_ppo import NumpyPPO
        a = self.args
        envs = [FactoryEnv(self.level, seed_mode="train") for _ in range(a.n_envs)]
        for i, e in enumerate(envs):
            e.reset(seed=a.seed * 1000 + i)
        hp = dict(learning_rate=a.lr, n_steps=a.n_steps, batch_size=a.batch_size, n_epochs=a.n_epochs,
                  gamma=a.gamma, gae_lambda=a.gae_lambda, clip_range=a.clip_range, ent_coef=a.ent_coef,
                  vf_coef=a.vf_coef, max_grad_norm=a.max_grad_norm, hidden=tuple(a.hidden), seed=a.seed)
        if a.resume:
            model = NumpyPPO.load(self._resolve_ckpt(a.resume), learning_rate=a.lr)
            self.log(f"resumed from {a.resume} at {model.num_timesteps} steps")
        else:
            model = NumpyPPO(envs[0].observation_space.shape[0], envs[0].action_space.nvec, **hp)
            if a.init_from:
                src = NumpyPPO.load(self._resolve_ckpt(a.init_from))
                for d, s in zip(model.pi.params + model.vf.params, src.pi.params + src.vf.params):
                    d[...] = s
                model.ret_rms = src.ret_rms
                self.log(f"initialized weights from {a.init_from} (curriculum transfer)")
        self.start_step = model.num_timesteps
        self.target = self.start_step + a.timesteps
        save = lambda p: model.save(p)
        recent = deque(maxlen=20)

        def log_fn(mdl):
            while mdl.ep_infos:
                recent.append(mdl.ep_infos.pop(0))
            row = episode_stats(list(recent))
            tl = mdl.train_log
            row.update({"train/policy_loss": tl.get("policy_loss"), "train/value_loss": tl.get("value_loss"),
                        "train/entropy": tl.get("entropy"), "train/approx_kl": tl.get("approx_kl"),
                        "train/clip_fraction": tl.get("clip_fraction"), "train/explained_variance": tl.get("explained_variance"),
                        "time/fps": tl.get("fps")})
            self.record(mdl.num_timesteps, row)
            if mdl.n_updates % 5 == 0 or mdl.num_timesteps >= self.target:
                self.log(f"step {mdl.num_timesteps}/{self.target} ep_rew_mean={row.get('rollout/ep_rew_mean', float('nan')):.1f} "
                         f"avgDel={row.get('fab/avg_delivery_time', float('nan')):.1f}s ent={tl.get('entropy', 0):.3f} fps={tl.get('fps', 0):.0f}")

        def cb(mdl):
            self.maybe_eval(mdl, mdl.num_timesteps, save)
            self.maybe_ckpt(mdl.num_timesteps, save)
            self.maybe_snapshot(mdl, mdl.num_timesteps, save)
            self.status("running", mdl.num_timesteps)
            if self.stop_file.exists():
                self.log("stop requested")
                return False
            return True

        self.status("running", self.start_step)
        self.snapshot(model, self.start_step, save)  # the untrained/initial policy, for comparison
        model.learn(envs, a.timesteps, callback=cb, log_fn=log_fn)
        if self.last_eval_step != model.num_timesteps:
            self.last_eval_step = -10 ** 12
            self.maybe_eval(model, model.num_timesteps, save)
        model.save(self.model_dir / "final_model")
        return model, model.num_timesteps

    # ------------------------------------------------------------------ sb3
    def train_sb3(self):  # pragma: no cover - requires stable_baselines3 + torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        a = self.args
        trainer = self

        def make(i):
            def _f():
                e = FactoryEnv(self.level, seed_mode="train")
                e.reset(seed=a.seed * 1000 + i)
                return Monitor(e)
            return _f

        venv = VecNormalize(DummyVecEnv([make(i) for i in range(a.n_envs)]), norm_obs=False, norm_reward=True, gamma=a.gamma)
        if a.resume:
            model = PPO.load(str(self._resolve_ckpt(a.resume, ".zip")), env=venv, device="cpu")
        else:
            model = PPO("MlpPolicy", venv, learning_rate=a.lr, n_steps=a.n_steps, batch_size=a.batch_size,
                        n_epochs=a.n_epochs, gamma=a.gamma, gae_lambda=a.gae_lambda, clip_range=a.clip_range,
                        ent_coef=a.ent_coef, vf_coef=a.vf_coef, max_grad_norm=a.max_grad_norm, seed=a.seed,
                        policy_kwargs={"net_arch": list(a.hidden)}, tensorboard_log=str(self.log_dir / "tb"),
                        verbose=0, device=a.device)
            if a.init_from:
                src = PPO.load(str(self._resolve_ckpt(a.init_from, ".zip")), device="cpu")
                model.policy.load_state_dict(src.policy.state_dict())
        self.start_step = model.num_timesteps
        self.target = self.start_step + a.timesteps
        save = lambda p: (model.save(str(p)), str(p) + ".zip")[1]

        class CB(BaseCallback):
            def __init__(self):
                super().__init__()
                self.recent = deque(maxlen=20)

            def _on_step(self):
                for info, done in zip(self.locals["infos"], self.locals["dones"]):
                    if done and "metrics" in info:
                        self.recent.append({"r": info.get("episode", {}).get("r", 0.0), "l": info.get("episode", {}).get("l", 0),
                                            "metrics": info["metrics"]})
                for k, v in episode_stats(list(self.recent)).items():
                    if k.startswith("fab/"):
                        self.logger.record(k, v)
                trainer.maybe_ckpt(self.num_timesteps, save)
                return not trainer.stop_file.exists()

            def _on_rollout_end(self):
                row = episode_stats(list(self.recent))
                trainer.record(self.num_timesteps, row)
                trainer.maybe_eval(self.model, self.num_timesteps, save)
                trainer.maybe_snapshot(self.model, self.num_timesteps, save)
                trainer.status("running", self.num_timesteps)

        self.snapshot(model, self.start_step, save)
        model.learn(total_timesteps=a.timesteps, callback=CB(), reset_num_timesteps=not a.resume)
        model.save(str(self.model_dir / "final_model"))
        return model, model.num_timesteps

    def _resolve_ckpt(self, ref, ext=".npz"):
        p = Path(ref)
        cands = [p, p.with_suffix(ext), MODELS_DIR / ref / f"final_model{ext}", MODELS_DIR / ref / f"best_model{ext}"]
        ck = CKPT_DIR / ref
        if ck.is_dir():
            files = sorted(ck.glob(f"*{ext}"), key=lambda x: int(x.stem.split("_")[-2]) if x.stem.split("_")[-2].isdigit() else 0)
            if files:
                cands.insert(0, files[-1])
        for c in cands:
            if c.exists() and c.is_file():
                return c
        raise FileNotFoundError(f"cannot resolve model/checkpoint '{ref}'")

    def run(self):
        a = self.args
        self.algo = a.algo if a.algo != "auto" else ("sb3" if have_sb3() else "numpy")
        self.target = a.timesteps
        self.start_step = 0
        checker = check_env(FactoryEnv(self.level))
        fps = benchmark_env(self.level)
        plan = {"run": self.run_name, "level": self.level, "algo": self.algo, "timesteps": a.timesteps,
                "envCheck": checker, "envStepsPerSecond": round(fps, 1),
                "estimatedMinutes": round(a.timesteps / max(fps * 0.6, 1e-9) / 60.0, 1),
                "hyperparameters": {k: v for k, v in vars(a).items() if k not in ("status_file",)},
                "evalSeeds": self.eval_seeds, "started": datetime.now().isoformat()}
        write_json(self.log_dir / "plan.json", plan)
        self.log(f"training plan: {json.dumps({k: plan[k] for k in ('level', 'algo', 'timesteps', 'envCheck', 'envStepsPerSecond', 'estimatedMinutes')})}")
        try:
            model, steps = self.train_sb3() if self.algo == "sb3" else self.train_numpy()
        except Exception as e:
            self.log(f"training failed: {e!r}")
            self.status("failed", getattr(self, "start_step", 0), {"error": repr(e)})
            raise
        meta = {"run": self.run_name, "level": self.level, "algo": self.algo, "timesteps": steps,
                "best_eval": self.best, "best_metrics": self.best_metrics, "finished": datetime.now().isoformat(),
                "hyperparameters": plan["hyperparameters"]}
        write_json(self.model_dir / "run_meta.json", meta)
        self.plot()
        self.status("finished", steps)
        self.log(f"done: {steps} steps, best eval reward {self.best:.1f}")
        return meta

    def plot(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return
        if not self.history:
            return
        h = self.history
        x = [r["timesteps"] for r in h]
        fig, axs = plt.subplots(2, 2, figsize=(11, 7))
        for ax, (k, title) in zip(axs.flat, [("rollout/ep_rew_mean", "Episode reward (train)"),
                                              ("fab/avg_delivery_time", "Avg delivery time (s)"),
                                              ("fab/throughput_per_hour", "Moves / h"),
                                              ("train/entropy", "Policy entropy")]):
            ax.plot(x, [r.get(k, np.nan) if r.get(k) is not None else np.nan for r in h])
            ax.set_title(title)
            ax.set_xlabel("timesteps")
            ax.grid(alpha=0.3)
        ev = self.log_dir / "evaluations.csv"
        if ev.exists():
            rows = [list(map(float, l.split(","))) for l in ev.read_text().strip().splitlines()]
            axs[0, 0].plot([r[0] for r in rows], [r[1] for r in rows], "o-", label="eval (deterministic)")
            axs[0, 0].legend()
        fig.suptitle(f"{self.run_name} ({self.level}, {self.algo})")
        fig.tight_layout()
        fig.savefig(self.log_dir / "training_curves.png", dpi=110)
        plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", default="normal", help="scenario name (config/scenarios.json)")
    p.add_argument("--run", default=None)
    p.add_argument("--algo", choices=["auto", "sb3", "numpy"], default="auto")
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=512, help="rollout steps per env per update")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--eval-freq", type=int, default=20_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--save-freq", type=int, default=50_000)
    p.add_argument("--snapshot-freq", type=int, default=10_000,
                   help="save latest_model for the live 'watch training' view every N steps")
    p.add_argument("--resume", default=None, help="run name or checkpoint path to resume from")
    p.add_argument("--init-from", default=None, help="initialize weights from another run (curriculum)")
    p.add_argument("--device", default="cpu", help="sb3 only: cpu | mps | cuda")
    p.add_argument("--status-file", default=None)
    p.add_argument("--config", default=None, help="JSON file with argument overrides")
    p.add_argument("--smoke", action="store_true", help="tiny run to verify the pipeline")
    a = p.parse_args(argv)
    if a.config:
        for k, v in json.loads(Path(a.config).read_text()).items():
            setattr(a, k.replace("-", "_"), v)
    if a.smoke:
        a.timesteps, a.n_steps, a.n_envs, a.eval_freq, a.eval_episodes, a.save_freq = 1024, 128, 2, 512, 1, 512
        a.snapshot_freq = 256
    if a.run is None:
        a.run = f"oht_{a.level}_{'smoke' if a.smoke else 'ppo'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if a.resume and a.run is None:
        a.run = Path(a.resume).name
    return a


def main(argv=None):
    args = parse_args(argv)
    return Trainer(args).run()


if __name__ == "__main__":
    main()
