import numpy as np
import pytest

from backend.ai.oht_env import C, EVAL_SEED_BASE, F, G, TYPES, OHTEnv, check_env, check_env_lite, eval_seeds
from backend.ai.rollout import run_episode
from backend.policies.rules import BasicRulePolicy, RandomPolicy, SmartRulePolicy


def test_check_env():
    assert check_env_lite(OHTEnv(None)) == "lite"
    assert check_env(OHTEnv(None)) in ("lite", "sb3")


def test_spaces_fixed_and_masks():
    e = OHTEnv(None)
    assert e.observation_space.shape == (G + len(TYPES) + C + e.K * F,)
    assert e.action_space.nvec.tolist() == [e.K + 1]
    o, _ = e.reset(seed=3)
    for _ in range(300):
        m = e.action_masks()
        assert m[e.K]  # defer/default always valid
        n = len(e.decision.candidates)
        assert m[:n].all() and not m[n:e.K].any()
        cb = G + len(TYPES) + C
        assert (o[cb:cb + e.K * F].reshape(e.K, F)[:, 0] > 0.5).sum() == n  # candidate validity flags
        o, r, te, tr, _ = e.step(np.array([0]))
        assert np.isfinite(o).all()


def test_invalid_action_falls_back_and_is_penalized():
    e = OHTEnv(None)
    e.reset(seed=1)
    while len(e.decision.candidates) >= e.K:
        e.step(np.array([0]))
    _, r, _, _, info = e.step(np.array([e.K - 1]))
    assert info["reward_info"]["invalid"] == -1.0


def test_episode_reward_matches_components():
    e = OHTEnv(None, seed_mode="eval")
    ep = run_episode(e, SmartRulePolicy(e), eval_seeds(1)[0])
    assert ep["reward"] == pytest.approx(sum(ep["reward_info"].values()))
    assert ep["reward_info"]["delivery"] == pytest.approx(10.0 * ep["delivered"])


def test_rules_beat_random_and_mix_mode():
    e = OHTEnv(None, seed_mode="eval")
    s = eval_seeds(1)[0]
    r_rand = run_episode(e, RandomPolicy(e), s)["reward"]
    r_basic = run_episode(e, BasicRulePolicy(e), s)["reward"]
    r_smart = run_episode(e, SmartRulePolicy(e), s)["reward"]
    assert r_basic > r_rand and r_smart > r_rand
    m = OHTEnv("mix")
    seen = {m.reset(seed=k)[1]["scenario"] for k in range(12)}
    assert len(seen) >= 3  # scenarios are sampled per episode


def test_seed_separation():
    e = OHTEnv(None)
    with pytest.raises(ValueError):
        e.reset(seed=EVAL_SEED_BASE + 1)
    assert all(s >= EVAL_SEED_BASE for s in eval_seeds(5))
