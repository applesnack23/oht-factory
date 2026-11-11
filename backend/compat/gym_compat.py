"""Gymnasium import with a minimal API-compatible fallback.

The project targets the real `gymnasium` package (see requirements.txt). When
it is not installed (e.g. an offline sandbox), this module provides a small
subset with the same API (Env, spaces.Box, spaces.MultiDiscrete,
spaces.Discrete) so the simulation, tests and NumPy PPO still run.
"""
from __future__ import annotations

try:  # pragma: no cover - exercised when gymnasium is installed
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore

    HAVE_GYMNASIUM = True
except Exception:  # pragma: no cover - exercised in offline sandbox
    HAVE_GYMNASIUM = False
    import types

    import numpy as np

    class Space:
        def __init__(self, shape=None, dtype=None, seed=None):
            self.shape = shape
            self.dtype = np.dtype(dtype) if dtype is not None else None
            self._np_random = np.random.default_rng(seed)

        def seed(self, seed=None):
            self._np_random = np.random.default_rng(seed)
            return [seed]

        @property
        def np_random(self):
            return self._np_random

    class Box(Space):
        def __init__(self, low, high, shape=None, dtype=np.float32, seed=None):
            shape = tuple(shape) if shape is not None else np.shape(low)
            super().__init__(shape, dtype, seed)
            self.low = np.full(shape, low, dtype=self.dtype) if np.isscalar(low) else np.asarray(low, dtype=self.dtype)
            self.high = np.full(shape, high, dtype=self.dtype) if np.isscalar(high) else np.asarray(high, dtype=self.dtype)

        def sample(self):
            return self._np_random.uniform(self.low, self.high).astype(self.dtype)

        def contains(self, x):
            x = np.asarray(x)
            return x.shape == self.shape and np.can_cast(x.dtype, self.dtype) and bool(np.all(x >= self.low) and np.all(x <= self.high))

        def __repr__(self):
            return f"Box({self.low.min()}, {self.high.max()}, {self.shape}, {self.dtype})"

    class MultiDiscrete(Space):
        def __init__(self, nvec, dtype=np.int64, seed=None):
            self.nvec = np.asarray(nvec, dtype=np.int64)
            super().__init__(self.nvec.shape, dtype, seed)

        def sample(self):
            return (self._np_random.random(self.nvec.shape) * self.nvec).astype(self.dtype)

        def contains(self, x):
            x = np.asarray(x)
            return x.shape == self.shape and np.issubdtype(x.dtype, np.integer) and bool(np.all(x >= 0) and np.all(x < self.nvec))

        def __repr__(self):
            return f"MultiDiscrete({self.nvec.tolist()})"

    class Discrete(Space):
        def __init__(self, n, seed=None):
            super().__init__((), np.int64, seed)
            self.n = int(n)

        def sample(self):
            return int(self._np_random.integers(self.n))

        def contains(self, x):
            return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

    class Env:
        metadata = {"render_modes": []}
        render_mode = None
        _np_random = None

        @property
        def np_random(self):
            if self._np_random is None:
                self._np_random = np.random.default_rng()
            return self._np_random

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self._np_random = np.random.default_rng(seed)

        def step(self, action):
            raise NotImplementedError

        def close(self):
            pass

        @property
        def unwrapped(self):
            return self

    spaces = types.SimpleNamespace(Box=Box, MultiDiscrete=MultiDiscrete, Discrete=Discrete, Space=Space)
    gym = types.SimpleNamespace(Env=Env, spaces=spaces)

__all__ = ["gym", "spaces", "HAVE_GYMNASIUM"]
