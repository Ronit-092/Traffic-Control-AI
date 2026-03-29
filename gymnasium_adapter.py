"""
gymnasium_adapter.py — Thin Gymnasium wrapper for PPO training.
Wraps TrafficEnvironment so stable-baselines3 can train on it.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from models import TrafficAction
from server.environment import TrafficEnvironment, MAX_QUEUE, DIRECTIONS


class TrafficGymnasiumEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        self._core = TrafficEnvironment()
        self.action_space      = spaces.Discrete(4)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._core.reset(seed=seed)
        return np.array(obs.obs_normalized, dtype=np.float32), {}

    def step(self, action: int):
        obs = self._core.step(TrafficAction(action_id=int(action)))
        return (
            np.array(obs.obs_normalized, dtype=np.float32),
            obs.reward or 0.0,
            obs.done,
            False,
            {"emergency": any(obs.emergency.values()), "junction": obs.junction_type},
        )

    def close(self):
        pass
