"""
server/environment.py — Traffic signal control game logic.

Fixes:
  - Emergency vehicle always adds 1 to the lane count so it shows visually
  - Waiting penalty reduced to 0.1 per vehicle (was 0.5 — too harsh)
  - Emergency miss penalty reduced to 10 (was 20)
  - Emergency served bonus stays at 50
"""

import random
import uuid
from typing import Optional

from openenv.core.env_server import Environment
from models import TrafficAction, TrafficObservation, TrafficState


MAX_QUEUE   = 30
MAX_ARRIVAL = 6
EMERGENCY_P = 0.25

ACTION_MAP = {
    "cross": {0: ["north","south"], 1: ["east","west"], 2: ["north"], 3: ["east"]},
    "T":     {0: ["north","south"], 1: ["east"],        2: ["north"], 3: ["south"]},
    "Y":     {0: ["north"],         1: ["east"],         2: ["west"],  3: ["north","east"]},
}

ACTIVE_LANES = {
    "cross": ["north","south","east","west"],
    "T":     ["north","south","east"],
    "Y":     ["north","east","west"],
}

DIRECTIONS = ["north","south","east","west"]

ACTION_LABELS = {
    "cross": {0:"NS green", 1:"EW green",   2:"North only",  3:"East only"},
    "T":     {0:"NS green", 1:"East green", 2:"North only",  3:"South only"},
    "Y":     {0:"North",    1:"East",       2:"West",        3:"North+East"},
}


class TrafficEnvironment(Environment):
    """Traffic signal control environment."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state        = TrafficState()
        self._junction     = "cross"
        self._traffic      = {d: 0 for d in DIRECTIONS}
        self._emergency    = {d: False for d in DIRECTIONS}
        self._total_reward = 0.0
        self._em_total     = 0
        self._em_served    = 0

    def reset(self, seed: Optional[int] = None,
              episode_id: Optional[str] = None, **kwargs) -> TrafficObservation:
        if seed is not None:
            random.seed(seed)

        self._junction     = random.choice(["cross", "T", "Y"])
        self._traffic      = {d: 0 for d in DIRECTIONS}
        self._emergency    = {d: False for d in DIRECTIONS}
        self._total_reward = 0.0
        self._em_total     = 0
        self._em_served    = 0
        self._state        = TrafficState(
            episode_id    = episode_id or str(uuid.uuid4()),
            step_count    = 0,
            junction_type = self._junction,
        )

        for lane in ACTIVE_LANES[self._junction]:
            self._traffic[lane] = random.randint(2, 10)

        return self._make_obs(reward=None, done=False,
                              msg=f"New episode — {self._junction} junction")

    def step(self, action: TrafficAction,
             timeout_s: Optional[float] = None, **kwargs) -> TrafficObservation:
        act = int(action.action_id)
        self._state.step_count += 1

        # ── Emergency event ────────────────────────────────────────────
        # Pick any active lane (or any lane if all empty)
        candidates = [d for d, v in self._traffic.items()
                      if v > 0 and d in ACTIVE_LANES[self._junction]]
        if not candidates:
            candidates = ACTIVE_LANES[self._junction]

        self._emergency = {d: False for d in DIRECTIONS}
        if random.random() < EMERGENCY_P:
            em_dir = random.choice(candidates)
            self._emergency[em_dir] = True
            # FIX: ensure emergency lane shows at least 1 vehicle
            if self._traffic[em_dir] == 0:
                self._traffic[em_dir] = 1

        # ── Clear served lanes ─────────────────────────────────────────
        served  = ACTION_MAP[self._junction].get(act, [])
        cleared = 0
        for d in served:
            cleared += self._traffic[d]
            self._traffic[d] = 0

        reward = float(cleared)

        # ── Emergency bonus / penalty ──────────────────────────────────
        em_lanes      = [d for d, v in self._emergency.items() if v]
        em_served_now = False
        if em_lanes:
            self._em_total += 1
            if em_lanes[0] in served:
                reward       += 50.0   # strong bonus for serving emergency
                em_served_now = True
                self._em_served += 1
            else:
                reward -= 10.0         # FIX: reduced from 20 to 10

        # ── Queue waiting penalty (reduced from 0.5 to 0.1) ───────────
        total_waiting  = sum(self._traffic.values())
        reward        -= 0.1 * total_waiting   # FIX: was 0.5

        # ── New arrivals ───────────────────────────────────────────────
        for lane in ACTIVE_LANES[self._junction]:
            if self._traffic[lane] > 0:
                self._traffic[lane] += random.randint(0, MAX_ARRIVAL)
            elif random.random() < 0.4:
                self._traffic[lane] += random.randint(1, 3)
            self._traffic[lane] = min(self._traffic[lane], MAX_QUEUE)

        self._total_reward       += reward
        self._state.total_reward  = self._total_reward
        self._state.em_total      = self._em_total
        self._state.em_served     = self._em_served

        label  = ACTION_LABELS[self._junction].get(act, str(act))
        em_str = ""
        if em_lanes:
            em_str = f" | 🚑 {'SERVED +50' if em_served_now else 'MISSED -10'}"
        msg = f"Step {self._state.step_count}: {label} — cleared {cleared}{em_str}"

        return self._make_obs(reward=reward, done=False, msg=msg)

    @property
    def state(self) -> TrafficState:
        return self._state

    def _make_obs(self, reward, done, msg) -> TrafficObservation:
        return TrafficObservation(
            done           = done,
            reward         = reward,
            traffic_counts = dict(self._traffic),
            obs_normalized = [self._traffic[d] / MAX_QUEUE for d in DIRECTIONS],
            emergency      = dict(self._emergency),
            junction_type  = self._junction,
            message        = msg,
        )