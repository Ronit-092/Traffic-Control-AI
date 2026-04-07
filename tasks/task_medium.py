"""tasks/task_medium.py — Emergency-aware weighted scorer (15 steps)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import TrafficAction, TrafficObservation
from server.environment import TrafficEnvironment, ACTION_MAP


def _pick(obs: TrafficObservation, jt: str) -> int:
    em = [d for d, v in obs.emergency.items() if v]
    if em:
        for act, dirs in ACTION_MAP[jt].items():
            if em[0] in dirs:
                return act
    t = obs.traffic_counts
    return max(ACTION_MAP[jt],
               key=lambda a: sum(t[d] + 0.1 * t[d] ** 2
                                 for d in ACTION_MAP[jt][a]))


def run(steps: int = 15, seed: int = None) -> dict:
    env = TrafficEnvironment()
    obs = env.reset(seed=seed)
    total = 0.0

    for _ in range(steps):
        obs = env.step(TrafficAction(action_id=_pick(obs, env._junction)))
        total += obs.reward or 0.0

    return {
        "reward":    total,
        "em_total":  env._em_total,
        "em_served": env._em_served,
        "steps":     steps,
    }