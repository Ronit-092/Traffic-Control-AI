"""tasks/task_easy.py — Simple NS vs EW heuristic (10 steps)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import TrafficAction
from server.environment import TrafficEnvironment


def run(steps: int = 10, seed: int = None) -> dict:
    env = TrafficEnvironment()
    obs = env.reset(seed=seed)
    total = 0.0

    for _ in range(steps):
        ns = obs.obs_normalized[0] + obs.obs_normalized[1]
        ew = obs.obs_normalized[2] + obs.obs_normalized[3]
        obs = env.step(TrafficAction(action_id=0 if ns >= ew else 1))
        total += obs.reward or 0.0

    return {
        "reward":    total,
        "em_total":  env._em_total,
        "em_served": env._em_served,
        "steps":     steps,
    }