"""tasks/task_hard.py — 1-step lookahead greedy planner."""
import random
from models import TrafficAction, TrafficObservation
from server.environment import TrafficEnvironment, ACTION_MAP, ACTIVE_LANES, MAX_QUEUE, MAX_ARRIVAL


def _simulate(obs: TrafficObservation, jt: str, act: int) -> float:
    t  = dict(obs.traffic_counts)
    em = obs.emergency
    dirs = ACTION_MAP[jt].get(act, [])
    cleared = sum(t[d] for d in dirs)
    for d in dirs: t[d] = 0
    reward = float(cleared)
    em_lanes = [d for d, v in em.items() if v]
    if em_lanes:
        reward += 50.0 if em_lanes[0] in dirs else -20.0
    for lane in ACTIVE_LANES[jt]:
        t[lane] = min(t[lane] + (MAX_ARRIVAL/2 if t[lane] > 0 else 0.8), MAX_QUEUE)
    reward -= 0.5 * sum(t.values())
    return reward


def run(steps: int = 20, seed: int = None) -> dict:
    env = TrafficEnvironment()
    obs = env.reset(seed=seed)
    for lane in ACTIVE_LANES[env._junction]:
        env._traffic[lane] = min(env._traffic[lane] + 10, MAX_QUEUE)
    obs = env._make_obs(reward=None, done=False, msg="surge")
    total = 0.0

    for _ in range(steps):
        if random.random() < 0.3:
            spike = random.choice(ACTIVE_LANES[env._junction])
            env._traffic[spike] = min(env._traffic[spike] + random.randint(5,15), MAX_QUEUE)
            obs = env._make_obs(reward=None, done=False, msg="spike")
        act = max(range(4), key=lambda a: _simulate(obs, env._junction, a))
        obs = env.step(TrafficAction(action_id=act))
        total += obs.reward or 0.0

    return {"reward": total, "em_total": env._em_total,
            "em_served": env._em_served, "steps": steps}
