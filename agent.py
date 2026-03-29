"""agent.py — Evaluate trained PPO agent. Run: python agent.py"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
from stable_baselines3 import PPO
from gymnasium_adapter import TrafficGymnasiumEnv

STEPS    = 20
EPISODES = 15
LABELS   = {
    "cross": {0:"NS",1:"EW", 2:"N",3:"E"},
    "T":     {0:"NS",1:"E",  2:"N",3:"S"},
    "Y":     {0:"N", 1:"E",  2:"W",3:"NE"},
}


def load_model():
    for p in ["models/best/best_model", "models/traffic_model"]:
        if os.path.exists(p + ".zip"):
            return PPO.load(p)
    raise FileNotFoundError("Run train.py first.")


def main():
    print("\n" + "="*54)
    print("  Traffic-AI — Agent Evaluation")
    print("="*54)

    model = load_model()
    env   = TrafficGymnasiumEnv()
    rewards, total_em, total_sv = [], 0, 0

    for ep in range(1, EPISODES+1):
        obs, _ = env.reset()
        ep_r, em_t, em_s = 0.0, 0, 0
        verbose = ep <= 2

        if verbose:
            jt = env._core._junction
            print(f"\n{'='*54}\n  Episode {ep}  |  Junction: {jt}\n{'='*54}")

        for step in range(STEPS):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            before = {d:v for d,v in env._core._traffic.items() if v > 0}
            obs, reward, _, _, info = env.step(action)
            ep_r += reward
            if info.get("emergency"):
                em_t += 1

            if verbose:
                jt  = env._core._junction
                lbl = LABELS[jt][action]
                print(f"  Step {step+1:>2} | {str(before):<28} → {lbl:<6} | {reward:+.1f}")

        rewards.append(ep_r)
        total_em += em_t
        total_sv += em_s
        if not verbose:
            print(f"  Episode {ep:>2} | Reward: {ep_r:+7.1f}  Em: {em_s}/{em_t}")

    arr = np.array(rewards)
    print(f"\n{'='*54}")
    print(f"  SUMMARY  ({EPISODES} eps × {STEPS} steps)")
    print(f"{'='*54}")
    print(f"  Mean   : {arr.mean():+.2f}  Std: {arr.std():.2f}")
    print(f"  Best   : {arr.max():+.2f}  Worst: {arr.min():+.2f}")
    env.close()


if __name__ == "__main__":
    main()
