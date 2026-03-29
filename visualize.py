"""visualize.py — Live ASCII visualiser. Run: python visualize.py"""

import os, time, warnings
warnings.filterwarnings("ignore")

from stable_baselines3 import PPO
from gymnasium_adapter import TrafficGymnasiumEnv

STEPS  = 20
LABELS = {
    "cross": {0:"North-South",1:"East-West", 2:"North only",3:"East only"},
    "T":     {0:"North-South",1:"East",       2:"North only",3:"South only"},
    "Y":     {0:"North",      1:"East",        2:"West",      3:"North+East"},
}


def bar(n, mx=30, w=15):
    f = int(n/mx*w)
    return "█"*f + "░"*(w-f)


def load_model():
    for p in ["models/best/best_model", "models/traffic_model"]:
        if os.path.exists(p + ".zip"):
            return PPO.load(p)
    raise FileNotFoundError("Run train.py first.")


def render(env, step, action, reward):
    t   = env._core._traffic
    em  = env._core._emergency
    jt  = env._core._junction
    lbl = LABELS[jt][action]
    tag = lambda d: "🚑" if em.get(d) else "  "

    print(f"""
╔══════════════════════════════════════════════╗
║  Traffic-AI   Step {step:>2}/{STEPS}   [{jt}]
╠══════════════════════════════════════════════╣
║  NORTH {t['north']:>3}  {bar(t['north']):<15} {tag('north')}
║  SOUTH {t['south']:>3}  {bar(t['south']):<15} {tag('south')}
║  EAST  {t['east']:>3}  {bar(t['east']):<15} {tag('east')}
║  WEST  {t['west']:>3}  {bar(t['west']):<15} {tag('west')}
╠══════════════════════════════════════════════╣
║  🚦 {lbl:<41}
║  💰 Reward: {reward:>+8.1f}
╚══════════════════════════════════════════════╝""")


def main():
    model = load_model()
    env   = TrafficGymnasiumEnv()
    obs, _ = env.reset()
    total = 0.0

    for step in range(1, STEPS+1):
        os.system("cls" if os.name=="nt" else "clear")
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, _, _, _ = env.step(action)
        total += reward
        render(env, step, action, reward)
        time.sleep(1.0)

    os.system("cls" if os.name=="nt" else "clear")
    print(f"\n{'='*48}")
    print(f"  Done — {STEPS} steps   Total reward: {total:+.1f}")
    print(f"{'='*48}\n")
    env.close()


if __name__ == "__main__":
    main()
