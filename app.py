"""app.py — Rule-based vs PPO comparison. Run: python app.py"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
from stable_baselines3 import PPO
from gymnasium_adapter import TrafficGymnasiumEnv
from graders.grader import grade
from tasks.task_easy   import run as easy_run
from tasks.task_medium import run as medium_run
from tasks.task_hard   import run as hard_run

TASKS = {
    "easy":   {"steps": 10, "fn": easy_run},
    "medium": {"steps": 15, "fn": medium_run},
    "hard":   {"steps": 20, "fn": hard_run},
}


def load_model():
    for p in ["models/best/best_model", "models/traffic_model"]:
        if os.path.exists(p + ".zip"):
            return PPO.load(p)
    raise FileNotFoundError("Run train.py first.")


def run_ai(model, steps, seed=None):
    env = TrafficGymnasiumEnv()
    obs, _ = env.reset(seed=seed)
    total, em_t, em_s = 0.0, 0, 0
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, _, _, info = env.step(int(action))
        total += reward
        if info.get("emergency"):
            em_t += 1
    env.close()
    return {"reward": total, "em_total": em_t, "em_served": em_s, "steps": steps}


def avg(fn, n=10, **kw):
    rs = [fn(seed=i, **kw) for i in range(n)]
    return {
        "reward":    float(np.mean([r["reward"]    for r in rs])),
        "em_total":  sum(r["em_total"]  for r in rs),
        "em_served": sum(r["em_served"] for r in rs),
        "steps":     kw.get("steps", 10),
    }


def main():
    print("\n" + "="*60)
    print("  Traffic-AI — Rule-Based vs PPO")
    print("="*60)

    try:
        model = load_model()
        print("  ✅ PPO model loaded\n")
    except FileNotFoundError as e:
        print(f"  ⚠️  {e}\n")
        model = None

    totals = {}
    for diff, cfg in TASKS.items():
        steps = cfg["steps"]
        print("─"*60)
        print(f"  {diff.upper()}  ({steps} steps, avg 10 runs)")
        print("─"*60)

        rb = avg(cfg["fn"], n=10, steps=steps)
        rb_score = grade(rb)
        ep = rb["em_served"]/rb["em_total"]*100 if rb["em_total"] else 0
        print(f"  Rule-based → reward: {rb['reward']:>8.1f}  "
              f"score: {rb_score}/10  em: {rb['em_served']}/{rb['em_total']} ({ep:.0f}%)")

        if model:
            ai = avg(lambda seed, steps: run_ai(model, steps, seed=seed), n=10, steps=steps)
            ai_score = grade(ai)
            ep_ai = ai["em_served"]/ai["em_total"]*100 if ai["em_total"] else 0
            print(f"  PPO Agent  → reward: {ai['reward']:>8.1f}  "
                  f"score: {ai_score}/10  em: {ai['em_served']}/{ai['em_total']} ({ep_ai:.0f}%)")
            delta = ai["reward"] - rb["reward"]
            print(f"  Winner     → {'PPO 🤖' if delta>0 else 'Rule-based 📋'}  (Δ {delta:+.1f})")
            totals[diff] = (rb_score, ai_score)
        print()

    if totals and model:
        print("="*60)
        rb_t = sum(v[0] for v in totals.values())
        ai_t = sum(v[1] for v in totals.values())
        print(f"  Rule-based: {rb_t}/30  |  PPO: {ai_t}/30")
        print(f"  🏆  {'PPO Agent 🤖' if ai_t >= rb_t else 'Rule-based 📋'}")
        print("="*60)


if __name__ == "__main__":
    main()
