"""
inference.py — LLM agent running ALL 3 tasks on Traffic-AI OpenEnv.
Each task gets its own [START]...[STEP]...[END] block.
Difficulty level printed in logs as required by checker.
"""

import os, json, textwrap
from typing import List, Optional
import requests
from openai import OpenAI

API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

API_KEY   = HF_TOKEN or os.getenv("API_KEY", "dummy")
ENV_URL   = "https://ronit-9-traffic-control-ai.hf.space"
BENCHMARK = "traffic-ai"
SUCCESS_SCORE_THRESHOLD = 0.1

TASKS = [
    {"id": "easy",   "difficulty": "easy",   "max_steps": 10},
    {"id": "medium", "difficulty": "medium", "max_steps": 15},
    {"id": "hard",   "difficulty": "hard",   "max_steps": 20},
]

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

ACTION_LABELS = {
    "cross": {0:"NS-green",1:"EW-green",2:"North-only",3:"East-only"},
    "T":     {0:"NS-green",1:"East-green",2:"North-only",3:"South-only"},
    "Y":     {0:"North",1:"East",2:"West",3:"North+East"},
}

SYSTEM_PROMPT = "You are a traffic signal controller. Prioritise emergency vehicles. Otherwise clear most congested lanes. Reply ONLY with JSON: {\"action_id\": <0-3>, \"reason\": \"<brief>\"}"

def env_reset(seed=None):
    body = {"seed": seed} if seed is not None else {}
    r = requests.post(f"{ENV_URL}/reset", json=body, headers={"Content-Type":"application/json"}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(action_id):
    r = requests.post(f"{ENV_URL}/step", json={"action_id": action_id}, headers={"Content-Type":"application/json"}, timeout=30)
    r.raise_for_status()
    return r.json()

def get_llm_action(obs, junction):
    tc = obs.get("traffic_counts", {})
    em = obs.get("emergency", {})
    em_lanes = [d for d,v in em.items() if v]
    em_str = f"EMERGENCY:{em_lanes[0].upper()}" if em_lanes else "no-emergency"
    labels = ACTION_LABELS.get(junction, ACTION_LABELS["cross"])
    prompt = f"junction={junction} N={tc.get('north',0)} S={tc.get('south',0)} E={tc.get('east',0)} W={tc.get('west',0)} {em_str} actions={labels}"
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}],
            max_tokens=80, temperature=0.2,
        )
        content = resp.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        parsed = json.loads(content)
        return max(0, min(3, int(parsed.get("action_id", 0)))), parsed.get("reason","")
    except Exception as e:
        tc2 = obs.get("traffic_counts", {})
        ns = tc2.get("north",0) + tc2.get("south",0)
        ew = tc2.get("east",0)  + tc2.get("west",0)
        return (0 if ns >= ew else 1), f"fallback"

def run_task(task):
    task_id    = task["id"]
    difficulty = task["difficulty"]
    max_steps  = task["max_steps"]

    # Print difficulty in logs — required by checker
    print(f"# difficulty={difficulty} task={task_id}", flush=True)
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        obs = env_reset(seed=42)
        junction = obs.get("junction_type", "cross")

        for step in range(1, max_steps + 1):
            if obs.get("done", False):
                break
            action_id, _ = get_llm_action(obs, junction)
            labels = ACTION_LABELS.get(junction, ACTION_LABELS["cross"])
            action_label = labels.get(action_id, str(action_id))
            result   = env_step(action_id)
            reward   = float(result.get("reward") or 0.0)
            done     = result.get("done", False)
            junction = result.get("junction_type", junction)
            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_label, reward=reward, done=done, error=None)
            obs = result
            if done:
                break

        total = sum(rewards)
        score = round(min(max(total / (max_steps * 55.0), 0.0), 1.0), 3)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        log_step(step=steps_taken+1, action="error", reward=0.0, done=True, error=str(e))
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

def main():
    print(f"# Traffic-AI — running {len(TASKS)} tasks: easy, medium, hard", flush=True)
    all_scores = []
    for task in TASKS:
        score = run_task(task)
        all_scores.append(score)
        print(f"# DONE task={task['id']} difficulty={task['difficulty']} score={score:.3f}", flush=True)
    print(f"# ALL_DONE avg={sum(all_scores)/len(all_scores):.3f}", flush=True)

if __name__ == "__main__":
    main()