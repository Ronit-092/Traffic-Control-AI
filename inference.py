"""
inference.py — LLM agent for Traffic-AI OpenEnv environment.

Follows the EXACT sample inference.py log format:
  [START] task=<task> env=<env> model=<model>
  [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
"""

import os
import json
import textwrap
from typing import List, Optional
import requests
from openai import OpenAI

# ── Environment variables (mandatory) ─────────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

API_KEY = HF_TOKEN or os.getenv("API_KEY", "dummy")

# ── Config ─────────────────────────────────────────────────────────────
ENV_URL    = "https://ronit-9-traffic-control-ai.hf.space"
TASK_NAME  = "traffic-signal-control"
BENCHMARK  = "traffic-ai"
MAX_STEPS  = 15
TEMPERATURE = 0.3
MAX_TOKENS  = 150
SUCCESS_SCORE_THRESHOLD = 0.1

# ── OpenAI client ──────────────────────────────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── Log helpers (exact format from sample) ─────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} "
          f"done={str(done).lower()} error={err}", flush=True)

def log_end(success: bool, steps: int,
            score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.3f} rewards={rewards_str}", flush=True)

# ── Action descriptions ────────────────────────────────────────────────
ACTION_DESCRIPTIONS = {
    "cross": {0:"NS green",1:"EW green",2:"North only",3:"East only"},
    "T":     {0:"NS green",1:"East green",2:"North only",3:"South only"},
    "Y":     {0:"North",   1:"East",     2:"West",      3:"North+East"},
}

SYSTEM_PROMPT = textwrap.dedent("""
    You are an intelligent traffic signal controller AI.
    At each step you receive vehicle queues and emergency status.
    
    PRIORITY: If an emergency vehicle is present, ALWAYS serve that lane first.
    Otherwise clear the most congested lanes.
    
    Respond ONLY with a JSON object:
    {"action_id": <0-3>, "reason": "<brief>"}
""").strip()

# ── Environment helpers ────────────────────────────────────────────────
def env_reset() -> dict:
    r = requests.post(f"{ENV_URL}/reset", json={},
                      headers={"Content-Type": "application/json"}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(action_id: int) -> dict:
    r = requests.post(f"{ENV_URL}/step",
                      json={"action_id": action_id},
                      headers={"Content-Type": "application/json"}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_state() -> dict:
    r = requests.get(f"{ENV_URL}/state", timeout=10)
    r.raise_for_status()
    return r.json()

# ── LLM decision ──────────────────────────────────────────────────────
def build_prompt(obs: dict) -> str:
    traffic   = obs.get("traffic_counts", {})
    emergency = obs.get("emergency", {})
    junction  = obs.get("junction_type", "cross")
    em_lanes  = [d for d, v in emergency.items() if v]
    em_str    = f"🚑 EMERGENCY at {em_lanes[0].upper()}!" if em_lanes else "None"
    actions   = ACTION_DESCRIPTIONS.get(junction, ACTION_DESCRIPTIONS["cross"])
    act_str   = "\n".join(f"  {k}: {v}" for k, v in actions.items())
    return (f"Junction: {junction}\n"
            f"Queues: N={traffic.get('north',0)} S={traffic.get('south',0)} "
            f"E={traffic.get('east',0)} W={traffic.get('west',0)}\n"
            f"Emergency: {em_str}\n"
            f"Actions:\n{act_str}\n"
            f"Choose action_id and give reason.")

def get_llm_action(obs: dict) -> tuple[int, str]:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(obs)},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        content = resp.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        parsed    = json.loads(content)
        action_id = max(0, min(3, int(parsed.get("action_id", 0))))
        reason    = parsed.get("reason", "")
        return action_id, reason
    except Exception as e:
        # Fallback to highest queue
        traffic = obs.get("traffic_counts", {})
        ns = traffic.get("north", 0) + traffic.get("south", 0)
        ew = traffic.get("east",  0) + traffic.get("west",  0)
        return (0 if ns >= ew else 1), f"fallback({e})"

# ── Main ───────────────────────────────────────────────────────────────
def main():
    rewards:    List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs   = env_reset()
        junction = obs.get("junction_type", "cross")

        for step in range(1, MAX_STEPS + 1):
            done = obs.get("done", False)
            if done:
                break

            action_id, reason = get_llm_action(obs)

            # Get action label for logging
            labels = ACTION_DESCRIPTIONS.get(junction, ACTION_DESCRIPTIONS["cross"])
            action_label = labels.get(action_id, str(action_id))

            result  = env_step(action_id)
            reward  = float(result.get("reward") or 0.0)
            done    = result.get("done", False)
            message = result.get("message", "")

            # Update junction in case it changed (it doesn't mid-episode but safe)
            junction = result.get("junction_type", junction)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_label,
                     reward=reward, done=done, error=None)

            obs = result

            if done:
                break

        # Score: normalise total reward to [0,1]
        total = sum(rewards)
        # rough max: 50 per step (emergency serve) × steps
        max_possible = MAX_STEPS * 55.0
        score   = min(max(total / max_possible, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        log_step(step=steps_taken + 1, action="error",
                 reward=0.0, done=True, error=str(e))
    finally:
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)


if __name__ == "__main__":
    main()