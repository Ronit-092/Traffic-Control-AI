"""
inference.py — LLM agent for Traffic-AI OpenEnv.

Required env vars:
    API_BASE_URL   — LLM API endpoint
    MODEL_NAME     — model identifier
    HF_TOKEN       — Hugging Face / API key
    ENV_URL        — Traffic-AI server URL (your HF Space)
"""

import os
import json
import requests
from typing import List, Optional
from openai import OpenAI

# ── Env vars ────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", "")
ENV_URL      = os.getenv("ENV_URL", "https://ronit-9-traffic-control-ai.hf.space")
TASK_NAME    = os.getenv("TRAFFIC_TASK", "medium")
BENCHMARK    = "traffic-ai"
MAX_STEPS    = 15

# Score normalisation: medium task, per-step reward max ~10
MAX_TOTAL_REWARD = MAX_STEPS * 10.0
SUCCESS_SCORE_THRESHOLD = 0.3

# ── Logging (exact format required by OpenEnv) ──────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── OpenAI client ───────────────────────────────────────────────────────
_client = None
def get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN if HF_TOKEN else "dummy",
        )
    return _client

# ── Env HTTP calls ──────────────────────────────────────────────────────
def env_reset():
    resp = requests.post(f"{ENV_URL}/reset", json={}, timeout=30)
    resp.raise_for_status()
    return resp.json()

def env_step(action_id: int):
    resp = requests.post(f"{ENV_URL}/step", json={"action_id": action_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()

# ── Action descriptions ─────────────────────────────────────────────────
ACTION_DESCRIPTIONS = {
    "cross": {0:"NS green",1:"EW green",2:"North only",3:"East only"},
    "T":     {0:"NS green",1:"East green",2:"North only",3:"South only"},
    "Y":     {0:"North",1:"East",2:"West",3:"North+East"},
}

SYSTEM_PROMPT = """You are an intelligent traffic signal controller AI.
At each step you receive vehicle queues and emergency status.
ALWAYS prioritise emergency vehicles first.
Respond ONLY with a JSON object: {"action_id": <0-3>, "reason": "<brief>"}"""

def format_obs(obs: dict) -> str:
    tc  = obs.get("traffic_counts", {})
    em  = obs.get("emergency", {})
    jt  = obs.get("junction_type", "cross")
    em_lanes = [d for d, v in em.items() if v]
    em_str   = f"EMERGENCY at {em_lanes[0].upper()}!" if em_lanes else "None"
    acts = ACTION_DESCRIPTIONS.get(jt, ACTION_DESCRIPTIONS["cross"])
    act_str = "\n".join(f"  {k}: {v}" for k, v in acts.items())
    return (f"Junction: {jt}\n"
            f"Queues — N:{tc.get('north',0)} S:{tc.get('south',0)} "
            f"E:{tc.get('east',0)} W:{tc.get('west',0)}\n"
            f"Emergency: {em_str}\nActions:\n{act_str}\nChoose action_id 0-3.")

def heuristic_action(obs: dict) -> tuple:
    tc = obs.get("traffic_counts", {})
    em = obs.get("emergency", {})
    em_lanes = [d for d, v in em.items() if v]
    if em_lanes:
        d = em_lanes[0]
        return (0, f"emergency-{d}") if d in ("north","south") else (1, f"emergency-{d}")
    ns = tc.get("north",0) + tc.get("south",0)
    ew = tc.get("east",0)  + tc.get("west",0)
    return (0,"busy-NS") if ns >= ew else (1,"busy-EW")

def get_llm_action(obs: dict, step: int) -> tuple:
    raw = ""
    try:
        client   = get_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": format_obs(obs)},
            ],
            max_tokens=100,
            temperature=0.1,
        )
        raw     = response.choices[0].message.content.strip()
        content = raw
        if "```" in content:
            parts   = content.split("```")
            content = parts[1] if len(parts) > 1 else parts[0]
            if content.lower().startswith("json"):
                content = content[4:]
        parsed    = json.loads(content.strip())
        action_id = max(0, min(3, int(parsed.get("action_id", 0))))
        reason    = parsed.get("reason", "")
        return action_id, reason
    except json.JSONDecodeError:
        for c in raw:
            if c in "0123":
                return int(c), "json-parse-fallback"
        return heuristic_action(obs)
    except Exception as e:
        print(f"[DEBUG] LLM error step {step}: {e}", flush=True)
        return heuristic_action(obs)

# ── Main ────────────────────────────────────────────────────────────────
def main():
    rewards: List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env_reset()

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break

            action_id, reason = get_llm_action(obs, step)
            error_msg = None

            try:
                result = env_step(action_id)
            except Exception as e:
                error_msg = str(e)
                result = {"reward": 0.0, "done": False, **obs}

            reward = float(result.get("reward") or 0.0)
            done   = result.get("done", False)

            rewards.append(reward)
            steps_taken = step

            jt         = obs.get("junction_type", "cross")
            action_str = ACTION_DESCRIPTIONS.get(jt, {}).get(action_id, str(action_id))
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            obs = result
            if done:
                break

    finally:
        score   = min(max(sum(rewards) / MAX_TOTAL_REWARD, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()