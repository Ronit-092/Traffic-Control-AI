"""
inference.py — LLM agent playing the Traffic-AI OpenEnv environment.

Required environment variables:
    API_BASE_URL   — LLM API endpoint (e.g. https://api.openai.com/v1)
    MODEL_NAME     — model identifier  (e.g. gpt-4o-mini)
    HF_TOKEN       — Hugging Face / API key

Optional:
    LOCAL_IMAGE_NAME — if using from_docker_image()

Logs follow the required START / STEP / END structured format exactly.
"""

import os
import json
import requests

# ── Environment variables ───────────────────────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN", "")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ── OpenAI client — created lazily so bad env vars don't crash on import ─
_client = None

def get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN if HF_TOKEN else "dummy",
        )
    return _client

# ── Environment endpoint ────────────────────────────────────────────────
ENV_URL = os.getenv("ENV_URL", "https://ronit-9-traffic-control-ai.hf.space")

MAX_STEPS = 10

# ── Action map for the LLM to understand ───────────────────────────────
ACTION_DESCRIPTIONS = {
    "cross": {
        0: "Give green light to North-South traffic",
        1: "Give green light to East-West traffic",
        2: "Give green light to North only",
        3: "Give green light to East only",
    },
    "T": {
        0: "Give green light to North-South traffic",
        1: "Give green light to East traffic",
        2: "Give green light to North only",
        3: "Give green light to South only",
    },
    "Y": {
        0: "Give green light to North traffic",
        1: "Give green light to East traffic",
        2: "Give green light to West traffic",
        3: "Give green light to North and East traffic",
    },
}

SYSTEM_PROMPT = """You are an intelligent traffic signal controller AI.

You control traffic signals at road junctions. At each step you receive:
- The vehicle queue at each direction (North, South, East, West)
- Whether there is an emergency vehicle (ambulance) waiting
- The junction type (cross, T, or Y)

Your goal is to:
1. ALWAYS prioritise emergency vehicles — if an ambulance is waiting, serve that direction immediately
2. Clear the most congested lanes to maximise vehicle throughput
3. Avoid letting any lane get severely backed up

You must respond with ONLY a JSON object in this exact format:
{"action_id": <number>, "reason": "<brief explanation>"}

action_id must be 0, 1, 2, or 3."""


def env_reset():
    try:
        resp = requests.post(
            f"{ENV_URL}/reset",
            json={},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [WARN] env_reset failed: {e}")
        return {"observation": {"traffic_counts": {"north": 0, "south": 0, "east": 0, "west": 0},
                                "emergency": {}, "junction_type": "cross"}}


def env_step(action_id: int):
    try:
        resp = requests.post(
            f"{ENV_URL}/step",
            json={"action": {"action_id": action_id}},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [WARN] env_step failed: {e}")
        return {"observation": {"traffic_counts": {"north": 0, "south": 0, "east": 0, "west": 0},
                                "emergency": {}, "junction_type": "cross"},
                "reward": 0.0, "done": False}


def env_state():
    try:
        resp = requests.get(f"{ENV_URL}/state", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [WARN] env_state failed: {e}")
        return {}


def format_observation(obs: dict) -> str:
    """Format observation into a natural language prompt for the LLM."""
    observation = obs.get("observation", obs)
    traffic     = observation.get("traffic_counts", {})
    emergency   = observation.get("emergency", {})
    junction    = observation.get("junction_type", "cross")

    em_lanes = [d for d, v in emergency.items() if v]
    em_str   = f"EMERGENCY VEHICLE at: {em_lanes[0].upper()}!" if em_lanes else "No emergency vehicles"

    actions    = ACTION_DESCRIPTIONS.get(junction, ACTION_DESCRIPTIONS["cross"])
    action_str = "\n".join(f"  {k}: {v}" for k, v in actions.items())

    return f"""Junction type: {junction}

Current vehicle queues:
  North: {traffic.get('north', 0)} vehicles
  South: {traffic.get('south', 0)} vehicles
  East:  {traffic.get('east',  0)} vehicles
  West:  {traffic.get('west',  0)} vehicles

Emergency status: {em_str}

Available actions:
{action_str}

Choose the best action_id (0-3) and explain your reasoning."""


def heuristic_action(obs: dict) -> tuple:
    """Fallback rule-based action when LLM is unavailable."""
    observation = obs.get("observation", obs)
    traffic     = observation.get("traffic_counts", {})
    emergency   = observation.get("emergency", {})

    # Emergency priority
    em_lanes = [d for d, v in emergency.items() if v]
    if em_lanes:
        direction = em_lanes[0]
        if direction in ("north", "south"):
            return 0, f"Emergency vehicle at {direction} — serving NS"
        else:
            return 1, f"Emergency vehicle at {direction} — serving EW"

    # Serve busiest pair
    ns = traffic.get("north", 0) + traffic.get("south", 0)
    ew = traffic.get("east",  0) + traffic.get("west",  0)
    if ns >= ew:
        return 0, "Higher NS queue — serving North-South"
    return 1, "Higher EW queue — serving East-West"


def get_llm_action(obs: dict, step: int) -> tuple:
    """Ask the LLM to choose an action. Falls back to heuristic on any error."""
    user_message = format_observation(obs)
    raw_content  = ""

    try:
        client   = get_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=150,
            temperature=0.1,
        )

        raw_content = response.choices[0].message.content.strip()
        content     = raw_content

        # Strip markdown code fences if present
        if "```" in content:
            parts   = content.split("```")
            content = parts[1] if len(parts) > 1 else parts[0]
            if content.lower().startswith("json"):
                content = content[4:]
        content = content.strip()

        parsed    = json.loads(content)
        action_id = int(parsed.get("action_id", 0))
        reason    = parsed.get("reason", "No reason given")
        action_id = max(0, min(3, action_id))
        return action_id, reason

    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON parse error at step {step}: {e}")
        # Try to salvage a digit from raw response
        action_id = 0
        for char in raw_content:
            if char in "0123":
                action_id = int(char)
                break
        return action_id, f"JSON parse fallback (raw: {raw_content[:80]})"

    except Exception as e:
        print(f"  [WARN] LLM call failed at step {step}: {e} — using heuristic")
        return heuristic_action(obs)


def main():
    total_reward = 0.0
    em_total     = 0
    em_served    = 0

    # ── [START] log ─────────────────────────────────────────────────────
    print("[START]")
    print(f"  env_url:    {ENV_URL}")
    print(f"  model:      {MODEL_NAME}")
    print(f"  max_steps:  {MAX_STEPS}")

    # Reset environment
    obs      = env_reset()
    state    = env_state()
    obs_data = obs.get("observation", obs)
    junction = obs_data.get("junction_type", "unknown")

    print(f"  junction:   {junction}")
    print(f"  episode_id: {state.get('episode_id', 'unknown')}")

    step = 0  # ensure defined even if loop body never executes

    # ── Run episode ──────────────────────────────────────────────────────
    for step in range(1, MAX_STEPS + 1):

        # Get LLM decision (with heuristic fallback)
        action_id, reason = get_llm_action(obs, step)

        # Execute action
        result  = env_step(action_id)
        res_obs = result.get("observation", result)
        reward  = result.get("reward", 0.0) or 0.0
        done    = result.get("done", False)

        total_reward += reward

        # Track emergency stats
        emergency = res_obs.get("emergency", {})
        em_active = any(emergency.values())
        if em_active:
            em_total += 1
            served_dirs = {
                "cross": {0: ["north", "south"], 1: ["east", "west"], 2: ["north"], 3: ["east"]},
                "T":     {0: ["north", "south"], 1: ["east"],         2: ["north"], 3: ["south"]},
                "Y":     {0: ["north"],           1: ["east"],         2: ["west"],  3: ["north", "east"]},
            }.get(junction, {}).get(action_id, [])
            em_dir = [d for d, v in emergency.items() if v]
            if em_dir and em_dir[0] in served_dirs:
                em_served += 1

        traffic = res_obs.get("traffic_counts", {})
        message = res_obs.get("message", "")

        # ── [STEP] log ───────────────────────────────────────────────────
        print(f"[STEP]")
        print(f"  step:         {step}")
        print(f"  action_id:    {action_id}")
        print(f"  reason:       {reason}")
        print(f"  reward:       {reward}")
        print(f"  total_reward: {total_reward}")
        print(f"  traffic:      N={traffic.get('north',0)} S={traffic.get('south',0)} E={traffic.get('east',0)} W={traffic.get('west',0)}")
        print(f"  emergency:    {em_active}")
        print(f"  message:      {message}")
        print(f"  done:         {done}")

        obs = result

        if done:
            break

    # Final state
    final_state = env_state()

    # ── [END] log ────────────────────────────────────────────────────────
    print("[END]")
    print(f"  total_reward:    {total_reward}")
    print(f"  steps_taken:     {step}")
    print(f"  em_total:        {em_total}")
    print(f"  em_served:       {em_served}")
    print(f"  em_response_pct: {round(em_served / em_total * 100, 1) if em_total > 0 else 0.0}")
    print(f"  episode_id:      {final_state.get('episode_id', 'unknown')}")
    print(f"  final_step_count:{final_state.get('step_count', step)}")


if __name__ == "__main__":
    main()
