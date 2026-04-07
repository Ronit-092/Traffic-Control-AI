---
title: Traffic AI Signal Control
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
app_file: server/app.py
pinned: false
---

# 🚦 Traffic-AI — OpenEnv Traffic Signal Control

> **ScalarX × Meta Hackathon 2026** — Round 1 Submission  
> Built by **Team: 3 Bit Coders** on the OpenEnv standard (Meta PyTorch × Hugging Face)

A Reinforcement Learning environment where a **PPO agent** learns to control traffic signals at dynamic road junctions — maximising vehicle throughput and prioritising emergency vehicles over normal traffic.

Live demo: **[huggingface.co/spaces/Ronit-9/Traffic-Control-AI](https://huggingface.co/spaces/Ronit-9/Traffic-Control-AI)**

---

## What this project does

Real traffic lights use fixed timers — they switch every 60 seconds regardless of actual traffic. This project replaces that with an AI agent that:

- **Watches** vehicle queues at all 4 directions in real time
- **Decides** which direction gets a green light each step
- **Prioritises** emergency vehicles (ambulances) immediately when they appear
- **Learns** through 50,000 training steps using Proximal Policy Optimization (PPO)

The environment simulates 3 real-world junction types (Cross, T, Y) and the agent must generalise across all of them.

---

## Live Dashboard

Visit the HF Space App tab to interact with the environment directly in your browser:

- **Junction display** — real-time vehicle counts at each direction, red highlight for emergencies
- **Signal controls** — click any action button to control the traffic lights yourself
- **Step log** — every action, reward, and emergency event logged in real time
- **Reward history** — bar chart showing performance over the episode

---

## OpenEnv Compliance

| Criterion | Detail | Status |
|---|---|---|
| Runtime correctness | All endpoints return 200 OK, no crashes | ✅ |
| Interface compliance | `create_fastapi_app()`, typed models, `openenv validate` all YES | ✅ |
| Task design | 3 tasks with genuinely different algorithms | ✅ |
| Grading logic | Per-step normalised scoring + emergency response bonus | ✅ |
| Docker packaging | Builds from `python:3.11-slim`, `openenv.yaml` manifest present | ✅ |
| `inference.py` | LLM agent with `[START]`/`[STEP]`/`[END]` log format | ✅ |

---

## Project Structure

```
traffic-ai/
│
├── inference.py              ← LLM agent (START/STEP/END logs, OpenAI client)
├── models.py                 ← Pydantic Action / Observation / State (OpenEnv types)
├── client.py                 ← EnvClient WebSocket client
├── openenv.yaml              ← Environment manifest
├── pyproject.toml            ← uv project config (enables: uv run server)
│
├── server/
│   ├── environment.py        ← TrafficEnvironment core logic
│   ├── app.py                ← FastAPI server + live dashboard UI
│   ├── requirements.txt      ← Python dependencies
│   └── Dockerfile            ← Container definition
│
├── tasks/
│   ├── task_easy.py          ← Simple NS vs EW heuristic (10 steps)
│   ├── task_medium.py        ← Emergency-aware quadratic scorer (15 steps)
│   └── task_hard.py          ← 1-step lookahead greedy planner (20 steps)
│
├── graders/
│   └── grader.py             ← Per-step normalised scoring (score 2–10)
│
├── gymnasium_adapter.py      ← Thin Gymnasium wrapper for PPO/SB3 training
├── train.py                  ← PPO training (50k steps, no callbacks)
├── agent.py                  ← Multi-episode evaluation with stats
├── app.py                    ← Rule-based vs PPO head-to-head comparison
└── visualize.py              ← Live ASCII junction visualiser (terminal)
```

---

## Junction Types

| Type | Active Lanes | Available Actions |
|---|---|---|
| **Cross** | N, S, E, W | 0=NS · 1=EW · 2=North only · 3=East only |
| **T-junction** | N, S, E | 0=NS · 1=East · 2=North only · 3=South only |
| **Y-junction** | N, E, W | 0=North · 1=East · 2=West · 3=North+East |

The junction type is randomised on every `reset()`. The observation is always a 4-vector `[north, south, east, west]` normalised 0–1 — inactive lanes for a given junction type are always 0.

---

## Reward Function

| Event | Reward | Reason |
|---|---|---|
| Vehicle cleared | +1 per vehicle | Encourages throughput |
| Emergency vehicle served | +50 | Strong signal to always prioritise ambulances |
| Emergency vehicle ignored | −10 | Penalty for missing one |
| Vehicle still waiting | −0.1 per vehicle | Teaches urgency, prevents ignoring backed-up lanes |

**Emergency timing:** The emergency is spawned at the **end** of each step and shown in the response — so the player/agent always sees the emergency **before** choosing their next action. This makes the `+50` bonus fully achievable if you pick the right direction.

---

## Tasks (increasing difficulty)

### Easy — `tasks/task_easy.py`
Simple heuristic: compare NS load vs EW load, open whichever is heavier. No emergency awareness. Runs 10 steps.

```python
action = 0 if (north + south) >= (east + west) else 1
```

### Medium — `tasks/task_medium.py`
Emergency-aware weighted scorer. If an emergency is active, override and serve that direction. Otherwise, pick the action with the highest quadratic-weighted score across all 4 actions. Runs 15 steps.

```python
if emergency_active:
    serve_emergency_direction()
else:
    pick_action_with_max_quadratic_score()
```

### Hard — `tasks/task_hard.py`
1-step lookahead greedy planner. Simulates all 4 possible actions and picks the one with the best expected reward (clearance + emergency + residual queue estimate). Also injects random traffic surges. Runs 20 steps.

```python
best_action = max(range(4), key=lambda a: simulate_reward(a))
```

---

## Grading Logic

```python
per_step = total_reward / steps

if   per_step >= 6.0:  score = 10   # excellent
elif per_step >= 3.5:  score = 8    # good
elif per_step >= 1.5:  score = 6    # acceptable
elif per_step >= 0.0:  score = 4    # poor
else:                  score = 2    # net negative

# Emergency response bonus
if em_total > 0 and (em_served / em_total) >= 0.80:
    score = min(score + 1, 10)
```

Scoring is **per-step normalised** so easy (10 steps), medium (15), and hard (20) produce comparable scores.

---

## API Endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `GET` | `/health` | — | Liveness check |
| `POST` | `/reset` | `{}` or `{"seed": 42}` | Start new episode |
| `POST` | `/step` | `{"action_id": 0}` | Execute one action |
| `GET` | `/state` | — | Episode metadata |
| `GET` | `/docs` | — | Swagger UI |
| `GET` | `/` | — | Live dashboard |

### Example usage

```bash
# Health check
curl https://ronit-9-traffic-control-ai.hf.space/health

# Start episode
curl -X POST https://ronit-9-traffic-control-ai.hf.space/reset \
  -H "Content-Type: application/json" -d '{}'

# Take action (0=NS, 1=EW, 2=N-only, 3=E-only for cross junction)
curl -X POST https://ronit-9-traffic-control-ai.hf.space/step \
  -H "Content-Type: application/json" -d '{"action_id": 0}'

# Episode state
curl https://ronit-9-traffic-control-ai.hf.space/state
```

### Python client

```python
from client import TrafficEnv
from models import TrafficAction

with TrafficEnv(base_url="https://ronit-9-traffic-control-ai.hf.space") as env:
    result = env.reset()
    print(result.observation.junction_type)   # "cross" | "T" | "Y"
    print(result.observation.traffic_counts)  # {"north": 5, "south": 3, ...}

    result = env.step(TrafficAction(action_id=0))
    print(result.reward)   # e.g. +16.5
    print(result.done)     # False
```

---

## PPO Training

```bash
# Install dependencies
pip install -r server/requirements.txt

# Train the PPO agent (~3-5 minutes)
python train.py

# Compare rule-based vs PPO across all 3 difficulties
python app.py

# Detailed evaluation (15 episodes with stats)
python agent.py

# Live ASCII visualiser
python visualize.py
```

### Hyperparameters

| Parameter | Value | Reason |
|---|---|---|
| Algorithm | PPO (stable-baselines3) | Standard RL, stable on discrete action spaces |
| Total timesteps | 50,000 | ~3-5 min on CPU, good convergence |
| `n_steps` | 2048 | Long rollouts capture multi-step queue dynamics |
| `batch_size` | 64 | Stable updates on single env |
| `learning_rate` | 2.5e-4 | Conservative, avoids overshooting |
| `clip_range` | 0.15 | Tighter clipping for stable policy updates |
| `ent_coef` | 0.01 | Encourages exploration of all 4 actions |
| `net_arch` | [128, 128] | Deeper MLP for richer feature extraction |

---

## inference.py — LLM Agent

The `inference.py` file implements an LLM-powered agent that plays the environment using the OpenAI client. It follows the required `[START]`/`[STEP]`/`[END]` log format exactly.

### Required environment variables

```bash
export API_BASE_URL="https://api.openai.com/v1"   # LLM API endpoint
export MODEL_NAME="gpt-4o-mini"                    # Model identifier
export HF_TOKEN="hf_..."                           # Hugging Face / API key
```

### Running the LLM agent

```bash
python inference.py
```

### Output format

```
[START]
  env_url:    https://ronit-9-traffic-control-ai.hf.space
  model:      gpt-4o-mini
  max_steps:  10
  junction:   cross
  episode_id: abc123

[STEP]
  step:         1
  action_id:    0
  reason:       North+South has 8+5=13 vehicles vs East+West 3+2=5. Serving NS.
  reward:       +13.0
  total_reward: 13.0
  traffic:      N=0 S=0 E=3 W=2
  emergency:    False
  message:      Step 1: NS green — cleared 13
  done:         False

[END]
  total_reward:    87.5
  steps_taken:     10
  em_total:        3
  em_served:       3
  em_response_pct: 100.0
  episode_id:      abc123
  final_step_count:10
```

---

## Docker

```bash
# Build locally
docker build -t traffic-ai -f server/Dockerfile .

# Run
docker run -p 7860:7860 traffic-ai

# Test
curl http://localhost:7860/health
```

---

## Deployment

```bash
# Login to Hugging Face
python -c "import huggingface_hub; huggingface_hub.login(token='hf_...')"

# Push via git
git remote add origin https://USERNAME:HF_TOKEN@huggingface.co/spaces/Ronit-9/Traffic-Control-AI
git push origin main
```

---

## How the RL loop works

```
Agent observes [north, south, east, west] (normalised 0-1)
        ↓
Chooses action 0-3 (which lanes get green light)
        ↓
Environment clears served lanes, checks emergency, adds new arrivals
        ↓
Returns reward + new observation
        ↓
After 50,000 steps: agent learns to maximise reward
```

The agent learns three key behaviours:
1. **Clear the heaviest lanes first** — maximises +1 per vehicle cleared
2. **Always serve emergencies** — +50 bonus is too large to ignore
3. **Don't let any lane stay blocked** — −0.1 per waiting vehicle teaches urgency

---

*Built with OpenEnv · stable-baselines3 · FastAPI · Hugging Face Spaces*