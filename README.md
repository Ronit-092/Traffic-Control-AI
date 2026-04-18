---
title: Traffic AI Signal Control
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
app_file: server/app.py
pinned: false
---

# 🚦 Traffic-AI — Intelligent Traffic Signal Control

A Reinforcement Learning environment where an AI agent learns to control
traffic signals at road junctions, maximising vehicle throughput and
prioritising emergency vehicles.

Live demo: **[huggingface.co/spaces/Ronit-9/Traffic-Control-AI](https://huggingface.co/spaces/Ronit-9/Traffic-Control-AI)**
---

## Overview

The agent observes vehicle queue lengths at four directions (North, South,
East, West) and decides which direction to give a green light. It is
rewarded for clearing vehicles quickly and strongly rewarded for serving
emergency vehicles first.

Three junction types are supported: **Cross** (4-way), **T-junction**, and
**Y-junction**. The agent handles all three dynamically within the same
episode.

---

## Project Structure

```
traffic-ai/
├── models.py                 ← Pydantic Action / Observation / State
├── client.py                 ← WebSocket client
├── openenv.yaml              ← Environment manifest
├── pyproject.toml
├── inference.py              ← LLM agent baseline
│
├── server/
│   ├── environment.py        ← Core simulation logic
│   ├── app.py                ← FastAPI server
│   ├── requirements.txt
│   └── Dockerfile
│
├── tasks/
│   ├── task_easy.py          ← NS vs EW heuristic (10 steps)
│   ├── task_medium.py        ← Emergency-aware scorer (15 steps)
│   └── task_hard.py          ← Lookahead planner + surges (20 steps)
│
├── graders/
│   └── grader.py             ← Normalised scoring [0, 1]
│
├── gymnasium_adapter.py      ← Gymnasium wrapper for PPO training
├── train.py                  ← PPO training script
├── agent.py                  ← Evaluation script
├── app.py                    ← Rule-based vs PPO comparison
└── visualize.py              ← Live ASCII visualiser
```

---

## Quickstart

```bash
pip install -r server/requirements.txt
python train.py
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/reset` | Start new episode |
| POST | `/step` | Take signal action |
| GET | `/state` | Episode metadata |
| GET | `/baseline` | Run all 3 tasks, return scores |
| POST | `/grader` | Grade a result |
| GET | `/tasks` | List available tasks |
| GET | `/docs` | Swagger UI |

---

## Reward Function

| Event | Reward |
|-------|--------|
| Vehicle cleared | +1 per vehicle |
| Emergency vehicle served | +50 |
| Emergency vehicle ignored | −10 |
| Vehicle still waiting | −0.1 per vehicle |

---

## Tasks

| Task | Algorithm | Steps |
|------|-----------|-------|
| Easy | NS vs EW heuristic | 10 |
| Medium | Emergency-aware quadratic scorer | 15 |
| Hard | 1-step lookahead planner + traffic surges | 20 |

---

## Training

```bash
python train.py     # trains PPO for 50k steps
python agent.py     # evaluate trained model
python app.py       # compare rule-based vs PPO
python visualize.py # live ASCII visualiser
```

---

## Docker

```bash
docker build -t traffic-ai .
docker run -p 7860:7860 traffic-ai
```