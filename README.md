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

**ScalarX × Meta Hackathon 2026**

A Reinforcement Learning environment built on the **OpenEnv standard**.
A PPO agent learns to control traffic signals at Cross, T, and Y junctions —
maximising throughput and prioritising emergency vehicles.

## Project structure

```
traffic-ai/
├── models.py                 ← Action / Observation / State (OpenEnv types)
├── client.py                 ← EnvClient WebSocket client
├── openenv.yaml              ← Manifest
├── pyproject.toml            ← uv project config
├── server/
│   ├── environment.py        ← TrafficEnvironment (game logic)
│   ├── app.py                ← create_fastapi_app() server
│   ├── requirements.txt
│   └── Dockerfile
├── tasks/
│   ├── task_easy.py          ← NS vs EW heuristic (10 steps)
│   ├── task_medium.py        ← Emergency-aware scorer (15 steps)
│   └── task_hard.py          ← 1-step lookahead planner (20 steps)
├── graders/grader.py         ← Per-step normalised scoring (2-10)
├── gymnasium_adapter.py      ← Gymnasium wrapper for PPO
├── train.py                  ← PPO training
├── agent.py                  ← Evaluation
├── app.py                    ← Rule-based vs PPO comparison
└── visualize.py              ← Live ASCII visualiser
```

## Quickstart

```bash
pip install -r server/requirements.txt
python train.py
uvicorn server.app:app --host 0.0.0.0 --port 8000
python app.py
```

## Reward function

| Event | Reward |
|---|---|
| Vehicle cleared | +1 per vehicle |
| Emergency served | +50 |
| Emergency ignored | −20 |
| Vehicle still waiting | −0.5 per vehicle |

## OpenEnv endpoints

- `GET  /health`
- `POST /reset`
- `POST /step`   body: `{"action_id": 0}`
- `GET  /state`
- `GET  /docs`   (Swagger UI)
