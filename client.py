"""
client.py — Typed WebSocket client.

Implements the 3 abstract methods from EnvClient exactly as shown in Module 4.

Usage:
    from client import TrafficEnv
    from models import TrafficAction

    # From Hugging Face (after openenv push)
    with TrafficEnv.from_hub("Ronit-9/traffic-ai") as env:
        result = env.reset()
        result = env.step(TrafficAction(action_id=0))
        state  = env.state()

    # Local server
    with TrafficEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
"""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from models import TrafficAction, TrafficObservation, TrafficState


class TrafficEnv(EnvClient[TrafficAction, TrafficObservation, TrafficState]):

    def _step_payload(self, action: TrafficAction) -> dict:
        return {"action_id": action.action_id}

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", payload)
        return StepResult(
            observation = TrafficObservation(
                done           = payload.get("done", False),
                reward         = payload.get("reward"),
                traffic_counts = obs_data.get("traffic_counts", {}),
                obs_normalized = obs_data.get("obs_normalized", [0.0]*4),
                emergency      = obs_data.get("emergency", {}),
                junction_type  = obs_data.get("junction_type", "cross"),
                message        = obs_data.get("message", ""),
            ),
            reward = payload.get("reward"),
            done   = payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> TrafficState:
        return TrafficState(
            episode_id   = payload.get("episode_id", ""),
            step_count   = payload.get("step_count", 0),
            junction_type= payload.get("junction_type", ""),
            total_reward = payload.get("total_reward", 0.0),
            em_total     = payload.get("em_total", 0),
            em_served    = payload.get("em_served", 0),
        )
