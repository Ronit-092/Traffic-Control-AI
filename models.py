"""
models.py — Action, Observation, State types.

Inherits from openenv.core.env_server base classes (Pydantic models).
"""

from typing import Dict, List, Optional
from openenv.core.env_server import Action, Observation, State


class TrafficAction(Action):
    """
    Signal action for the traffic controller.

    Cross  : 0=NS  1=EW  2=N-only  3=E-only
    T-junc : 0=NS  1=E   2=N-only  3=S-only
    Y-junc : 0=N   1=E   2=W       3=NE
    """
    action_id: int


class TrafficObservation(Observation):
    """
    Observation returned after every reset() and step().

    done and reward are inherited from Observation base class.
    """
    traffic_counts: Dict[str, int]   # {"north": 4, "south": 0, ...}
    obs_normalized: List[float]      # [0.13, 0.0, 0.27, 0.0]  — for RL agents
    emergency:      Dict[str, bool]  # {"north": False, "east": True, ...}
    junction_type:  str              # "cross" | "T" | "Y"
    message:        str = ""         # human-readable step description


class TrafficState(State):
    """
    Episode-level metadata.

    episode_id and step_count are inherited from State base class.
    """
    junction_type:   str   = ""
    total_reward:    float = 0.0
    em_total:        int   = 0
    em_served:       int   = 0
