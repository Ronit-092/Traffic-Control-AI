"""
graders/grader.py — Scores a task result.

Returns a normalised float in [0.0, 1.0] as required by OpenEnv.
"""


def grade(result: dict) -> float:
    """
    Args:
        result: dict with keys:
            reward    (float) — total episode reward
            em_total  (int)   — emergencies that occurred
            em_served (int)   — emergencies the agent handled
            steps     (int)   — episode length

    Returns:
        float score in [0.0, 1.0]
    """
    reward    = result.get("reward",    0.0)
    em_total  = result.get("em_total",  0)
    em_served = result.get("em_served", 0)
    steps     = result.get("steps",     10)

    per_step = reward / max(steps, 1)

    # Map per-step reward to 0-1 scale
    # per_step range roughly: -20 (bad) to +55 (perfect emergency serve)
    # We normalise against a reasonable max of 10.0 per step
    raw_score = min(max(per_step / 10.0, 0.0), 1.0)

    # Emergency response bonus: up to 0.1 extra if ≥80% emergencies served
    em_bonus = 0.0
    if em_total > 0 and (em_served / em_total) >= 0.80:
        em_bonus = 0.1

    score = min(raw_score + em_bonus, 1.0)
    return round(score, 3)