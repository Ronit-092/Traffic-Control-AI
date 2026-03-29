"""graders/grader.py — Per-step normalised scoring."""


def grade(result: dict) -> int:
    reward    = result.get("reward",    0.0)
    em_total  = result.get("em_total",  0)
    em_served = result.get("em_served", 0)
    steps     = result.get("steps",     10)

    per_step = reward / max(steps, 1)

    if   per_step >= 6.0: score = 10
    elif per_step >= 3.5: score = 8
    elif per_step >= 1.5: score = 6
    elif per_step >= 0.0: score = 4
    else:                 score = 2

    if em_total > 0 and (em_served / em_total) >= 0.80:
        score = min(score + 1, 10)

    return score
