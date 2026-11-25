"""Deterministic scoring used by the scoring-rubric skill."""


def weighted_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted average of competency scores (0-10), rounded to one decimal place."""
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0
    total = sum(scores.get(name, 0.0) * weight for name, weight in weights.items())
    return round(total / total_weight, 1)
