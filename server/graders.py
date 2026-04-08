"""
Trajectory scorers for the Pharmacovigilance Signal Detector.

These functions are intentionally pharmacovigilance-specific rather than
generic "reward bucket" adapters. The scoring rubric emphasizes:

1. Signal sensitivity: missing a true novel safety signal is costly.
2. Operational judgment: escalation/log/dismiss choices matter independently.
3. Causal calibration: high scores should reflect not just suspicion, but
   identifying the right drug or interaction.

All public grader outputs are forced into the judge-safe interval (0.01, 0.99).
"""

from typing import Any, Iterable, List


STRICT_MIN = 0.01
STRICT_MAX = 0.99


def _bounded(value: float) -> float:
    return min(max(round(value, 4), STRICT_MIN), STRICT_MAX)


def _as_reward_list(trajectory: dict | None) -> List[float]:
    payload = trajectory or {}

    rewards = payload.get("rewards")
    if isinstance(rewards, list) and rewards:
        return [float(item) for item in rewards]

    if "score" in payload:
        return [float(payload["score"])]

    reward = payload.get("reward")
    if isinstance(reward, dict) and "total" in reward:
        return [float(reward["total"])]
    if reward is not None:
        return [float(reward)]

    return []


def _reward_profile(reward: float) -> str:
    """
    Translate a step reward into a pharmacovigilance interpretation bucket.

    This keeps the grader coupled to the meaning of the environment rather than
    to borrowed labels from a different domain.
    """
    if reward <= 0.05:
        return "unsafe_miss"
    if reward <= 0.20:
        return "bad_call"
    if reward < 0.50:
        return "weak_triage"
    if reward < 0.80:
        return "workable_triage"
    if reward < 0.95:
        return "strong_triage"
    return "expert_triage"


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.5
    return sum(items) / len(items)


def _score_episode(
    rewards: List[float],
    *,
    miss_cost: float,
    overcall_cost: float,
    stability_gain: float,
    expertise_gain: float,
) -> float:
    if not rewards:
        return 0.5

    labels = [_reward_profile(reward) for reward in rewards]
    mean_reward = _mean(rewards)
    total_steps = len(rewards)

    unsafe_miss_count = labels.count("unsafe_miss")
    bad_call_count = labels.count("bad_call")
    weak_count = labels.count("weak_triage")
    strong_count = labels.count("strong_triage") + labels.count("expert_triage")
    expert_count = labels.count("expert_triage")

    downward_pressure = (
        min(unsafe_miss_count * miss_cost, 0.35)
        + min(bad_call_count * overcall_cost, 0.15)
        + min(weak_count * 0.015, 0.06)
    )

    upward_pressure = 0.0
    if strong_count / total_steps >= 0.80:
        upward_pressure += stability_gain
    if expert_count / total_steps >= 0.60:
        upward_pressure += expertise_gain

    return _bounded(mean_reward - downward_pressure + upward_pressure)


def easy_grader(trajectory: dict = None) -> float:
    """
    Easy tier: obvious known-signal recognition and straightforward handling.

    The scorer expects high reliability here. Weak or missed judgments are
    penalized more sharply because these are the least ambiguous cases.
    """
    rewards = _as_reward_list(trajectory)
    return _score_episode(
        rewards,
        miss_cost=0.12,
        overcall_cost=0.03,
        stability_gain=0.05,
        expertise_gain=0.01,
    )


def medium_grader(trajectory: dict = None) -> float:
    """
    Medium tier: cluster recognition and escalation readiness.

    These cases reward agents that can move from single-case thinking to
    population-level signal interpretation.
    """
    rewards = _as_reward_list(trajectory)
    return _score_episode(
        rewards,
        miss_cost=0.09,
        overcall_cost=0.04,
        stability_gain=0.03,
        expertise_gain=0.02,
    )


def hard_grader(trajectory: dict = None) -> float:
    """
    Hard tier: confounding, blame reassignment, and interaction reasoning.

    The hard scorer gives extra value to near-expert trajectories because this
    tier is specifically designed to separate shallow pattern matching from
    mechanistic causal reasoning.
    """
    rewards = _as_reward_list(trajectory)
    return _score_episode(
        rewards,
        miss_cost=0.07,
        overcall_cost=0.03,
        stability_gain=0.02,
        expertise_gain=0.04,
    )


def known_signal_easy_grader(trajectory: dict = None) -> float:
    return easy_grader(trajectory)


def cluster_signal_medium_grader(trajectory: dict = None) -> float:
    return medium_grader(trajectory)


def confounded_hard_grader(trajectory: dict = None) -> float:
    return hard_grader(trajectory)


__all__ = [
    "easy_grader",
    "medium_grader",
    "hard_grader",
    "known_signal_easy_grader",
    "cluster_signal_medium_grader",
    "confounded_hard_grader",
]
