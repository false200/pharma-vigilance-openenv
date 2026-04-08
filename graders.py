"""Public grader entrypoints for OpenEnv validation and judging."""

from server.graders import (
    cluster_signal_medium_grader,
    confounded_hard_grader,
    easy_grader,
    hard_grader,
    known_signal_easy_grader,
    medium_grader,
)

TASK_TO_GRADER = {
    "known_signal_easy": known_signal_easy_grader,
    "cluster_signal_medium": cluster_signal_medium_grader,
    "confounded_hard": confounded_hard_grader,
}

TIER_TO_GRADER = {
    "easy": easy_grader,
    "medium": medium_grader,
    "hard": hard_grader,
}

__all__ = [
    "TASK_TO_GRADER",
    "TIER_TO_GRADER",
    "easy_grader",
    "medium_grader",
    "hard_grader",
    "known_signal_easy_grader",
    "cluster_signal_medium_grader",
    "confounded_hard_grader",
]
