import random
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from data import TASK_DATA


class GroundTruth(BaseModel):
    classification: str
    suspect_drug: str
    severity_assessment: str
    recommended_action: str


class TaskDefinition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, revalidate_instances="never")

    task_id: str = Field(..., description="Unique pharmacovigilance task identifier")
    difficulty: str = Field(..., description="easy | medium | hard")
    reports: List[Any] = Field(default_factory=list, description="Synthetic adverse event reports")
    drug_interaction_db: dict = Field(default_factory=dict, description="Hardcoded interaction and safety context")
    ground_truth: GroundTruth
    action_grader: Callable[[Any], Any]
    description: str = Field(default="", description="Human-readable task summary")

    @property
    def id(self) -> str:
        return self.task_id


def _base_breakdown(action: Any, ground_truth: GroundTruth) -> dict:
    action_suspect = action.suspect_drug.strip().lower()
    truth_suspect = ground_truth.suspect_drug.strip().lower()
    suspect_match = (
        action_suspect == truth_suspect
        or action_suspect in truth_suspect
        or truth_suspect in action_suspect
    )

    breakdown = {
        "classification": 0.25 if action.classification == ground_truth.classification else 0.0,
        "suspect_drug": 0.25 if suspect_match else 0.0,
        "severity_assessment": 0.25 if action.severity_assessment == ground_truth.severity_assessment else 0.0,
        "recommended_action": 0.25 if action.recommended_action == ground_truth.recommended_action else 0.0,
        "false_alarm_penalty": 0.0,
        "missed_signal_penalty": 0.0,
        "reasoning_bonus": 0.0,
    }

    if action.classification == "new_signal" and ground_truth.classification == "noise":
        breakdown["false_alarm_penalty"] = -0.10
    if action.classification == "noise" and ground_truth.classification == "new_signal":
        breakdown["missed_signal_penalty"] = -0.20

    return breakdown


def _reward_from_breakdown(breakdown: dict):
    from env import Reward

    total = round(sum(breakdown.values()), 4)
    return Reward(total=max(0.0, min(1.0, total)), breakdown=breakdown)


def known_signal_easy_action_grader(action: Any):
    truth = GroundTruth(**TASK_DATA["known_signal_easy"]["ground_truth"])
    breakdown = _base_breakdown(action, truth)
    return _reward_from_breakdown(breakdown)


def cluster_signal_medium_action_grader(action: Any):
    truth = GroundTruth(**TASK_DATA["cluster_signal_medium"]["ground_truth"])
    breakdown = _base_breakdown(action, truth)
    return _reward_from_breakdown(breakdown)


def confounded_hard_action_grader(action: Any):
    truth = GroundTruth(**TASK_DATA["confounded_hard"]["ground_truth"])
    breakdown = _base_breakdown(action, truth)
    reasoning = action.reasoning.lower()
    if any(
        term in reasoning
        for term in ("drug interaction", "tacrolimus", "voriconazole", "azole", "calcineurin", "level monitoring")
    ):
        breakdown["reasoning_bonus"] = 0.15
    return _reward_from_breakdown(breakdown)


def _grader_score_from_trajectory(trajectory: Any = None) -> float:
    trajectory = trajectory or {}
    raw_score = 0.5

    if isinstance(trajectory, dict):
        if "score" in trajectory:
            raw_score = float(trajectory["score"])
        elif "rewards" in trajectory and trajectory["rewards"]:
            raw_score = float(trajectory["rewards"][-1])
        elif "reward" in trajectory:
            reward_val = trajectory["reward"]
            if isinstance(reward_val, dict) and "total" in reward_val:
                raw_score = float(reward_val["total"])
            else:
                raw_score = float(reward_val)

    return max(0.01, min(0.99, round(raw_score, 4)))


def known_signal_easy_grader(trajectory: Any = None) -> float:
    from server.graders import known_signal_easy_grader as _delegate

    return _delegate(trajectory)


def cluster_signal_medium_grader(trajectory: Any = None) -> float:
    from server.graders import cluster_signal_medium_grader as _delegate

    return _delegate(trajectory)


def confounded_hard_grader(trajectory: Any = None) -> float:
    from server.graders import confounded_hard_grader as _delegate

    return _delegate(trajectory)


def _task_definition(
    task_id: str,
    difficulty: str,
    description: str,
    action_grader: Callable[[Any], Any],
) -> TaskDefinition:
    from env import AdverseEventReport

    task_data = TASK_DATA[task_id]
    return TaskDefinition(
        task_id=task_id,
        difficulty=difficulty,
        reports=[AdverseEventReport(**report) for report in task_data["reports"]],
        drug_interaction_db=task_data["drug_interaction_db"],
        ground_truth=GroundTruth(**task_data["ground_truth"]),
        action_grader=action_grader,
        description=description,
    )


def _build_all_tasks() -> Dict[str, List[TaskDefinition]]:
    """Build and return the complete task pool grouped by difficulty."""
    return {
        "easy": [
            _task_definition(
                "known_signal_easy",
                "easy",
                "Known ACE-inhibitor cough case that should be logged and monitored rather than escalated.",
                known_signal_easy_action_grader,
            ),
        ],
        "medium": [
            _task_definition(
                "cluster_signal_medium",
                "medium",
                "Cluster of bradycardia reports around a newly approved therapy that should be escalated as a signal.",
                cluster_signal_medium_action_grader,
            ),
        ],
        "hard": [
            _task_definition(
                "confounded_hard",
                "hard",
                "Confounded transplant case where the blamed drug is wrong and the real problem is a tacrolimus interaction.",
                confounded_hard_action_grader,
            ),
        ],
    }


def get_tasks(
    difficulty: Optional[str] = None,
    seed: Optional[int] = None,
    n: int = 5,
    grouped: bool = False,
):
    """
    Return tasks either as a flat task-id map or a difficulty-filtered list.

    Args:
        difficulty: Optional difficulty tier to select from.
        seed: Optional seed for reproducible shuffling within a difficulty pool.
        n: Maximum number of tasks to return when selecting by difficulty.
        grouped: When True and difficulty is None, return the difficulty-grouped dict.

    Returns:
        If grouped=True and difficulty is None:
            Dict[str, List[TaskDefinition]]
        If difficulty is None:
            Dict[str, TaskDefinition]
        Otherwise:
            List[TaskDefinition]
    """
    all_tasks = _build_all_tasks()

    if difficulty is None:
        if grouped:
            return {level: tasks[:n] for level, tasks in all_tasks.items()}
        return {
            task.task_id: task
            for tasks in all_tasks.values()
            for task in tasks[:n]
        }

    pool = list(all_tasks.get(difficulty, []))
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(pool)
    return pool[:n]


def get_task(task_id: str) -> TaskDefinition:
    tasks = get_tasks()
    if task_id not in tasks:
        raise KeyError(f"Unknown task_id: {task_id}")
    return tasks[task_id]
