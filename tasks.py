from typing import Any, Callable, Dict, List

from pydantic import BaseModel

from data import TASK_DATA


class GroundTruth(BaseModel):
    classification: str
    suspect_drug: str
    severity_assessment: str
    recommended_action: str


class TaskDefinition(BaseModel):
    task_id: str
    difficulty: str
    reports: List[Any]
    drug_interaction_db: dict
    ground_truth: GroundTruth
    grader: Callable[[Any], Any]
    description: str

    model_config = {"arbitrary_types_allowed": True}


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


def known_signal_easy_grader(action: Any):
    truth = GroundTruth(**TASK_DATA["known_signal_easy"]["ground_truth"])
    breakdown = _base_breakdown(action, truth)
    return _reward_from_breakdown(breakdown)


def cluster_signal_medium_grader(action: Any):
    truth = GroundTruth(**TASK_DATA["cluster_signal_medium"]["ground_truth"])
    breakdown = _base_breakdown(action, truth)
    return _reward_from_breakdown(breakdown)


def confounded_hard_grader(action: Any):
    truth = GroundTruth(**TASK_DATA["confounded_hard"]["ground_truth"])
    breakdown = _base_breakdown(action, truth)
    reasoning = action.reasoning.lower()
    if any(
        term in reasoning
        for term in ("cyp3a4", "drug interaction", "statin", "atorvastatin", "clarithromycin")
    ):
        breakdown["reasoning_bonus"] = 0.15
    return _reward_from_breakdown(breakdown)


def get_tasks() -> Dict[str, TaskDefinition]:
    from env import AdverseEventReport

    return {
        "known_signal_easy": TaskDefinition(
            task_id="known_signal_easy",
            difficulty="easy",
            reports=[AdverseEventReport(**report) for report in TASK_DATA["known_signal_easy"]["reports"]],
            drug_interaction_db=TASK_DATA["known_signal_easy"]["drug_interaction_db"],
            ground_truth=GroundTruth(**TASK_DATA["known_signal_easy"]["ground_truth"]),
            grader=known_signal_easy_grader,
            description="Known ibuprofen GI bleeding case that should be logged and monitored.",
        ),
        "cluster_signal_medium": TaskDefinition(
            task_id="cluster_signal_medium",
            difficulty="medium",
            reports=[AdverseEventReport(**report) for report in TASK_DATA["cluster_signal_medium"]["reports"]],
            drug_interaction_db=TASK_DATA["cluster_signal_medium"]["drug_interaction_db"],
            ground_truth=GroundTruth(**TASK_DATA["cluster_signal_medium"]["ground_truth"]),
            grader=cluster_signal_medium_grader,
            description="Cluster of visual events around a new diabetes drug that should be escalated.",
        ),
        "confounded_hard": TaskDefinition(
            task_id="confounded_hard",
            difficulty="hard",
            reports=[AdverseEventReport(**report) for report in TASK_DATA["confounded_hard"]["reports"]],
            drug_interaction_db=TASK_DATA["confounded_hard"]["drug_interaction_db"],
            ground_truth=GroundTruth(**TASK_DATA["confounded_hard"]["ground_truth"]),
            grader=confounded_hard_grader,
            description="Polypharmacy case where the blamed drug is innocent and the real issue is an interaction.",
        ),
    }


def get_task(task_id: str) -> TaskDefinition:
    tasks = get_tasks()
    if task_id not in tasks:
        raise KeyError(f"Unknown task_id: {task_id}")
    return tasks[task_id]
