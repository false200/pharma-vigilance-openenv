from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from tasks import TaskDefinition, get_task


class AdverseEventReport(BaseModel):
    report_id: str
    patient_age: int
    patient_sex: str
    drugs: List[str]
    suspect_drug: str
    reaction: str
    onset_days: int
    severity: str
    outcome: str
    similar_reports_last_30d: int


class Observation(BaseModel):
    task_id: str
    reports: List[AdverseEventReport]
    drug_interaction_db: dict
    step_number: int
    max_steps: int
    feedback: Optional[str] = None


class Action(BaseModel):
    classification: str
    suspect_drug: str
    severity_assessment: str
    recommended_action: str
    reasoning: str


class Reward(BaseModel):
    total: float = Field(..., ge=0.0, le=1.0)
    breakdown: dict


class PharmaVigilanceEnv:
    def __init__(self):
        self.current_task: Optional[TaskDefinition] = None
        self.current_task_id: Optional[str] = None
        self.step_number = 0
        self.max_steps = 1
        self.last_action: Optional[dict] = None
        self.last_reward: Optional[dict] = None

    def reset(self, task_id: str = "known_signal_easy") -> Observation:
        self.current_task = get_task(task_id)
        self.current_task_id = self.current_task.task_id
        self.step_number = 0
        self.last_action = None
        self.last_reward = None
        return Observation(
            task_id=self.current_task.task_id,
            reports=self.current_task.reports,
            drug_interaction_db=self.current_task.drug_interaction_db,
            step_number=self.step_number,
            max_steps=self.max_steps,
            feedback="Task loaded. Submit one final pharmacovigilance assessment.",
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self.current_task is None:
            raise RuntimeError("Call reset() before step().")

        reward = self.current_task.action_grader(action)
        self.step_number += 1
        self.last_action = action.model_dump()
        self.last_reward = reward.model_dump()
        done = True

        matched = sum(
            1
            for field in (
                "classification",
                "suspect_drug",
                "severity_assessment",
                "recommended_action",
            )
            if reward.breakdown.get(field, 0.0) > 0
        )

        if reward.total >= 0.9:
            feedback = "Strong assessment. The key safety judgment and follow-up were correct."
        elif reward.total >= 0.5:
            feedback = "Partially correct assessment. Some causal or operational details were missed."
        else:
            feedback = "Weak assessment. This case would need human analyst correction."

        observation = Observation(
            task_id=self.current_task.task_id,
            reports=self.current_task.reports,
            drug_interaction_db=self.current_task.drug_interaction_db,
            step_number=self.step_number,
            max_steps=self.max_steps,
            feedback=feedback,
        )
        info = {
            "matched_fields": matched,
            "difficulty": self.current_task.difficulty,
            "reward_breakdown": reward.breakdown,
        }
        return observation, reward, done, info

    def state(self) -> dict:
        return {
            "task_id": self.current_task_id,
            "step_number": self.step_number,
            "last_action": self.last_action,
            "last_reward": self.last_reward,
        }
