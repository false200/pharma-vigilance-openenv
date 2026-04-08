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
    confidence: Optional[int] = Field(default=None, ge=0, le=100)


class Reward(BaseModel):
    total: float = Field(..., ge=-1.0, le=1.0)
    breakdown: dict


class PharmaVigilanceEnv:
    def __init__(self):
        self.current_task: Optional[TaskDefinition] = None
        self.current_task_id: Optional[str] = None
        self.step_number = 0
        self.max_steps = 2
        self.last_action: Optional[dict] = None
        self.last_reward: Optional[dict] = None
        self.initial_action: Optional[Action] = None
        self.initial_reward: Optional[Reward] = None

    def _review_note(self) -> str:
        notes = {
            "known_signal_easy": (
                "Senior review note: labeling already documents ACE-inhibitor cough, "
                "and the recent case volume suggests this is a routine known-effect triage question."
            ),
            "cluster_signal_medium": (
                "Senior review note: the safety mailbox added 3 follow-up summaries showing "
                "symptomatic bradycardia with no competing causative drug class in common."
            ),
            "confounded_hard": (
                "Senior review note: tacrolimus trough levels returned at 4x baseline after "
                "recent voriconazole exposure, which is more mechanistically informative than the reporter's blamed drug."
            ),
        }
        return notes.get(self.current_task_id or "", "Senior review note: additional case review context is now available.")

    @staticmethod
    def _clamp_reward(total: float, breakdown: dict) -> Reward:
        return Reward(total=max(-0.25, min(1.0, round(total, 4))), breakdown=breakdown)

    def _initial_triage_reward(self, action: Action) -> Reward:
        truth = self.current_task.ground_truth
        action_suspect = action.suspect_drug.strip().lower()
        truth_suspect = truth.suspect_drug.strip().lower()
        suspect_match = (
            action_suspect == truth_suspect
            or action_suspect in truth_suspect
            or truth_suspect in action_suspect
        )

        breakdown = {
            "initial_classification": 0.15 if action.classification == truth.classification else 0.0,
            "initial_suspect_drug": 0.15 if suspect_match else 0.0,
            "initial_severity": 0.05 if action.severity_assessment == truth.severity_assessment else 0.0,
            "initial_action": 0.05 if action.recommended_action == truth.recommended_action else 0.0,
            "initial_false_alarm_penalty": 0.0,
            "initial_missed_signal_penalty": 0.0,
        }

        if action.classification == "new_signal" and truth.classification == "noise":
            breakdown["initial_false_alarm_penalty"] = -0.05
        if action.classification == "noise" and truth.classification == "new_signal":
            breakdown["initial_missed_signal_penalty"] = -0.10

        return self._clamp_reward(sum(breakdown.values()), breakdown)

    def _finalize_reward(self, action: Action) -> Reward:
        final_reward = self.current_task.action_grader(action)
        breakdown = dict(final_reward.breakdown)
        initial_total = self.initial_reward.total if self.initial_reward else 0.0

        breakdown["revision_bonus"] = 0.0
        breakdown["stubborn_penalty"] = 0.0
        breakdown["flip_penalty"] = 0.0

        if final_reward.total - initial_total >= 0.20:
            breakdown["revision_bonus"] = 0.05

        if (
            self.initial_action is not None
            and initial_total < 0.20
            and self.initial_action.classification == action.classification
            and self.initial_action.suspect_drug.strip().lower() == action.suspect_drug.strip().lower()
            and self.initial_action.recommended_action == action.recommended_action
        ):
            breakdown["stubborn_penalty"] = -0.05

        if self.initial_action is not None and initial_total >= 0.70 and initial_total - final_reward.total >= 0.25:
            breakdown["flip_penalty"] = -0.04

        return self._clamp_reward(sum(breakdown.values()), breakdown)

    def reset(self, task_id: str = "known_signal_easy") -> Observation:
        self.current_task = get_task(task_id)
        self.current_task_id = self.current_task.task_id
        self.step_number = 0
        self.last_action = None
        self.last_reward = None
        self.initial_action = None
        self.initial_reward = None
        return Observation(
            task_id=self.current_task.task_id,
            reports=self.current_task.reports,
            drug_interaction_db=self.current_task.drug_interaction_db,
            step_number=self.step_number,
            max_steps=self.max_steps,
            feedback="Task loaded. Submit an initial triage, then revise after senior review context arrives.",
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self.current_task is None:
            raise RuntimeError("Call reset() before step().")
        if self.step_number >= self.max_steps:
            raise RuntimeError("Episode already complete. Call reset() before another step().")

        if self.step_number == 0:
            reward = self._initial_triage_reward(action)
            self.initial_action = action
            self.initial_reward = reward
            self.step_number += 1
            self.last_action = action.model_dump()
            self.last_reward = reward.model_dump()
            done = False
            observation = Observation(
                task_id=self.current_task.task_id,
                reports=self.current_task.reports,
                drug_interaction_db=self.current_task.drug_interaction_db,
                step_number=self.step_number,
                max_steps=self.max_steps,
                feedback=(
                    "Initial triage recorded. "
                    f"{self._review_note()} "
                    "Review the added context and submit your final assessment."
                ),
            )
            info = {
                "phase": "initial_triage",
                "difficulty": self.current_task.difficulty,
                "reward_breakdown": reward.breakdown,
            }
            return observation, reward, done, info

        reward = self._finalize_reward(action)
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
            "phase": "final_review",
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
