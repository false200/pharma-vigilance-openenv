from typing import List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, ConfigDict, Field


class AdverseEventReport(BaseModel):
    model_config = ConfigDict(revalidate_instances="never")

    report_id: str = Field(..., description="Unique synthetic report identifier")
    patient_age: int = Field(..., description="Patient age in years")
    patient_sex: str = Field(..., description="Patient sex")
    drugs: List[str] = Field(default_factory=list, description="All drugs the patient was taking")
    suspect_drug: str = Field(..., description="Drug named as suspect by the original reporter")
    reaction: str = Field(..., description="Observed adverse reaction")
    onset_days: int = Field(..., description="Days from drug start to reaction onset")
    severity: str = Field(..., description="Reported case severity")
    outcome: str = Field(..., description="Clinical outcome status")
    similar_reports_last_30d: int = Field(..., description="Count of similar reports in the last 30 days")


class PharmaObservation(Observation):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
        revalidate_instances="never",
    )

    task_id: str = Field(..., description="Current task identifier")
    reports: List[AdverseEventReport] = Field(default_factory=list, description="Synthetic adverse event reports")
    drug_interaction_db: dict = Field(default_factory=dict, description="Hardcoded interaction and safety lookup")
    step_number: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=1, description="Maximum number of steps in the episode")
    feedback: Optional[str] = Field(default=None, description="Feedback after the previous action")

    reward: float = Field(default=0.0, description="Reward from the last action")
    done: bool = Field(default=False, description="Episode termination flag")
    metadata: dict = Field(default_factory=dict, description="Additional environment metadata")


class PharmaAction(Action):
    classification: str = Field(..., description="new_signal | known_side_effect | noise | duplicate")
    suspect_drug: str = Field(..., description="Drug or interaction believed to be causal")
    severity_assessment: str = Field(..., description="mild | moderate | severe | critical")
    recommended_action: str = Field(..., description="escalate | log_and_monitor | dismiss | request_more_info")
    reasoning: str = Field(default="", description="Short explanation of the decision")
    confidence: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Optional analyst confidence score from 0 to 100 for calibration-aware reward shaping",
    )


class PharmaReward(BaseModel):
    total: float = Field(..., description="Total reward in the 0.0-1.0 range")
    breakdown: dict = Field(default_factory=dict, description="Per-component reward breakdown")
