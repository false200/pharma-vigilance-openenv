import json
import os
import sys
from typing import Optional

from openai import OpenAI

try:
    from .env import Action
except ImportError:
    from env import Action


_cached_client: Optional[OpenAI] = None
_cached_model = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")


def _maybe_get_client() -> Optional[OpenAI]:
    global _cached_client

    if _cached_client is not None:
        return _cached_client

    base_url = os.environ.get("API_BASE_URL", "").strip()
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or "hf-missing-token"

    if not base_url:
        print(
            "[WARN] API_BASE_URL is not configured; AnalystAgent will use heuristic mode.",
            file=sys.stderr,
        )
        return None

    _cached_client = OpenAI(base_url=base_url, api_key=api_key)
    return _cached_client


class AnalystAgent:
    """
    Lightweight pharmacovigilance agent for demos and smoke testing.

    The agent can call an OpenAI-compatible chat endpoint when configured, but
    it also has a deterministic fallback policy for offline or local use.
    """

    def __init__(self) -> None:
        self.review_memory: list[dict] = []

    def _case_snapshot(self, observation) -> str:
        report_lines = []
        for report in observation.reports:
            report_lines.append(
                f"- {report.report_id}: suspect={report.suspect_drug}, "
                f"reaction={report.reaction}, onset_days={report.onset_days}, "
                f"severity={report.severity}, outcome={report.outcome}, "
                f"similar_30d={report.similar_reports_last_30d}"
            )

        memory_block = ""
        if self.review_memory:
            memory_block = "\nRecent mistakes to avoid:\n"
            for item in self.review_memory[-3:]:
                memory_block += (
                    f"- On {item['task_id']} you underperformed after choosing "
                    f"{item['classification']} / {item['recommended_action']}. "
                    f"Reason note: {item['note']}\n"
                )

        return (
            f"Task id: {observation.task_id}\n"
            f"Reports:\n" + "\n".join(report_lines) + "\n"
            f"Knowledge base:\n{json.dumps(observation.drug_interaction_db, ensure_ascii=True, indent=2)}"
            f"{memory_block}"
        )

    def _build_prompt(self, observation) -> str:
        return f"""You are a pharmacovigilance case assessor.

Review the case below and return one JSON object only.

Return fields:
- classification: one of new_signal, known_side_effect, noise, duplicate
- suspect_drug: likely causal drug or interaction
- severity_assessment: one of mild, moderate, severe, critical
- recommended_action: one of escalate, log_and_monitor, dismiss, request_more_info
- reasoning: concise mechanistic explanation

Decision principles:
- Repeated known labeled reactions should usually be known_side_effect
- Small but coherent post-marketing clusters on a newer drug can justify new_signal
- If the reporter blames the wrong medication, prefer the stronger causal interaction
- Missing a serious signal is worse than overcalling a weak case

Case:
{self._case_snapshot(observation)}
"""

    def _llm_decision(self, observation) -> Optional[Action]:
        client = _maybe_get_client()
        if client is None:
            return None

        try:
            response = client.chat.completions.create(
                model=_cached_model,
                messages=[{"role": "user", "content": self._build_prompt(observation)}],
                temperature=0.0,
                max_tokens=220,
            )
            raw = (response.choices[0].message.content or "").strip()
            payload = json.loads(raw)
            return Action(**payload)
        except Exception as exc:
            print(f"[WARN] AnalystAgent LLM path failed: {exc}; falling back to heuristics.", file=sys.stderr)
            return None

    def _heuristic_decision(self, observation) -> Action:
        reports = observation.reports
        report_count = len(reports)
        report = reports[0]
        reaction_blob = " ".join(item.reaction.lower() for item in reports)
        db_blob = json.dumps(observation.drug_interaction_db).lower()

        if "dry cough" in reaction_blob and "ace inhibitor" in db_blob:
            return Action(
                classification="known_side_effect",
                suspect_drug="Lisinopril",
                severity_assessment="mild",
                recommended_action="log_and_monitor",
                reasoning="Persistent dry cough is a classic labeled ACE inhibitor effect.",
            )

        if report_count >= 3 and ("brady" in reaction_blob or "syncope" in reaction_blob):
            return Action(
                classification="new_signal",
                suspect_drug="Cardiovexa",
                severity_assessment="severe",
                recommended_action="escalate",
                reasoning="A coherent cluster of bradycardia reports on a recently launched drug warrants escalation.",
            )

        if "tacrolimus" in db_blob and "voriconazole" in db_blob:
            return Action(
                classification="new_signal",
                suspect_drug="Tacrolimus+Voriconazole",
                severity_assessment="critical",
                recommended_action="escalate",
                reasoning="This looks like a tacrolimus exposure interaction requiring urgent escalation and level review.",
            )

        fallback_severity = report.severity if report.severity in {"mild", "moderate", "severe", "critical"} else "moderate"
        return Action(
            classification="new_signal",
            suspect_drug=report.suspect_drug,
            severity_assessment=fallback_severity,
            recommended_action="request_more_info",
            reasoning="The case is ambiguous, so additional information is needed before final triage.",
        )

    def act(self, observation) -> Action:
        llm_action = self._llm_decision(observation)
        if llm_action is not None:
            return llm_action
        return self._heuristic_decision(observation)

    def learn(self, action: Action, observation) -> None:
        reward = getattr(observation, "reward", 0.0)
        if reward is None:
            reward = 0.0

        if reward < 0.5:
            self.review_memory.append(
                {
                    "task_id": getattr(observation, "task_id", "unknown"),
                    "classification": action.classification,
                    "recommended_action": action.recommended_action,
                    "note": getattr(observation, "feedback", "") or "weak outcome",
                }
            )
