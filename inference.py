"""
Baseline runner for the Pharmacovigilance Signal Detector submission.

This script queries a chat model through the OpenAI client, sends its decision
to the environment server, and prints the exact machine-readable lines expected
by the evaluator.
"""

import argparse
import json
import os
from typing import Iterable, List

import requests
from openai import OpenAI
from pydantic import ValidationError

try:
    from .models import PharmaAction
except ImportError:
    from models import PharmaAction


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")
TASK_OVERRIDE = os.getenv("TASK_NAME", "").strip()
BENCHMARK = "pharma-vigilance"

TASK_SETS = {
    "easy": ("known_signal_easy",),
    "medium": ("cluster_signal_medium",),
    "hard": ("confounded_hard",),
    "all": ("known_signal_easy", "cluster_signal_medium", "confounded_hard"),
}

SYSTEM_MESSAGE = """
You are acting as a pharmacovigilance triage analyst.

Read the synthetic case bundle and reply with exactly one JSON object.
Allowed keys:
- classification
- suspect_drug
- severity_assessment
- recommended_action
- reasoning
- confidence

Allowed values:
- classification: new_signal, known_side_effect, noise, duplicate
- severity_assessment: mild, moderate, severe, critical
- recommended_action: escalate, log_and_monitor, dismiss, request_more_info
- confidence: integer from 0 to 100

No markdown. No explanation outside the JSON object.
""".strip()


def emit_start(task_name: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def emit_step(step_no: int, action_text: str, reward: float, done: bool, error: str | None) -> None:
    error_text = error if error else "null"
    print(
        f"[STEP] step={step_no} action={action_text} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_text}",
        flush=True,
    )


def emit_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    reward_text = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.6f} rewards={reward_text}",
        flush=True,
    )


def choose_tasks(selection: str) -> Iterable[str]:
    if TASK_OVERRIDE:
        return (TASK_OVERRIDE,)
    return TASK_SETS[selection]


def client() -> OpenAI:
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN or API_KEY must be set before running inference.py")
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def fetch_reset(task_name: str) -> dict:
    response = requests.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_name},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def submit_action(action: PharmaAction) -> dict:
    response = requests.post(
        f"{ENV_URL}/step",
        json={"action": action.model_dump()},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def prompt_for_case(observation: dict) -> str:
    return (
        "Assess the following pharmacovigilance case.\n\n"
        "Return one final structured judgment.\n\n"
        f"{json.dumps(observation, ensure_ascii=True, indent=2)}\n\n"
        "Focus on whether the case is novel or known, the most plausible causal "
        "drug or interaction, the right severity band, and the operational next step."
    )


def ask_model(llm: OpenAI, observation: dict) -> PharmaAction:
    completion = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt_for_case(observation)},
        ],
        temperature=0.0,
        max_tokens=260,
        stream=False,
    )
    text = (completion.choices[0].message.content or "").strip()
    payload = json.loads(text)
    return PharmaAction(**payload)


def compact_action(action: PharmaAction) -> str:
    label = action.classification
    if action.suspect_drug:
        return f"{label}/{action.suspect_drug}"
    return label


def final_score(rewards: List[float]) -> float:
    score = sum(rewards) / len(rewards) if rewards else 0.0
    return min(max(round(score, 4), 0.01), 0.99)


def run_one_task(llm: OpenAI, task_name: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    emit_start(task_name)

    try:
        result = fetch_reset(task_name)
        done = bool(result.get("done", False))

        while not done:
            observation = result
            action = ask_model(llm, observation)
            action_text = compact_action(action)

            result = submit_action(action)
            reward_payload = result.get("reward", {})
            reward = (
                float(reward_payload.get("total", 0.0))
                if isinstance(reward_payload, dict)
                else float(reward_payload)
            )
            done = bool(result.get("done", False))

            rewards.append(reward)
            steps_taken += 1
            emit_step(steps_taken, action_text, reward, done, None)

        score = final_score(rewards)
        success = score >= 0.60

    except json.JSONDecodeError:
        rewards = [0.0]
        steps_taken = 1
        emit_step(1, "parse_error", 0.0, True, "parse_error")
    except ValidationError:
        rewards = [0.0]
        steps_taken = 1
        emit_step(1, "schema_error", 0.0, True, "schema_error")
    except Exception as exc:
        rewards = [0.0]
        steps_taken = 1
        emit_step(1, "error", 0.0, True, str(exc))
    finally:
        emit_end(success, steps_taken, score, rewards or [0.0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pharmacovigilance baseline agent")
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Which task subset to run",
    )
    args = parser.parse_args()

    llm = client()
    for task_name in choose_tasks(args.difficulty):
        run_one_task(llm, task_name)


if __name__ == "__main__":
    main()
