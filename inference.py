"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

import requests
from openai import OpenAI


IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("PHARMA_VIGILANCE_TASK", "")
BENCHMARK = os.getenv("PHARMA_VIGILANCE_BENCHMARK", "pharma-vigilance")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
MAX_STEPS = 1
TEMPERATURE = 0.0
MAX_TOKENS = 250
SUCCESS_SCORE_THRESHOLD = 0.75
TASKS = [TASK_NAME] if TASK_NAME else ["known_signal_easy", "cluster_signal_medium", "confounded_hard"]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a pharmacovigilance safety analyst.
    You will receive synthetic adverse event reports plus a hardcoded drug interaction database.
    Respond with exactly one valid JSON object and no extra text.

    Required JSON schema:
    {
      "classification": "new_signal | known_side_effect | noise | duplicate",
      "suspect_drug": "string",
      "severity_assessment": "mild | moderate | severe | critical",
      "recommended_action": "escalate | log_and_monitor | dismiss | request_more_info",
      "reasoning": "short explanation"
    }
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def build_user_prompt(observation: dict) -> str:
    return textwrap.dedent(
        f"""
        Review this pharmacovigilance case and return a final assessment.

        Observation:
        {json.dumps(observation, ensure_ascii=True, indent=2)}

        Pick the best classification, identify the most likely suspect drug or interaction,
        assess severity, choose the correct operational action, and explain your reasoning.
        Return only valid JSON.
        """
    ).strip()


def get_model_action(client: OpenAI, observation: dict) -> dict:
    user_prompt = build_user_prompt(observation)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    text = (completion.choices[0].message.content or "").strip()
    return json.loads(text)


def action_to_log_string(action: dict) -> str:
    classification = action.get("classification", "unknown")
    suspect_drug = action.get("suspect_drug", "")
    return f"{classification}/{suspect_drug}" if suspect_drug else classification


async def post_json(url: str, payload: dict) -> dict:
    def _call() -> dict:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    return await asyncio.to_thread(_call)


async def run_task(client: OpenAI, task_name: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_payload = {"task_id": task_name}
        observation = await post_json(f"{ENV_URL}/reset", reset_payload)

        for step in range(1, MAX_STEPS + 1):
            action = get_model_action(client, observation)
            action_str = action_to_log_string(action)

            step_payload = await post_json(f"{ENV_URL}/step", action)
            reward_obj = step_payload.get("reward", {})
            reward = float(reward_obj.get("total", 0.0))
            done = bool(step_payload.get("done", False))
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

            observation = step_payload.get("observation", {})

        score = min(max(sum(rewards), 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except json.JSONDecodeError:
        log_step(step=1, action="parse_error", reward=0.00, done=True, error="parse_error")
    except Exception as exc:
        step_num = 1 if steps_taken == 0 else steps_taken
        log_step(step=step_num, action="error", reward=0.00, done=True, error=str(exc))
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards or [0.0])


async def main() -> None:
    if not API_KEY:
        raise EnvironmentError("HF_TOKEN or API_KEY must be set before running inference.py")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_name in TASKS:
        await run_task(client, task_name)


if __name__ == "__main__":
    asyncio.run(main())
