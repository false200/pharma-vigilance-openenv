---
title: Pharmacovigilance Signal Detector
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: OpenEnv pharmacovigilance signal detection environment
tags:
  - openenv
  - healthcare
  - pharmacovigilance
  - safety
  - real-world
---

# Pharmacovigilance Signal Detector

`Pharmacovigilance Signal Detector` is a real-world OpenEnv environment where an agent acts like a drug-safety analyst. The agent reviews synthetic adverse event reports, uses a hardcoded drug interaction knowledge base, and decides whether the case is a new safety signal, a known side effect, or low-value noise. This mirrors pharmacovigilance triage work performed by regulators and pharmaceutical safety teams.

All case data in this repo is synthetic. No real patient data is used.

## Why This Environment Matters

Pharmacovigilance teams are responsible for detecting harmful safety patterns after a drug is already on the market. That work is operationally important, high-stakes, and difficult: analysts must distinguish expected reactions from true emerging risks, recognize confounding from polypharmacy, and escalate only when justified. This makes the domain a strong fit for agent evaluation because it tests causal reasoning, prioritization, and safety-sensitive decision making.

## Environment Overview

| Item | Value |
|---|---|
| Environment name | `pharma-vigilance` |
| Domain | Pharmacovigilance / drug safety triage |
| Episode length | 1 step per task |
| Task count | 3 |
| Difficulties | Easy, Medium, Hard |
| Reward range | `0.0` to `1.0` |
| API | `reset()`, `step()`, `state()` |
| Server | FastAPI |

The agent receives one final-decision task per episode. Each task includes one or more synthetic reports plus a hardcoded drug interaction database. The environment never exposes ground truth to the agent.

## Action Space

| Field | Type | Allowed values | Purpose |
|---|---|---|---|
| `classification` | `str` | `new_signal`, `known_side_effect`, `noise`, `duplicate` | Overall pharmacovigilance judgment |
| `suspect_drug` | `str` | Free text | Drug or interaction the agent believes is causal |
| `severity_assessment` | `str` | `mild`, `moderate`, `severe`, `critical` | Clinical severity assessment |
| `recommended_action` | `str` | `escalate`, `log_and_monitor`, `dismiss`, `request_more_info` | Operational follow-up |
| `reasoning` | `str` | Free text | Short explanation used for grading bonus on hard task |

## Observation Space

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Current task identifier |
| `reports` | `List[AdverseEventReport]` | Synthetic adverse event reports for the task |
| `drug_interaction_db` | `dict` | Hardcoded safety and interaction hints |
| `step_number` | `int` | Current step index |
| `max_steps` | `int` | Maximum number of steps in the episode |
| `feedback` | `Optional[str]` | Feedback message after the previous action |

Each `AdverseEventReport` contains:

| Field | Description |
|---|---|
| `report_id` | Unique synthetic report identifier |
| `patient_age` | Patient age |
| `patient_sex` | Patient sex |
| `drugs` | All drugs the patient was taking |
| `suspect_drug` | Drug named by the original reporter |
| `reaction` | Observed adverse reaction |
| `onset_days` | Days after drug start when reaction began |
| `severity` | Reported severity |
| `outcome` | Recovery status |
| `similar_reports_last_30d` | Count of similar recent reports |

## Tasks

| Task | Difficulty | Scenario | Ground-truth goal | Expected baseline |
|---|---|---|---|---|
| `known_signal_easy` | Easy | Patient on `Lisinopril` develops persistent dry cough with many similar recent reports already known in-label | Recognize a known side effect and recommend `log_and_monitor` | Around `0.85` |
| `cluster_signal_medium` | Medium | Four recent `Cardiovexa` cases show symptomatic bradycardia and near-syncope despite no labeled rhythm toxicity | Recognize a plausible emerging signal and `escalate` | Around `0.65` |
| `confounded_hard` | Hard | Transplant patient with acute kidney injury is blamed on `Trimethoprim-sulfamethoxazole`, but the deeper issue is a `Voriconazole`-`Tacrolimus` interaction | Detect the interaction, classify as `new_signal`, and `escalate` | Around `0.40` |

The hard task is intentionally more difficult because the named suspect drug is not the true cause. The agent must reason over interaction evidence and therapeutic drug-monitoring clues in the provided hardcoded drug database.

## Reward Function

The environment uses deterministic programmatic graders.

| Reward component | Value |
|---|---|
| Correct `classification` | `+0.25` |
| Correct `suspect_drug` | `+0.25` |
| Correct `severity_assessment` | `+0.25` |
| Correct `recommended_action` | `+0.25` |
| False alarm penalty: agent says `new_signal` when truth is `noise` | `-0.10` |
| Missed signal penalty: agent says `noise` when truth is `new_signal` | `-0.20` |
| Hard-task reasoning bonus if explanation mentions `drug interaction`, `tacrolimus`, `voriconazole`, `azole`, `calcineurin`, or `level monitoring` | `+0.15` |

Notes:
- Final reward is clamped to `[0.0, 1.0]`.
- `suspect_drug` matching is forgiving for the hard task and allows substring matches.
- The environment is deterministic and reproducible because all tasks and grading logic are hardcoded.

## Project Structure

| Path | Purpose |
|---|---|
| `env.py` | Main environment class and Pydantic models |
| `tasks.py` | Task definitions and grader functions |
| `data.py` | Synthetic reports and drug interaction database |
| `server.py` | Root FastAPI entrypoint |
| `server/app.py` | OpenEnv-compatible app entrypoint |
| `inference.py` | Baseline inference runner |
| `openenv.yaml` | OpenEnv metadata |
| `Dockerfile` | Multi-stage OpenEnv-style container build |
| `tests/test_env.py` | Local tests |
| `validate-submission.sh` | Pre-submission validation helper |

## Running Locally

### Option 1: Local virtual environment

If you already created the local virtual environment in this repo:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies if needed:

```bash
pip install -r requirements.txt
```

Start the server:

```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```

### Option 2: Docker

Build the image:

```bash
docker build -t pharmacovigilance-env .
```

Run the container:

```bash
docker run -p 7860:7860 pharmacovigilance-env
```

The health endpoint will be available at:

```text
http://localhost:7860/health
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reset` | Starts a task and returns the initial observation |
| `POST` | `/step` | Submits the final agent action and returns observation, reward, done, info |
| `GET` | `/state` | Returns internal environment state summary |
| `GET` | `/tasks` | Lists available task ids |
| `GET` | `/health` | Health check endpoint |

## Baseline Inference Script

The required baseline runner is `inference.py`.

It:
- reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`, and optional `ENV_URL`
- uses the OpenAI client for all model calls
- runs all three tasks sequentially
- emits the required `[START]`, `[STEP]`, and `[END]` lines
- keeps stdout restricted to the judge-expected line types

Required environment variables:

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_your_token_here
export ENV_URL=http://localhost:7860
```

Run:

```bash
python inference.py
```

## Testing And Validation

Run local tests:

```bash
pytest tests/test_env.py -q
```

Run OpenEnv validation:

```bash
openenv validate
```

Run the pre-submission helper:

```bash
chmod +x validate-submission.sh
./validate-submission.sh https://your-space.hf.space
```

That script checks:
1. your Hugging Face Space responds to `POST /reset`
2. the Docker image builds
3. `openenv validate` passes

## Submission Checklist

- `openenv validate` passes
- `docker build` succeeds
- `docker run` starts cleanly
- `POST /reset` returns HTTP `200`
- `inference.py` runs all 3 tasks successfully
- your Hugging Face Space responds to `POST /reset`
- replace the expected baseline values with your measured live baseline values before final submission

## Notes

- No external API calls are made by the environment itself.
- The drug interaction database is hardcoded.
- Ground truth is never exposed in the observation returned to the agent.
- The environment is lightweight enough for a 2 vCPU / 8GB RAM target.
- The expected baseline scores in this README are planning targets until replaced with measured live results.
