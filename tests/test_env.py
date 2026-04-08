import sys
from pathlib import Path

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import Action, PharmaVigilanceEnv
from tasks import (
    cluster_signal_medium_action_grader,
    cluster_signal_medium_grader,
    confounded_hard_action_grader,
    confounded_hard_grader,
    get_task,
    get_tasks,
    known_signal_easy_action_grader,
    known_signal_easy_grader,
)


def test_reset_loads_easy_task():
    env = PharmaVigilanceEnv()
    obs = env.reset("known_signal_easy")
    assert obs.task_id == "known_signal_easy"
    assert obs.step_number == 0
    assert obs.max_steps == 2
    assert len(obs.reports) == 1


def test_known_signal_grader_full_credit():
    reward = known_signal_easy_action_grader(
        Action(
            classification="known_side_effect",
            suspect_drug="Lisinopril",
            severity_assessment="mild",
            recommended_action="log_and_monitor",
            reasoning="Known reaction pattern.",
            confidence=91,
        )
    )
    assert reward.total == 1.0


def test_medium_cluster_grader_partial_credit():
    reward = cluster_signal_medium_action_grader(
        Action(
            classification="new_signal",
            suspect_drug="Cardiovexa",
            severity_assessment="moderate",
            recommended_action="escalate",
            reasoning="A cluster is forming.",
        )
    )
    assert reward.total == 0.75


def test_hard_grader_reasoning_bonus():
    reward = confounded_hard_action_grader(
        Action(
            classification="new_signal",
            suspect_drug="Tacrolimus+Voriconazole",
            severity_assessment="critical",
            recommended_action="escalate",
            reasoning="This looks like a tacrolimus-voriconazole drug interaction with toxic levels.",
        )
    )
    assert reward.total == 1.0
    assert reward.breakdown["reasoning_bonus"] == 0.05


def test_hard_grader_substring_suspect_match():
    reward = confounded_hard_action_grader(
        Action(
            classification="new_signal",
            suspect_drug="Tacrolimus",
            severity_assessment="critical",
            recommended_action="escalate",
            reasoning="Voriconazole likely raised tacrolimus exposure.",
        )
    )
    assert reward.breakdown["suspect_drug"] == 0.25


def test_env_step_returns_done():
    env = PharmaVigilanceEnv()
    env.reset("confounded_hard")
    obs, reward, done, info = env.step(
        Action(
            classification="new_signal",
            suspect_drug="Tacrolimus+Voriconazole",
            severity_assessment="critical",
            recommended_action="escalate",
            reasoning="Tacrolimus toxicity from an azole interaction.",
        )
    )
    assert done is False
    assert obs.step_number == 1
    assert "reward_breakdown" in info
    assert reward.total >= 0.20

    obs, reward, done, info = env.step(
        Action(
            classification="new_signal",
            suspect_drug="Tacrolimus+Voriconazole",
            severity_assessment="critical",
            recommended_action="escalate",
            reasoning="Tacrolimus toxicity from an azole interaction.",
        )
    )
    assert done is True
    assert obs.step_number == 2
    assert reward.total >= 0.85


def test_first_step_returns_partial_reward_and_review_feedback():
    env = PharmaVigilanceEnv()
    obs = env.reset("cluster_signal_medium")

    obs, reward, done, info = env.step(
        Action(
            classification="new_signal",
            suspect_drug="Cardiovexa",
            severity_assessment="severe",
            recommended_action="escalate",
            reasoning="Clustered bradycardia on a newer therapy.",
            confidence=88,
        )
    )
    assert done is False
    assert obs.step_number == 1
    assert reward.total > 0.0
    assert info["phase"] == "initial_triage"
    assert "Senior review note" in obs.feedback


def test_final_step_awards_revision_bonus_when_agent_improves():
    env = PharmaVigilanceEnv()
    env.reset("cluster_signal_medium")

    env.step(
        Action(
            classification="noise",
            suspect_drug="Unknown",
            severity_assessment="mild",
            recommended_action="dismiss",
            reasoning="Weak initial guess.",
            confidence=90,
        )
    )
    _, reward, done, info = env.step(
        Action(
            classification="new_signal",
            suspect_drug="Cardiovexa",
            severity_assessment="severe",
            recommended_action="escalate",
            reasoning="Follow-up reports confirm a coherent bradycardia cluster.",
            confidence=82,
        )
    )
    assert done is True
    assert reward.breakdown["revision_bonus"] == 0.05
    assert info["phase"] == "final_review"


def test_final_step_applies_stubborn_penalty_for_repeating_weak_answer():
    env = PharmaVigilanceEnv()
    env.reset("confounded_hard")

    weak = Action(
        classification="noise",
        suspect_drug="Trimethoprim-sulfamethoxazole",
        severity_assessment="mild",
        recommended_action="dismiss",
        reasoning="Reporter probably overcalled it.",
        confidence=85,
    )
    env.step(weak)
    _, reward, done, _ = env.step(weak)
    assert done is True
    assert reward.breakdown["stubborn_penalty"] == -0.05


def test_initial_step_can_return_negative_reward_for_unsafe_triage():
    env = PharmaVigilanceEnv()
    env.reset("cluster_signal_medium")

    _, reward, done, info = env.step(
        Action(
            classification="noise",
            suspect_drug="Unknown",
            severity_assessment="mild",
            recommended_action="dismiss",
            reasoning="No obvious concern.",
            confidence=95,
        )
    )
    assert done is False
    assert info["phase"] == "initial_triage"
    assert reward.total < 0.0


def test_single_step_action_grader_can_return_negative_total():
    reward = cluster_signal_medium_action_grader(
        Action(
            classification="noise",
            suspect_drug="Unknown",
            severity_assessment="mild",
            recommended_action="dismiss",
            reasoning="Probably unrelated.",
            confidence=95,
        )
    )
    assert reward.total < 0.0


def test_overconfidence_penalty_applies_on_weak_single_step_grading():
    reward = cluster_signal_medium_action_grader(
        Action(
            classification="noise",
            suspect_drug="Unknown",
            severity_assessment="mild",
            recommended_action="dismiss",
            reasoning="This is probably nothing.",
            confidence=95,
        )
    )
    assert reward.breakdown["confidence_adjustment"] == -0.10


def test_low_confidence_penalty_applies_on_strong_answer():
    reward = known_signal_easy_action_grader(
        Action(
            classification="known_side_effect",
            suspect_drug="Lisinopril",
            severity_assessment="mild",
            recommended_action="log_and_monitor",
            reasoning="Known labeled ACE-inhibitor cough.",
            confidence=20,
        )
    )
    assert reward.breakdown["confidence_adjustment"] == -0.03


def test_episode_rejects_third_step_after_completion():
    env = PharmaVigilanceEnv()
    env.reset("known_signal_easy")
    good = Action(
        classification="known_side_effect",
        suspect_drug="Lisinopril",
        severity_assessment="mild",
        recommended_action="log_and_monitor",
        reasoning="Known ACE-inhibitor cough.",
        confidence=90,
    )
    env.step(good)
    env.step(good)
    with pytest.raises(RuntimeError, match="Episode already complete"):
        env.step(good)


def test_state_tracks_last_action():
    env = PharmaVigilanceEnv()
    env.reset("known_signal_easy")
    env.step(
        Action(
            classification="known_side_effect",
            suspect_drug="Lisinopril",
            severity_assessment="mild",
            recommended_action="log_and_monitor",
            reasoning="Known adverse effect.",
            confidence=90,
        )
    )
    env.step(
        Action(
            classification="known_side_effect",
            suspect_drug="Lisinopril",
            severity_assessment="mild",
            recommended_action="log_and_monitor",
            reasoning="Known adverse effect.",
            confidence=90,
        )
    )
    state = env.state()
    assert state["step_number"] == 2
    assert state["last_action"]["classification"] == "known_side_effect"


def test_all_tasks_available():
    tasks = get_tasks()
    assert set(tasks.keys()) == {
        "known_signal_easy",
        "cluster_signal_medium",
        "confounded_hard",
    }


def test_grouped_tasks_expose_easy_medium_hard_pools():
    grouped = get_tasks(grouped=True)
    assert set(grouped.keys()) == {"easy", "medium", "hard"}
    assert grouped["easy"][0].task_id == "known_signal_easy"
    assert grouped["medium"][0].task_id == "cluster_signal_medium"
    assert grouped["hard"][0].task_id == "confounded_hard"


def test_get_task_returns_hard_truth():
    task = get_task("confounded_hard")
    assert task.ground_truth.suspect_drug == "Tacrolimus+Voriconazole"


def test_public_graders_are_strictly_bounded():
    assert known_signal_easy_grader({"rewards": [1.0]}) == 0.99
    assert cluster_signal_medium_grader({"rewards": [0.0]}) == 0.01
    assert confounded_hard_grader({"score": 1.5}) == 0.99


def test_inference_final_score_uses_public_task_grader():
    pytest.importorskip("openenv")
    from inference import final_score

    rewards = [0.4, 1.0]
    assert final_score("known_signal_easy", rewards) == known_signal_easy_grader({"rewards": rewards})
    assert final_score("cluster_signal_medium", rewards) == cluster_signal_medium_grader({"rewards": rewards})
    assert final_score("confounded_hard", rewards) == confounded_hard_grader({"rewards": rewards})


def test_http_reset_then_step_roundtrip():
    pytest.importorskip("openenv")
    from fastapi.testclient import TestClient
    from server.app import app

    client = TestClient(app)

    reset_response = client.post("/reset", json={})
    assert reset_response.status_code == 200

    first_step = client.post(
        "/step",
        json={
            "action": {
                "classification": "known_side_effect",
                "suspect_drug": "Lisinopril",
                "severity_assessment": "mild",
                "recommended_action": "log_and_monitor",
                "reasoning": "Known ACE inhibitor cough.",
                "confidence": 90,
            }
        },
    )
    assert first_step.status_code == 200
    first_payload = first_step.json()
    assert first_payload["done"] is False

    step_response = client.post(
        "/step",
        json={
            "action": {
                "classification": "known_side_effect",
                "suspect_drug": "Lisinopril",
                "severity_assessment": "mild",
                "recommended_action": "log_and_monitor",
                "reasoning": "Known ACE inhibitor cough.",
                "confidence": 90,
            }
        },
    )
    assert step_response.status_code == 200
    payload = step_response.json()
    assert payload["done"] is True
    assert payload["reward"] == 1.0
