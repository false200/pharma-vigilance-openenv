import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import Action, PharmaVigilanceEnv
from tasks import (
    cluster_signal_medium_grader,
    confounded_hard_grader,
    get_task,
    get_tasks,
    known_signal_easy_grader,
)


def test_reset_loads_easy_task():
    env = PharmaVigilanceEnv()
    obs = env.reset("known_signal_easy")
    assert obs.task_id == "known_signal_easy"
    assert obs.step_number == 0
    assert len(obs.reports) == 1


def test_known_signal_grader_full_credit():
    reward = known_signal_easy_grader(
        Action(
            classification="known_side_effect",
            suspect_drug="Ibuprofen",
            severity_assessment="moderate",
            recommended_action="log_and_monitor",
            reasoning="Known reaction pattern.",
        )
    )
    assert reward.total == 1.0


def test_medium_cluster_grader_partial_credit():
    reward = cluster_signal_medium_grader(
        Action(
            classification="new_signal",
            suspect_drug="Gliptozin",
            severity_assessment="moderate",
            recommended_action="escalate",
            reasoning="A cluster is forming.",
        )
    )
    assert reward.total == 0.75


def test_hard_grader_reasoning_bonus():
    reward = confounded_hard_grader(
        Action(
            classification="new_signal",
            suspect_drug="Atorvastatin+Clarithromycin",
            severity_assessment="critical",
            recommended_action="escalate",
            reasoning="This looks like a CYP3A4 drug interaction causing statin toxicity.",
        )
    )
    assert reward.total == 1.0
    assert reward.breakdown["reasoning_bonus"] == 0.15


def test_hard_grader_substring_suspect_match():
    reward = confounded_hard_grader(
        Action(
            classification="new_signal",
            suspect_drug="Atorvastatin",
            severity_assessment="critical",
            recommended_action="escalate",
            reasoning="Clarithromycin likely increased statin levels.",
        )
    )
    assert reward.breakdown["suspect_drug"] == 0.25


def test_env_step_returns_done():
    env = PharmaVigilanceEnv()
    env.reset("confounded_hard")
    obs, reward, done, info = env.step(
        Action(
            classification="new_signal",
            suspect_drug="Atorvastatin+Clarithromycin",
            severity_assessment="critical",
            recommended_action="escalate",
            reasoning="Statin toxicity from a drug interaction.",
        )
    )
    assert done is True
    assert obs.step_number == 1
    assert "reward_breakdown" in info
    assert reward.total >= 0.85


def test_state_tracks_last_action():
    env = PharmaVigilanceEnv()
    env.reset("known_signal_easy")
    env.step(
        Action(
            classification="known_side_effect",
            suspect_drug="Ibuprofen",
            severity_assessment="moderate",
            recommended_action="log_and_monitor",
            reasoning="Known adverse effect.",
        )
    )
    state = env.state()
    assert state["step_number"] == 1
    assert state["last_action"]["classification"] == "known_side_effect"


def test_all_tasks_available():
    tasks = get_tasks()
    assert set(tasks.keys()) == {
        "known_signal_easy",
        "cluster_signal_medium",
        "confounded_hard",
    }


def test_get_task_returns_hard_truth():
    task = get_task("confounded_hard")
    assert task.ground_truth.suspect_drug == "Atorvastatin+Clarithromycin"
