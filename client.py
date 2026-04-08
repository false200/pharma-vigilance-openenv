# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pharmacovigilance Signal Detector Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .env import Action, Observation, AdverseEventReport
except ImportError:
    from env import Action, Observation, AdverseEventReport


class PharmaVigilanceEnvClient(
    EnvClient[Action, Observation, State]
):
    """
    Client for the Pharmacovigilance Signal Detector environment.

    This client maintains a persistent connection to the environment server and
    parses server responses into strongly-typed observation models.

    Example:
        >>> with PharmaVigilanceEnvClient(base_url="http://localhost:7860") as env:
        ...     result = env.reset(task_id="known_signal_easy")
        ...     print(result.observation.task_id)
        ...
        ...     action = Action(
        ...         classification="known_side_effect",
        ...         suspect_drug="Ibuprofen",
        ...         severity_assessment="moderate",
        ...         recommended_action="log_and_monitor",
        ...         reasoning="GI bleeding is a known ibuprofen adverse effect.",
        ...     )
        ...     result = env.step(action)
        ...     print(result.observation.feedback)
        ...     print(result.reward)

    Example with Docker:
        >>> client = PharmaVigilanceEnvClient.from_docker_image("pharmacovigilance-env:latest")
        >>> try:
        ...     result = client.reset(task_id="cluster_signal_medium")
        ...     action = Action(
        ...         classification="new_signal",
        ...         suspect_drug="Gliptozin",
        ...         severity_assessment="severe",
        ...         recommended_action="escalate",
        ...         reasoning="Clustered vision loss on a new drug warrants escalation.",
        ...     )
        ...     result = client.step(action)
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: Action) -> Dict:
        """
        Convert an Action model into the JSON payload sent to /step.

        Args:
            action: Typed agent action.

        Returns:
            Dictionary representation suitable for JSON transport.
        """
        return {
            "classification": action.classification,
            "suspect_drug": action.suspect_drug,
            "severity_assessment": action.severity_assessment,
            "recommended_action": action.recommended_action,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict) -> StepResult[Observation]:
        """
        Parse a server /step response into StepResult[Observation].

        Args:
            payload: JSON response from the environment server.

        Returns:
            StepResult containing the typed observation, reward, and done flag.
        """
        obs_data = payload.get("observation", {})
        reports = [
            AdverseEventReport(**report)
            for report in obs_data.get("reports", [])
        ]

        observation = Observation(
            task_id=obs_data.get("task_id", ""),
            reports=reports,
            drug_interaction_db=obs_data.get("drug_interaction_db", {}),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 1),
            feedback=obs_data.get("feedback"),
        )

        reward_payload = payload.get("reward", 0.0)
        reward_total = (
            reward_payload.get("total", 0.0)
            if isinstance(reward_payload, dict)
            else reward_payload
        )

        return StepResult(
            observation=observation,
            reward=reward_total,
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse the /state response into an OpenEnv State object.

        Args:
            payload: JSON response from the state endpoint.

        Returns:
            State with a task-derived episode identifier and current step count.
        """
        return State(
            episode_id=payload.get("task_id", "pharma-vigilance"),
            step_count=payload.get("step_number", 0),
        )
