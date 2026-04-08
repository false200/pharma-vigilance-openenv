from fastapi import FastAPI
from openenv.core.env_server import create_web_interface_app
from openenv.core.env_server.types import State

try:
    from ..env import PharmaVigilanceEnv
    from ..models import PharmaAction, PharmaObservation
except ImportError:
    from env import PharmaVigilanceEnv
    from models import PharmaAction, PharmaObservation


TASK_IDS = ["known_signal_easy", "cluster_signal_medium", "confounded_hard"]


class OpenEnvPharmaAdapter:
    """
    Thin adapter that exposes the local environment through the interface
    expected by OpenEnv's HTTP server and web playground helpers.
    """

    _shared_env = PharmaVigilanceEnv()
    _shared_state = State(episode_id=None, step_count=0)

    def __init__(self) -> None:
        self._env = self.__class__._shared_env
        self._last_state = self.__class__._shared_state

    @staticmethod
    def _normalize_reports(reports):
        normalized = []
        for report in reports:
            if hasattr(report, "model_dump"):
                normalized.append(report.model_dump())
            else:
                normalized.append(report)
        return normalized

    def reset(self, task_id: str = "known_signal_easy") -> PharmaObservation:
        observation = self._env.reset(task_id=task_id)
        self._last_state = State(episode_id=task_id, step_count=0)
        self.__class__._shared_state = self._last_state
        return PharmaObservation(
            task_id=observation.task_id,
            reports=self._normalize_reports(observation.reports),
            drug_interaction_db=observation.drug_interaction_db,
            step_number=observation.step_number,
            max_steps=observation.max_steps,
            feedback=observation.feedback,
            reward=0.0,
            done=False,
            metadata={"difficulty": self._env.current_task.difficulty if self._env.current_task else None},
        )

    async def reset_async(self, task_id: str = "known_signal_easy") -> PharmaObservation:
        return self.reset(task_id=task_id)

    def step(self, action: PharmaAction) -> PharmaObservation:
        observation, reward, done, info = self._env.step(action)
        self._last_state = State(
            episode_id=observation.task_id,
            step_count=observation.step_number,
        )
        self.__class__._shared_state = self._last_state
        return PharmaObservation(
            task_id=observation.task_id,
            reports=self._normalize_reports(observation.reports),
            drug_interaction_db=observation.drug_interaction_db,
            step_number=observation.step_number,
            max_steps=observation.max_steps,
            feedback=observation.feedback,
            reward=reward.total,
            done=done,
            metadata=info,
        )

    async def step_async(self, action: PharmaAction) -> PharmaObservation:
        return self.step(action)

    @property
    def state(self) -> State:
        return self._last_state

    def close(self) -> None:
        return None


app: FastAPI = create_web_interface_app(
    OpenEnvPharmaAdapter,
    PharmaAction,
    PharmaObservation,
    env_name="pharma_vigilance_env",
)


@app.get("/tasks")
def list_tasks():
    return {"tasks": TASK_IDS}


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
