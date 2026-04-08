from fastapi import FastAPI

from env import Action, PharmaVigilanceEnv


app = FastAPI()
env = PharmaVigilanceEnv()


@app.post("/reset")
def reset(body: dict = {}):
    task_id = body.get("task_id", "known_signal_easy")
    obs = env.reset(task_id)
    return obs.model_dump()


@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return env.state()


@app.get("/tasks")
def list_tasks():
    return {"tasks": ["known_signal_easy", "cluster_signal_medium", "confounded_hard"]}


@app.get("/health")
def health():
    return {"status": "ok"}


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
