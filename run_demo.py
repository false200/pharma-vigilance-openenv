from agent import RuleBasedPharmaAgent
from env import PharmaVigilanceEnv


def main() -> None:
    env = PharmaVigilanceEnv()
    agent = RuleBasedPharmaAgent()

    for task_id in ("known_signal_easy", "cluster_signal_medium", "confounded_hard"):
        observation = env.reset(task_id)
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)

        print(f"\nTask: {task_id}")
        print(f"Action: {action.classification} / {action.suspect_drug}")
        print(f"Reward: {reward.total:.2f}")
        print(f"Done: {done}")
        print(f"Feedback: {observation.feedback}")
        print(f"Info: {info}")


if __name__ == "__main__":
    main()
