from env import RobustEnv
from q_learning_agent import QLearningAgent, train_q_learning
from ppo_agent import train_ppo, load_ppo_model
import joblib

def create_random_env():
    return RobustEnv()

def train_multiple_environments(train_func, agent_class, num_envs=10, num_episodes_per_env=100, is_ppo=False):
    for i in range(num_envs):
        env = create_random_env()
        agent = agent_class(env.action_space, env.observation_space) if not is_ppo else None
        print(f"Training on environment {i + 1}/{num_envs}")
        if is_ppo:
            train_func(env, agent)
        else:
            train_func(env, agent, num_episodes=num_episodes_per_env)
        env.close()
        if not is_ppo:
            joblib.dump(agent, f'{train_func.__name__}_agent_{i+1}.pkl')

if __name__ == "__main__":
    # 训练 Q-learning 模型
    print("训练 Q-learning 模型...")
    train_multiple_environments(train_q_learning, QLearningAgent)

    # 训练 PPO 模型
    print("训练 PPO 模型...")
    train_multiple_environments(train_ppo, lambda env: None, is_ppo=True)