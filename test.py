import pytest
from env import RobustEnv  # 确保你的环境模块路径正确
from q_learning_agent import QLearningAgent, train_q_learning  # 确保你的代理模块路径正确
from stable_baselines3 import PPO
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

@pytest.fixture
def env():
    env_instance = RobustEnv()
    env_instance.obstacle_pos = [(1, 1), (2, 2)]  # 设置固定障碍物位置
    return env_instance

@pytest.fixture
def q_agent(env):
    # 创建 Q-learning 代理
    return QLearningAgent(env.action_space, env.observation_space)

@pytest.fixture
def ppo_agent(env):
    # 加载 PPO 代理模型
    model_path = "E:/topic/game_train/ppo_agent.zip"
    assert os.path.exists(model_path), f"模型文件 {model_path} 不存在"
    return PPO.load(model_path)

def test_agent_q_learning(env, q_agent):
    # 训练 Q-learning 代理
    train_q_learning(env, q_agent, num_episodes=100)
    # 保存模型
    model_path = "E:/topic/game_train/train_q_learning_agent_10.pkl"
    joblib.dump(q_agent, model_path)
    # 检查模型文件是否存在
    assert os.path.exists(model_path), f"模型文件 {model_path} 不存在"
    # 加载已训练的 Q-learning 代理
    q_agent_loaded = joblib.load(model_path)
    # 测试 Q-learning 代理
    steps_q_learning, path_q_learning = test_agent_inner(env, q_agent_loaded)
    print(f"Q-learning 从起点到终点的步数: {steps_q_learning}")
    print("Q-learning 路径:", path_q_learning)  # 打印路径数据
    plot_path(path_q_learning, env.obstacle_pos, "Q-learning Path")

def test_agent_ppo(env, ppo_agent):
    # 测试 PPO 代理
    steps_ppo, path_ppo = test_agent_inner(env, ppo_agent, use_ppo=True)
    print(f"PPO 从起点到终点的步数: {steps_ppo}")
    print("PPO 路径:", path_ppo)  # 打印路径数据
    plot_path(path_ppo, env.obstacle_pos, "PPO Path")

def test_agent_inner(env, agent, use_ppo=False):
    obs, _ = env.reset()
    path = [obs.copy()]
    done = False
    steps = 0

    while not done:
        if use_ppo:
            action, _ = agent.predict(obs)
        else:
            action = agent.select_action(tuple(obs))
        obs, reward, done, _, _ = env.step(action)
        path.append(obs.copy())
        steps += 1

    return steps, path

def plot_path(path, obstacles, title):
    print("绘图路径:", path)  # 打印路径数据
    path = np.array(path)
    plt.figure(figsize=(8, 8))
    plt.plot(path[:, 0], path[:, 1], marker='o', linestyle='-', color='r')  # 路径线段为红色
    plt.scatter([0, 8], [0, 8], color='r', marker='X', label='Start/Goal')  # 起点和终点
    if obstacles:
        obstacles = np.array(list(obstacles))
        plt.scatter(obstacles[:, 0], obstacles[:, 1], color='k', marker='x', label='Obstacles')  # 障碍物
    plt.xlim(-1, 9)  # 修改x轴范围
    plt.ylim(-1, 9)  # 修改y轴范围
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()


def test_inner_with_q_agent(env, q_agent):
    steps, path = test_agent_inner(env, q_agent)
    print(f"Inner test with Q-learning agent steps: {steps}")
    print("Inner Q-learning 路径:", path)  # 打印路径数据
    assert steps > 0
    assert len(path) > 0

def test_inner_with_ppo_agent(env, ppo_agent):
    steps, path = test_agent_inner(env, ppo_agent, use_ppo=True)
    print(f"Inner test with PPO agent steps: {steps}")
    print("Inner PPO 路径:", path)  # 打印路径数据
    assert steps > 0
    assert len(path) > 0