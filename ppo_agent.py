from stable_baselines3 import PPO

def train_ppo(env, agent, total_timesteps=10000):
    ppo_model = PPO('MlpPolicy', env, verbose=1)
    ppo_model.learn(total_timesteps=total_timesteps)
    ppo_model.save("ppo_agent")

def load_ppo_model():
    return PPO.load("ppo_agent")