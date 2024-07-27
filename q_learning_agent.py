from collections import defaultdict
import numpy as np

def default_q_table_value(action_space):
    return np.zeros(action_space.n)

class QLearningAgent:
    def __init__(self, action_space, observation_space, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.action_space = action_space
        self.observation_space = observation_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = defaultdict(self._default_q_table_value)

    def _default_q_table_value(self):
        return default_q_table_value(self.action_space)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + (0 if done else self.gamma * self.q_table[next_state][best_next_action])
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

def train_q_learning(env, q_agent, num_episodes=1000):
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = q_agent.select_action(tuple(obs))
            next_obs, reward, done, _, _ = env.step(action)
            q_agent.update_q_table(tuple(obs), action, reward, tuple(next_obs), done)
            obs = next_obs
        q_agent.decay_epsilon()