import gymnasium as gym
import numpy as np
from typing import Optional, Hashable
from tqdm import tqdm


class QlearningAgent:
    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        epsilon: float,
        gamma: float,
        step_size: float
    ):
        self.obs_space = obs_space
        self.action_space = action_space
        self.epsilon = epsilon
        self.gamma = gamma
        self.step_size = step_size

        self.q_table = np.zeros((self.obs_space.n, self.action_space.n))
        self.state_dict = {}


    # Maps states to rows in q_table
    def get_state_idx(self, state: Hashable) -> int:
        if state not in self.state_dict.keys():
            self.state_dict[state] = len(self.state_dict)
        state_idx = self.state_dict[state]
        return state_idx


    def choose_action(
        self,
        state: Hashable,
        epsilon: Optional[float] = None
    ) -> int:
        
        state_idx = self.get_state_idx(state)
        epsilon_ = epsilon if epsilon is not None else self.epsilon
        
        if np.random.rand() < epsilon_:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_table[state_idx, :])

    
    def update_q_table(
        self,
        state: Hashable,
        action: int,
        reward: float,
        next_state: Hashable
    ):
        state_idx = self.get_state_idx(state)
        next_state_idx = self.get_state_idx(next_state)
        self.q_table[state_idx, action] = self.q_table[state_idx, action] + self.step_size * (reward + self.gamma * np.max(self.q_table[next_state_idx]) - self.q_table[state_idx, action])


# Setup
env = gym.make('Taxi-v3')
epsilon = 0.15
gamma = 0.99
step_size = 0.2
train_episodes = 10_000
eval_episodes = 1_000

agent = QlearningAgent(env.observation_space, env.action_space, epsilon, gamma, step_size)

# Training
for _ in tqdm(range(train_episodes)):
    state, _ = env.reset()

    while True:
        action = agent.choose_action(state)
        next_state, reward, terminal, truncated, info = env.step(action)
        agent.update_q_table(state, action, reward, next_state)
        state = next_state

        if terminal or truncated:
            break

# Evaluation
rewards = []
for _ in tqdm(range(eval_episodes)):
    state, _ = env.reset()
    episode_rewards = 0
    while True:
        action = agent.choose_action(state, epsilon = 0)
        next_state, reward, terminal, truncated, info = env.step(action)
        state = next_state
        episode_rewards += reward

        if terminal or truncated:
            rewards.append(episode_rewards)
            break

print(f"Average reward obtained: {np.mean(rewards)}")
