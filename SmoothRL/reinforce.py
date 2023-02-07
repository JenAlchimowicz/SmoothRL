import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from tqdm import tqdm
from typing import List


class Policy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


class ReinforceAgent:
    def __init__(
        self,
        obs_space_dim: int,
        action_space_dim: int,
        gamma: float,
        learning_rate: float,
    ):

        self.policy = Policy(obs_space_dim, action_space_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        

    def act(self, state: np.array):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy.forward(state)
        action_distribution = Categorical(probs)
        action = action_distribution.sample()
        return action.item(), action_distribution.log_prob(action)


    def learn(
        self,
        log_probs: List[torch.Tensor],
        rewards: List[float]
    ):
        self.optimizer.zero_grad()
        episode_length = len(rewards)
        discounted_returns = np.zeros(episode_length)

        # Calculate discounted future returns for each timestep
        for t in reversed(range(episode_length)[:-1]):
            discounted_returns[t] += rewards[t] + self.gamma * discounted_returns[t+1]
        
        policy_loss = []
        for log_prob, discounted_return in zip(log_probs, discounted_returns):
            policy_loss.append(-log_prob * discounted_return)
        policy_loss = torch.cat(policy_loss).sum()

        policy_loss.backward()
        self.optimizer.step()


# Setup
gamma = 0.99
learning_rate = 0.0002
train_episodes = 500
eval_episodes = 200

env = gym.make('Acrobot-v1')
agent = ReinforceAgent(env.observation_space.shape[0], env.action_space.n, gamma, learning_rate)


# Training
for _ in tqdm(range(train_episodes)):
    log_probs = []
    rewards = []
    state, _ = env.reset()

    # Generate trajectory
    while True:
        action, log_prob = agent.act(state)
        next_state, reward, terminal, truncated, info = env.step(action)

        log_probs.append(log_prob)
        rewards.append(reward)

        state = next_state

        if terminal or truncated:
            break

    # Learn on trajectory
    agent.learn(log_probs, rewards)


# Evaluation
rewards = []
for _ in tqdm(range(eval_episodes)):
    state, _ = env.reset()
    episode_rewards = 0
    while True:
        action, _ = agent.act(state)
        next_state, reward, terminal, truncated, info = env.step(action)
        state = next_state
        episode_rewards += reward

        if terminal or truncated:
            rewards.append(episode_rewards)
            break

print(f"Average reward obtained: {np.mean(rewards)}")
