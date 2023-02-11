import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from typing import Tuple, List
from tqdm import tqdm


class ACnetwork(nn.Module):
    """
    Implements both Actor and Critic in one model (often Actor and Critic share parts of the network)
    """

    def __init__(self, input_dim: int, n_actions: int, action_bounds: List[float]):
        super().__init__()
        self.bounds = action_bounds

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        self.critic = nn.Linear(64, 1)
        self.actor_mean = nn.Linear(64, n_actions)
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_actions))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        state_values = self.critic(x)

        action_means = self.actor_mean(x)
        action_stds = torch.exp(self.actor_logstd)
        action_distributions = Normal(action_means, action_stds)
        actions = action_distributions.sample()

        return (
            torch.clamp(actions, self.bounds[0], self.bounds[1]).item(),
            action_distributions.log_prob(actions),
            state_values,
        )


class A2Cagent:
    def __init__(
        self,
        obs_space_dim: int,
        action_space_dim: int,
        action_bounds: List[float],
        learning_rate: float,
        gamma: float,
    ):
        self.ACnetwork = ACnetwork(obs_space_dim, action_space_dim, action_bounds)
        self.optimizer = torch.optim.Adam(self.ACnetwork.parameters(), lr=learning_rate)
        self.gamma = gamma

    def choose_action(self, state: np.array) -> Tuple[int, float]:
        state = torch.from_numpy(state).float().unsqueeze(0)
        action, action_logprob, _ = self.ACnetwork(state)
        return action, action_logprob

    def learn(
        self,
        state: np.array,
        action_logprob: float,
        reward: float,
        next_state: np.array,
        terminal: bool,
    ):
        self.optimizer.zero_grad()

        _, _, state_value = self.ACnetwork(torch.from_numpy(state).float().unsqueeze(0))
        _, _, next_state_value = self.ACnetwork(torch.from_numpy(next_state).float().unsqueeze(0))

        # Multiplying by (1-terminal) ensures terminal states have value 0
        advantage = (reward + self.gamma * next_state_value * (1 - terminal) - state_value)

        actor_loss = -action_logprob * advantage
        critic_loss = torch.pow(advantage, 2)
        total_loss = actor_loss + critic_loss

        total_loss.backward()
        self.optimizer.step()


# Setup
gamma = 0.99
learning_rate = 0.0005
train_episodes = 100
eval_episodes = 200

env = gym.make('MountainCarContinuous-v0')
observation_space_dim = np.product(env.observation_space.shape)
n_actions = np.product(env.action_space.shape)
max_action_value = env.action_space.high.item()
min_action_value = env.action_space.low.item()
agent = A2Cagent(observation_space_dim, n_actions, [min_action_value, max_action_value], learning_rate, gamma)


# Training
for _ in tqdm(range(train_episodes)):
    state, _ = env.reset()

    while True:
        action, action_logprob = agent.choose_action(state)
        action = np.array(action).reshape(n_actions,)
        next_state, reward, terminal, truncated, info = env.step(action)

        agent.learn(state, action_logprob, reward, next_state, terminal or truncated)

        state = next_state

        if terminal or truncated:
            break


# Evaluation
rewards = []
for _ in tqdm(range(eval_episodes)):
    state, _ = env.reset()
    episode_rewards = 0
    while True:
        action, _ = agent.choose_action(state)
        action = np.array(action).reshape(n_actions,)
        next_state, reward, terminal, truncated, info = env.step(action)
        state = next_state
        episode_rewards += reward

        if terminal or truncated:
            rewards.append(episode_rewards)
            break

print(f"Average reward obtained: {np.mean(rewards)}")
