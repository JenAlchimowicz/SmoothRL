import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from typing import Tuple
from tqdm import tqdm


class ACnetwork(nn.Module):
    """
    Implements both Actor and Critic in one model (often Actor and Critic share parts of the network)
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        self.actor_head = nn.Linear(64, output_dim)
        self.critic_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action_probs = F.softmax(self.actor_head(x), dim=1)
        state_values = self.critic_head(x)
        
        return action_probs, state_values


class A2Cagent:
    def __init__(
        self,
        obs_space_dim: int,
        action_space_dim: int,
        learning_rate: float,
        gamma: float
    ):
        self.ACnetwork = ACnetwork(obs_space_dim, action_space_dim)
        self.optimizer = torch.optim.Adam(self.ACnetwork.parameters(), lr = learning_rate)
        self.gamma = gamma


    def choose_action(self, state: np.array) -> Tuple[int, float]:
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, _ = self.ACnetwork(state)
        action_distribution = Categorical(probs)
        action = action_distribution.sample()
        return action.item(), action_distribution.log_prob(action)


    def learn(
        self,
        state: np.array,
        action_logprob: float,
        reward: float,
        next_state: np.array,
        terminal: bool
        ):
            self.optimizer.zero_grad()

            _, state_value = self.ACnetwork(torch.from_numpy(state).float().unsqueeze(0))
            _, next_state_value = self.ACnetwork(torch.from_numpy(next_state).float().unsqueeze(0))

            # Multiplying by (1-terminal) ensures terminal states have value 0
            advantage = reward + self.gamma * next_state_value * (1-terminal) - state_value

            actor_loss = -action_logprob * advantage
            critic_loss = torch.pow(advantage, 2)
            total_loss = actor_loss + critic_loss

            total_loss.backward()
            self.optimizer.step()


# Setup
gamma = 0.99
learning_rate = 0.0005
train_episodes = 1_000
eval_episodes = 200

env = gym.make('CartPole-v1')
observation_space_dim = np.product(env.observation_space.shape)
action_space_dim = env.action_space.n
agent = A2Cagent(observation_space_dim, action_space_dim, learning_rate, gamma)


# Training
for _ in tqdm(range(train_episodes)):
    state, _ = env.reset()

    while True:
        action, action_logprob = agent.choose_action(state)
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
        next_state, reward, terminal, truncated, info = env.step(action)
        state = next_state
        episode_rewards += reward

        if terminal or truncated:
            rewards.append(episode_rewards)
            break

print(f"Average reward obtained: {np.mean(rewards)}")
