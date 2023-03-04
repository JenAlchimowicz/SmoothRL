import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

from typing import Optional, Tuple
from tqdm import tqdm


class DuellingQnetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int
    ):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        features = self.fc(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        qvals = values + (advantages - advantages.mean())
        return qvals
    

class DQNagent:
    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        epsilon: float,
        gamma: float,
        lr: float,
        tau: float
    ):
        self.action_space = action_space
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau = tau
        
        self.Qnetwork = DuellingQnetwork(obs_space.shape[0], action_space.n)
        self.optimizer = torch.optim.Adam(self.Qnetwork.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def choose_action(
        self,
        state: np.array,
        epsilon: Optional[float] = None
    ) -> int:

        epsilon = epsilon if epsilon is not None else self.epsilon
        if np.random.rand() < epsilon:
            action = self.action_space.sample()
        else:
            q_values = self.Qnetwork(torch.from_numpy(state))
            action = torch.argmax(q_values).item()
        return action
    
    def learn(self, batch: Tuple):

        states, actions, rewards, next_states, terminals = batch

        # Get q_values of chosen state-action pairs
        q_states = self.Qnetwork(states).gather(1, actions.unsqueeze(-1)).squeeze()

        # Build TD target
        q_next_states = self.Qnetwork(next_states)
        target = rewards + self.gamma * torch.max(q_next_states, dim=1)[0] * (1 - terminals.float())

        # Get TD error and learn
        loss = self.loss(q_states, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ReplayBuffer:
    def __init__(self, max_size: int, state_dim: int):
        self.max_size = max_size
        self.memory_coutner = 0

        self.states = torch.zeros((max_size, *state_dim), dtype=torch.float)
        self.next_states = torch.zeros((max_size, *state_dim), dtype=torch.float)
        self.actions = torch.zeros(max_size, dtype=torch.int64)
        self.rewards = torch.zeros(max_size, dtype=torch.float)
        self.terminals = torch.zeros(max_size, dtype=torch.bool)

    def store_transition(
        self,
        state: np.array,
        action: int,
        reward: float,
        next_state: np.array,
        terminal: bool
    ):
        index = self.memory_coutner % self.max_size
        self.states[index] = torch.from_numpy(state)
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = torch.from_numpy(next_state)
        self.terminals[index] = terminal
        self.memory_coutner += 1

    def sample_buffer(self, batch_size: int):
        n_filled_rows = min(self.memory_coutner, self.max_size)
        batch_idxs = np.random.choice(n_filled_rows, batch_size, replace=False)

        states = self.states[batch_idxs]
        actions = self.actions[batch_idxs]
        rewards = self.rewards[batch_idxs]
        next_states = self.next_states[batch_idxs]
        terminals = self.terminals[batch_idxs]

        return states, actions, rewards, next_states, terminals


# Setup
epsilon = 0.15
gamma = 0.99
learning_rate = 0.001
tau = 0.01
replay_buffer_size = 100_000
batch_size = 128
train_episodes = 75
eval_episodes = 100

env = gym.make('Acrobot-v1')
agent = DQNagent(env.observation_space, env.action_space, epsilon, gamma, learning_rate, tau)
replay_buffer = ReplayBuffer(replay_buffer_size, env.observation_space.shape)


# Training
for _ in tqdm(range(train_episodes)):
    state, _ = env.reset()

    while True:
        action = agent.choose_action(state)
        next_state, reward, terminal, truncated, info = env.step(action)

        replay_buffer.store_transition(state, action, reward, next_state, terminal or truncated)
        if replay_buffer.memory_coutner >= batch_size:
            batch = replay_buffer.sample_buffer(batch_size)
            agent.learn(batch)

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
