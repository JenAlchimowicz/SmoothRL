import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Optional, Tuple
from tqdm import tqdm


class Qnetwork(nn.Module):
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
        x = self.fc3(x)
        return x


class PrioritizedReplayBuffer:
    def __init__(self, max_size: int, state_dim: int):
        self.max_size = max_size
        self.memory_coutner = 0

        self.states = torch.zeros((max_size, *state_dim), dtype=torch.float)
        self.next_states = torch.zeros((max_size, *state_dim), dtype=torch.float)
        self.actions = torch.zeros(max_size, dtype=torch.int64)
        self.rewards = torch.zeros(max_size, dtype=torch.float)
        self.terminals = torch.zeros(max_size, dtype=torch.bool)
        self.priorities = torch.zeros(max_size, dtype=torch.float) + 0.01

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
        # Initialy set the priority to max of existing priorities for high chance of sampling new experiences
        self.priorities[index] = torch.max(self.priorities)
        self.memory_coutner += 1

    def get_sampling_probabilities(self, priority_scale: float) -> np.array:
        n_filled_rows_in_buffer = min(self.memory_coutner, self.max_size)
        priorities = self.priorities[:n_filled_rows_in_buffer].detach().numpy()
        scaled_probabilities = priorities ** priority_scale
        sampling_probabilties = scaled_probabilities / np.sum(scaled_probabilities)
        return sampling_probabilties
    
    def get_importance_sampling_weights(self, sampling_probabilities: torch.Tensor, beta: float = 1.) -> torch.Tensor:
        n_filled_rows_in_buffer = min(self.memory_coutner, self.max_size)
        importance_weights = (1/n_filled_rows_in_buffer * 1/sampling_probabilities) ** beta
        normalized_weights = importance_weights / torch.max(importance_weights)
        return normalized_weights

    def sample_buffer(self, batch_size: int, priority_scale: float = 1., beta: float = 1.):
        n_filled_rows_in_buffer = min(self.memory_coutner, self.max_size)
        sampling_probabilities = self.get_sampling_probabilities(priority_scale)
        batch_idxs = np.random.choice(n_filled_rows_in_buffer, batch_size, replace=False, p=sampling_probabilities)

        states = self.states[batch_idxs]
        actions = self.actions[batch_idxs]
        rewards = self.rewards[batch_idxs]
        next_states = self.next_states[batch_idxs]
        terminals = self.terminals[batch_idxs]

        priorities = self.priorities[batch_idxs].detach().clone()
        importance_weights = self.get_importance_sampling_weights(priorities, beta)

        return (states, actions, rewards, next_states, terminals), importance_weights, batch_idxs
    
    def update_priorities(self, indeces: torch.Tensor, errors: torch.Tensor, offset: float = 0.1):
        self.priorities[indeces] = torch.abs(errors) + offset


class DQNagent:
    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        epsilon: float,
        gamma: float,
        lr: float,
    ):
        self.action_space = action_space
        self.epsilon = epsilon
        self.gamma = gamma
        self.latest_batch_errors = None

        self.Qnetwork = Qnetwork(obs_space.shape[0], action_space.n)
        self.optimizer = torch.optim.Adam(self.Qnetwork.parameters(), lr=lr)

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

    def learn(self, batch: Tuple, importance_weights: torch.Tensor):
        states, actions, rewards, next_states, terminals = batch
        self.optimizer.zero_grad()

        # Get q_values of chosen state-action pairs
        q_states = self.Qnetwork(states)[torch.arange(len(states)), actions]

        # Build TD target
        q_next_states = self.Qnetwork(next_states)
        targets = rewards + self.gamma * torch.max(q_next_states, dim=1)[0] * (1 - terminals.float())

        # Get TD error and learn
        errors = q_states - targets
        loss = torch.mean(torch.square(errors) * importance_weights)
        loss.backward()
        self.optimizer.step()

        # Save errors for Replay Buffer priorities update
        self.latest_batch_errors = errors


# Training
epsilon = 0.15
gamma = 0.99
learning_rate = 0.001
replay_buffer_size = 100_000
batch_size = 128
priority_scale = 1.
beta = 1.
train_episodes = 100
eval_episodes = 100

env = gym.make('CartPole-v1')
agent = DQNagent(env.observation_space, env.action_space, epsilon, gamma, learning_rate)
replay_buffer = PrioritizedReplayBuffer(replay_buffer_size, env.observation_space.shape)

for _ in tqdm(range(train_episodes)):
    state, _ = env.reset()

    while True:
        action = agent.choose_action(state)
        next_state, reward, terminal, truncated, info = env.step(action)
        replay_buffer.store_transition(state, action, reward, next_state, terminal or truncated)

        if replay_buffer.memory_coutner >= batch_size:
            batch, importance_weights, batch_idxs = replay_buffer.sample_buffer(batch_size, priority_scale, beta)
            agent.learn(batch, importance_weights)
            replay_buffer.update_priorities(batch_idxs, agent.latest_batch_errors)

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
