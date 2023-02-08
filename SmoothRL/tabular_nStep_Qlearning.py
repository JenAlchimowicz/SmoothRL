import gymnasium as gym
import numpy as np
from typing import Optional, Hashable
from itertools import count

from tqdm import tqdm


class nStep_QlearningAgent:
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
        state_idx: int,
        action_idx: int,
        target: float
    ):
        self.q_table[state_idx, action_idx] = self.q_table[state_idx, action_idx] + self.step_size * (target - self.q_table[state_idx, action_idx])


# Setup
epsilon = 0.15
gamma = 0.99
step_size = 0.02
n = 5
train_episodes = 5_000
eval_episodes = 1_000

env = gym.make('Taxi-v3')
agent = nStep_QlearningAgent(env.observation_space, env.action_space, epsilon, gamma, step_size)


# Training
for _ in tqdm(range(train_episodes)):
    state, _ = env.reset()
    action = agent.choose_action(state)

    states = [state]
    rewards = [0.0]
    actions = [action]

    T = np.inf
    for t in count():

        # Take step
        if t < T:
            next_state, reward, terminal, truncated, info = env.step(action)
            states.append(next_state)
            rewards.append(reward)

            if terminal or truncated:
                T = t + 1
            else:
                next_action = agent.choose_action(next_state)
                actions.append(next_action)

        # Learn
        timestep_of_update = t - n + 1
        if timestep_of_update > 0:  # take min n steps before start updating

            # Build target
            target = 0
            for i in range(timestep_of_update + 1, min(T, timestep_of_update + n) + 1):
                target += np.power(gamma, i - timestep_of_update - 1) * rewards[i]
            
            if timestep_of_update + n < T:
                nth_state_idx = agent.get_state_idx(states[timestep_of_update + n])
                target += np.power(gamma, n) * np.max(agent.q_table[nth_state_idx])
                
            # Update Qtable
            state_idx = agent.get_state_idx(states[timestep_of_update])
            action_idx = actions[timestep_of_update]
            agent.update_q_table(state_idx, action_idx, target)

        state = next_state
        action = next_action

        if timestep_of_update == T - 1:
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
