import torch
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete
from typing import Callable

class IRLGymEnv(gym.Env):
    """
    An environment wrapper that uses a learned reward for training,
    but also logs the *true* environment reward for evaluation.
    """
    def __init__(self, 
                 env: gym.Env, 
                 reward_function: Callable) -> None:
        super(IRLGymEnv, self).__init__()
        self.env = env
        self.obs = None
        self.discrete = isinstance(env.action_space, Discrete)
        
        # Standard gym attributes.
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        # The IRL reward function.
        self.reward_function = reward_function

        # For tracking original (true) env rewards.
        self.original_cumulative_reward = 0.0
        self.episode_length = 0

    def reset(self, 
              seed: int | None = None, 
              options: dict | None = None) -> tuple[np.ndarray, dict]:
        # Reset the underlying env.
        self.obs, info = self.env.reset(seed=seed, options=options)

        # Reset counters.
        self.original_cumulative_reward = 0.0
        self.episode_length = 0
        
        return self.obs, info

    def step(self, action):
        # Step the underlying env.
        next_obs, original_reward, terminated, truncated, info = self.env.step(action)
        
        # Accumulate the original env reward.
        self.original_cumulative_reward += original_reward
        self.episode_length += 1

        # Compute IRL-based reward (the one used for training).
        if self.discrete:
            # One-hot encode the action if discrete.
            one_hot_action = np.zeros(self.action_space.n)
            one_hot_action[action] = 1.0
            action_input = one_hot_action
        else:
            action_input = action
        
        irl_reward = self.reward_function(
            torch.from_numpy(np.append(self.obs, action_input))
        )

        # If the episode is done or truncated, store the original cumulative reward in info.
        if terminated or truncated:
            info["episode_original"] = {
                "r": self.original_cumulative_reward,
                "l": self.episode_length
            }
            self.original_cumulative_reward = 0.0
            self.episode_length = 0

        # Update the current observation.
        self.obs = next_obs
        return self.obs, irl_reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()
