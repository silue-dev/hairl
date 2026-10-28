# file: envs/poker/limit_holdem.py

import os
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import rlcard
from envs.poker.data.preprocessing import get_state_vector

def load_dqn(seed: int):
    """
    Load either a random agent or a saved RLCard agent from disk.
    
    """
    model_path = f'src/trained/rlcard/dqn/dqn_{seed}.model'
    agent = torch.load(model_path, map_location='cpu')
    if hasattr(agent, 'eval_mode'):
        agent.eval_mode = True
    return agent

class LimitHoldemGymEnv(gym.Env):
    """
    Custom Gym environment for Limit Texas Hold'em Poker.
    """
    def __init__(self,
                 opponent_algo: str = 'dqn',
                 model_seed: int = 42,
                 ) -> None:
        super(LimitHoldemGymEnv, self).__init__()
        self.player_id = 0
        self.opponent_id = 1

        # choose and load opponent.
        self.opponent_algo = opponent_algo
        self.env = rlcard.make('limit-holdem')
        self.dqn_opponent = load_dqn(model_seed)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=[61], dtype=np.float32
        )

    def reset(self,
              seed: int | None = None,
              options: dict | None = None
             ) -> tuple[np.ndarray, dict]:
        self.env.seed(seed)
        raw_obs = self.env.reset()[0]['raw_obs']
        obs, _, done = self.encode_raw_obs(raw_obs)
        return obs, {}

    def step(self, action_data: np.int_ | np.ndarray):
        # Opponent’s turn.
        if self.env.get_player_id() == self.opponent_id:
            state = self.env.get_state(self.opponent_id)
            if self.opponent_algo == 'random':
                opp_action = np.random.choice(
                    list(state['legal_actions'].keys())
                )
            else:
                if hasattr(self.dqn_opponent, 'eval_step'):
                    opp_action, _ = self.dqn_opponent.eval_step(state)
                else:
                    opp_action = self.dqn_opponent.step(state)
                if opp_action not in state['legal_actions']:
                    opp_action = np.random.choice(
                        list(state['legal_actions'].keys())
                    )
            self.env.step(opp_action)

            if self.env.is_over():
                raw_obs = self.env.get_state(self.player_id)['raw_obs']
                obs, reward, done = self.encode_raw_obs(raw_obs)
                return obs, reward * 1000., done, done, {}

        # Player’s turn.
        obs_dict = self.env.get_state(self.player_id)
        legal_actions = obs_dict['legal_actions']

        if isinstance(action_data, (np.integer, int)):
            action = int(action_data)
            if action not in legal_actions:
                action = np.random.choice(self.action_space.n)
        else:
            probs = np.array(action_data, dtype=np.float32)
            mask = np.array([i in legal_actions for i in range(len(probs))])
            probs = np.where(mask, probs, 0.0)
            if probs.sum() > 0:
                probs /= probs.sum()
                action = np.random.choice(len(probs), p=probs)
            else:
                action = np.random.choice(list(legal_actions.keys()))

        raw_obs = self.env.step(action)[0]['raw_obs']
        obs, reward, done = self.encode_raw_obs(raw_obs)
        return obs, reward * 1000., done, done, {}

    def encode_raw_obs(self, raw_obs: dict) -> tuple[np.ndarray, float, bool]:
        hand = raw_obs['hand']
        public_cards = raw_obs['public_cards']
        raises = raw_obs['raise_nums']

        obs = get_state_vector(hand, public_cards, raises, card_format='sr')
        obs = np.array(obs, dtype=np.float32)
        done = self.env.is_over()
        reward = self.env.get_payoffs()[self.player_id] if done else 0.0
        return obs, reward, done

    def close(self) -> None:
        pass

register(
    id='LimitHoldem-v0',
    entry_point='envs.poker.limit_holdem:LimitHoldemGymEnv'
)