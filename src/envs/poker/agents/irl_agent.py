import torch
import numpy as np
from copy import deepcopy
from rlcard.agents import DQNAgent as VanillaDQNAgent
from envs.poker.data.preprocessing import get_cards_vector, get_raises_vector

class DQNAgent(VanillaDQNAgent):
    """
    An extension of the DQNAgent that incorporates a custom reward function 
    that was obtained via Inverse Reinforcement Learning (IRL).

    Arguments
    ---------
    model_path    :  The path to the IRL model.
    payoff_weight :  The weight of the payoff in the reward.

    """
    def __init__(self, 
                 num_actions: int,
                 state_shape: list[int],
                 mlp_layers: list[int],
                 device: torch.device,
                 verbose: bool = True,
                 # Default params.
                 replay_memory_size=20_000,
                 replay_memory_init_size=100,
                 update_target_estimator_every=1000,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.0,
                 epsilon_decay_steps=5000,
                 batch_size=32,
                 train_every=1,
                 learning_rate=0.0005):
        
        # Initialize DQN.
        super().__init__(num_actions=num_actions,
                         state_shape=state_shape,
                         mlp_layers=mlp_layers,
                         device=device,
                         # Default params.
                         replay_memory_size=replay_memory_size,
                         replay_memory_init_size=replay_memory_init_size,
                         update_target_estimator_every=update_target_estimator_every,
                         discount_factor=discount_factor,
                         epsilon_start=epsilon_start,
                         epsilon_end=epsilon_end,
                         epsilon_decay_steps=epsilon_decay_steps,
                         batch_size=batch_size,
                         train_every=train_every,
                         learning_rate=learning_rate)
        
        self.verbose = verbose
    
    def train(self):
        ''' Train the network

        Returns:
            loss (float): The loss of the current batch.
        '''
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, legal_actions_batch = self.memory.sample()

        # Calculate best next actions using Q-network (Double DQN).
        q_values_next = self.q_estimator.predict_nograd(next_state_batch)
        legal_actions = []
        for b in range(self.batch_size):
            legal_actions.extend([i + b * self.num_actions for i in legal_actions_batch[b]])
        masked_q_values = -np.inf * np.ones(self.num_actions * self.batch_size, dtype=float)
        masked_q_values[legal_actions] = q_values_next.flatten()[legal_actions]
        masked_q_values = masked_q_values.reshape((self.batch_size, self.num_actions))
        best_actions = np.argmax(masked_q_values, axis=1)

        # Evaluate best next actions using Target-network (Double DQN).
        q_values_next_target = self.target_estimator.predict_nograd(next_state_batch)
        target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
            self.discount_factor * q_values_next_target[np.arange(self.batch_size), best_actions]

        # Perform gradient descent update.
        state_batch = np.array(state_batch)

        loss = self.q_estimator.update(state_batch, action_batch, target_batch)
        if self.verbose:
            print('\rINFO - Step {}, rl-loss: {}'.format(self.total_t, loss), end='')

        # Update the target estimator.
        if self.train_t % self.update_target_estimator_every == 0:
            self.target_estimator = deepcopy(self.q_estimator)
            if self.verbose:
                print("\nINFO - Copied model parameters to target network.")

        self.train_t += 1

        if self.save_path and self.train_t % self.save_every == 0:
            # To preserve every checkpoint separately, 
            # add another argument to the function call parameterized by self.train_t.
            self.save_checkpoint(self.save_path)
            if self.verbose:
                print("\nINFO - Saved model checkpoint.")

class IRL_DQNAgent(VanillaDQNAgent):
    """
    An extension of the DQNAgent that incorporates a custom reward function 
    that was obtained via Inverse Reinforcement Learning (IRL).

    Arguments
    ---------
    model_path    :  The path to the IRL model.
    payoff_weight :  The weight of the payoff in the reward.

    """
    def __init__(self, 
                 num_actions: int,
                 state_shape: list[int],
                 mlp_layers: list[int],
                 device: torch.device,
                 payoff_weight: float,
                 model_path: str,
                 verbose: bool = True,
                 # Default params.
                 replay_memory_size=20_000,
                 replay_memory_init_size=100,
                 update_target_estimator_every=1000,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.0,
                 epsilon_decay_steps=5000,
                 batch_size=32,
                 train_every=1,
                 learning_rate=0.0005):
        
        # Initialize DQN.
        super().__init__(num_actions=num_actions,
                         state_shape=state_shape,
                         mlp_layers=mlp_layers,
                         device=device,
                         # Default params.
                         replay_memory_size=replay_memory_size,
                         replay_memory_init_size=replay_memory_init_size,
                         update_target_estimator_every=update_target_estimator_every,
                         discount_factor=discount_factor,
                         epsilon_start=epsilon_start,
                         epsilon_end=epsilon_end,
                         epsilon_decay_steps=epsilon_decay_steps,
                         batch_size=batch_size,
                         train_every=train_every,
                         learning_rate=learning_rate)
        
        # Setup IRL.
        with open(model_path, 'rb') as f:
            self.reward_function = torch.load(f, weights_only=False)
        self.payoff_weight = payoff_weight
        self.actions = ['call', 'raise', 'fold', 'check']
        self.action_to_vector = {'call':[1,0,0,0], 'raise':[0,1,0,0],
                                 'fold':[0,0,1,0], 'check':[0,0,0,1]}
        self.verbose = verbose

    def feed(self, ts: list):
        """
        Retrieves the transition data, stores the data into the replay buffer, 
        and trains the agent.

        Arguments
        ---------
        ts :  The transition data as a list of 5 elements
              (state, action, payoff, next_state, and done).

        Returns
        -------
        An action id.

        """
        reward = self.get_reward(ts)
        state, action, payoff, next_state, done = ts
        self.feed_memory(state['obs'], 
                         action, 
                         reward, 
                         next_state['obs'], 
                         list(next_state['legal_actions'].keys()), 
                         done)
        self.total_t += 1
        tmp = self.total_t - self.replay_memory_init_size
        if tmp >= 0 and tmp % self.train_every == 0:
            self.train()

    def get_reward(self, ts: list) -> any:
        """
        Returns the reward from the reward function, given an transition.

        Arguments
        ---------
        ts :  The transition data as a list of 5 elements
              (state, action, payoff, next_state, and done).

        Returns
        -------
        reward :  The reward.

        """
        # Get state_action tensor.
        state, action, payoff, next_state, done = ts
        state_action_tensor = self.encode_state_action(state, action)

        # Compute the reward.
        w = self.payoff_weight
        raw_reward = self.reward_function(state_action_tensor).item()
        reward = w * payoff + (1 - w) * raw_reward
        
        return reward
    
    def encode_state_action(self, state, action) -> any:
        """
        Returns an encoded state-action tensor, given the state and the action
        in their raw form.

        Arguments
        ---------
        state  :  The state.
        action :  The raw action.
        
        """
        # Get state information.
        hand = state['raw_obs']['hand']
        public_cards = state['raw_obs']['public_cards']
        raises = state['raw_obs']['raise_nums']

        # Create the state vector.
        cards_vector = get_cards_vector(hand, public_cards, card_format='sr')
        raises_vector = get_raises_vector(raises)
        state_vector = cards_vector + raises_vector

        # Create the state-action tensor.
        action_vector = self.action_to_vector[self.actions[action]]
        state_action_tensor = torch.tensor(state_vector + action_vector)
        
        return state_action_tensor
    
    def train(self):
        ''' Train the network

        Returns:
            loss (float): The loss of the current batch.
        '''
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, legal_actions_batch = self.memory.sample()

        # Calculate best next actions using Q-network (Double DQN).
        q_values_next = self.q_estimator.predict_nograd(next_state_batch)
        legal_actions = []
        for b in range(self.batch_size):
            legal_actions.extend([i + b * self.num_actions for i in legal_actions_batch[b]])
        masked_q_values = -np.inf * np.ones(self.num_actions * self.batch_size, dtype=float)
        masked_q_values[legal_actions] = q_values_next.flatten()[legal_actions]
        masked_q_values = masked_q_values.reshape((self.batch_size, self.num_actions))
        best_actions = np.argmax(masked_q_values, axis=1)

        # Evaluate best next actions using Target-network (Double DQN).
        q_values_next_target = self.target_estimator.predict_nograd(next_state_batch)
        target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
            self.discount_factor * q_values_next_target[np.arange(self.batch_size), best_actions]

        # Perform gradient descent update.
        state_batch = np.array(state_batch)

        loss = self.q_estimator.update(state_batch, action_batch, target_batch)
        if self.verbose:
            print('\rINFO - Step {}, rl-loss: {}'.format(self.total_t, loss), end='')

        # Update the target estimator.
        if self.train_t % self.update_target_estimator_every == 0:
            self.target_estimator = deepcopy(self.q_estimator)
            if self.verbose:
                print("\nINFO - Copied model parameters to target network.")

        self.train_t += 1

        if self.save_path and self.train_t % self.save_every == 0:
            # To preserve every checkpoint separately, 
            # add another argument to the function call parameterized by self.train_t.
            self.save_checkpoint(self.save_path)
            if self.verbose:
                print("\nINFO - Saved model checkpoint.")
