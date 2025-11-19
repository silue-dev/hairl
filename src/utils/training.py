import time
import torch
import rlcard
import numpy as np
from rlcard.utils import tournament
from rlcard.utils import Logger
from torch.utils.data import Dataset
from stable_baselines3.common.callbacks import BaseCallback

# --------------------------- #
# -- Functions and classes -- #
# --------------------------- #

class Timer:
    def __enter__(self):
        self.start = time.time()
        print('Start time:', time.strftime('%H:%M:%S', time.localtime(self.start)))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        self.elapsed = self.end - self.start

        print('\nTimer Summary:')
        print('Start time:', time.strftime('%H:%M:%S', time.localtime(self.start)))
        print('End time:', time.strftime('%H:%M:%S', time.localtime(self.end)))
        print('Elapsed time:', time.strftime('%Hh %Mm %Ss', time.gmtime(self.elapsed)))

def load_algo(algo, env, seed, base_dir):
    if algo == 'random':
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)
    else:
        model_path = f'src/trained/rlcard/{base_dir}{algo}/{algo}_{seed}.model'
        agent = torch.load(model_path, weights_only=False)
    
    return agent

def run_tournament(algos: list[str], 
                   num_games: int = 10_000,
                   model_seed: int = 42,
                   env_seed: int = None,
                   env_name: str = 'limit-holdem',
                   verbose: bool = False,
                   base_dir: str = '') -> None:
    """
    Runs the evaluation of RL poker agents.
    
    """
    # Make the environment.
    env = rlcard.make(env_name, config={'seed': env_seed})
    
    # Use a random agent as opponent.
    agents = []
    for algo in algos:
        agents.append(load_algo(algo, env, model_seed, base_dir))
    env.set_agents(agents)

    # Evaluate.
    if verbose:
        print(f"Running evaluation: {algos[0]} vs. {algos[1]}.")
    payoffs = np.array(tournament(env, num_games)) * 1000
    for position, payoff in enumerate(payoffs):
        if verbose:
            print(f"Payoff for {agents[position].__class__.__name__}: {payoff:,.1f} mbb/h.")
    
    return payoffs[0]

class TrajectoryDataset(Dataset):
    """
    A dataset class for handling trajectory data in reinforcement 
    learning tasks, based on the Pytorch `Dataset` interface.

    Arguments
    ---------
    states  :  The states of the trajectories.
    actions :  The actions corresponding to the states.

    """
    def __init__(self, 
                 states: torch.Tensor, 
                 actions: torch.Tensor,
                 rewards: torch.Tensor) -> None:
        self.states = states
        self.actions = actions
        self.rewards = rewards

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns
        -------
        The number of samples in the dataset.

        """
        return len(self.states)

    def __getitem__(self, 
                    idx: int) -> tuple[torch.Tensor, 
                                       torch.Tensor, 
                                       torch.Tensor | None]:
        """
        Retrieves the sample at the given index.

        Arguments
        ---------
        idx :  The index of the sample to retrieve.

        Returns
        -------
        A tuple containing the state and action at the given index.

        """
        state = self.states[idx]
        action = self.actions[idx]
        reward = self.rewards[idx] if self.rewards is not None else np.nan
        return state, action, reward
    
class SaveRewardsCallback(BaseCallback):
    """
    A custom callback for saving both IRL rewards and the original env rewards
    after each episode ends.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.rl_episode_rewards = []
        self.env_episode_rewards = []

    def _on_step(self) -> bool:
        # When an episode ends.
        if self.locals['dones'][0]:
            # Store the reward used during the RL process.
            rl_ep_rew = self.locals['infos'][0].get('episode', {}).get('r', None)
            if rl_ep_rew is not None:
                self.rl_episode_rewards.append(rl_ep_rew)

            # Store the environment reward if we are using a learned reward function.
            orig_info = self.locals['infos'][0].get('episode_original', None)
            if orig_info is not None:
                self.env_episode_rewards.append(orig_info['r'])

        return True

class CustomLogger(Logger):
    def __init__(self, 
                 verbose: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose

    def log(self, text):
        self.txt_file.write(text+'\n')
        self.txt_file.flush()
        if self.verbose:
            print(text)

    def log_performance(self, episode, reward):
        self.writer.writerow({'episode': episode, 'reward': reward})
        if self.verbose:
            print('')
        self.log('----------------------------------------')
        self.log('  episode      |  ' + str(episode))
        self.log('  reward       |  ' + str(reward))
        self.log('----------------------------------------')

    def __exit__(self, type, value, traceback):
        if self.txt_path is not None:
            self.txt_file.close()
        if self.csv_path is not None:
            self.csv_file.close()
        if self.verbose:
            print('\nLogs saved in', self.log_dir)

class IRL_Params:
    """
    A class for encapsulating parameters used in inverse reinforcement learning tasks.

    Arguments
    ---------
    hidden_dims_p   :  The hidden dimensions of the policy.
    hidden_dims_d   :  The hidden dimensions of the discriminator.
    training_steps  :  The number of training steps.
    learning_rate_p :  The learning rate of the policy.
    learning_rate_d :  The learning rate of the discriminator.
    weight_decay_p  :  The weight decay of the policy.
    weight_decay_d  :  The weight decay of the discriminator.
    num_transitions :  The number of transitions to collect.
    num_eval_runs   :  The number of evaluation runs.
    batch_size      :  The batch size.
    gt_ratio_p      :  The ground truth ratio of the policy.

    """
    def __init__(self, 
                 hidden_dims_p: tuple,
                 hidden_dims_d: tuple,
                 training_steps: int, 
                 learning_rate_p: float, 
                 learning_rate_d: float, 
                 weight_decay_p: float,
                 weight_decay_d: float,
                 num_transitions: int, 
                 num_eval_runs: int,
                 batch_size: int,
                 gt_ratio_p: float,
                 gt_ratio_d: float,
                 noise_dict: bool) -> None:
        self.hidden_dims_p = hidden_dims_p
        self.hidden_dims_d = hidden_dims_d
        self.training_steps = training_steps
        self.learning_rate_p = learning_rate_p
        self.learning_rate_d = learning_rate_d
        self.weight_decay_p = weight_decay_p
        self.weight_decay_d = weight_decay_d
        self.num_transitions = num_transitions
        self.num_eval_runs = num_eval_runs
        self.batch_size = batch_size
        self.gt_ratio_p = gt_ratio_p
        self.gt_ratio_d = gt_ratio_d
        self.noise_dict = noise_dict
    
    def to_dict(self):
        return self.__dict__

class RL_Params:
    """
    A class for encapsulating parameters used in reinforcement learning tasks.

    Arguments
    ---------
    batch_size    : The batch size.
    clip_range    : The clipping range for PPO.
    ent_coef      : The entropy coefficient.
    gae_lambda    : The lambda parameter for Generalized Advantage Estimation (GAE).
    gamma         : The discount factor.
    max_grad_norm : The maximum norm for gradient clipping.
    n_epochs      : The number of epochs per training iteration.
    n_steps       : The number of steps to collect per environment.
    vf_coef       : The value function coefficient in the loss.
    learning_rate : The learning rate for optimization.

    """
    def __init__(self,
                 batch_size: int | None = None,
                 clip_range: float | None  = None,
                 ent_coef: float | None  = None,
                 gae_lambda: float | None  = None,
                 gamma: float | None  = None,
                 max_grad_norm: float | None  = None,
                 n_epochs: int | None  = None,
                 n_steps: int | None  = None,
                 vf_coef: float | None  = None,
                 learning_rate: float | None  = None,
                 timesteps: int | None  = None) -> None:
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.vf_coef = vf_coef
        self.learning_rate = learning_rate
        self.timesteps = timesteps

    def to_dict(self):
        return self.__dict__

# ----------------------------- #
# -- Hyperparameter settings -- #
# ----------------------------- #

default_params = {
    'Acrobot-v1': {
        'irl': IRL_Params(
            hidden_dims_p=(64,),
            hidden_dims_d=(64,),
            training_steps=30,
            learning_rate_p=1e-5,
            learning_rate_d=1e-5,
            weight_decay_p=0,
            weight_decay_d=0,
            num_transitions=100_000,
            num_eval_runs=10,
            batch_size=64,
            gt_ratio_p=0.10,
            gt_ratio_d=0.00,
            noise_dict={
                'dim': 'batch',
                'start': 1.00, 
                'final': 0.04, 
                'decay': 0.01,
            }
        ),
        'rl': RL_Params(
            batch_size=64,
            clip_range=0.1,
            ent_coef=0.0004,
            gae_lambda=0.95,
            gamma=0.95,
            max_grad_norm=0.8,
            n_epochs=20,
            n_steps=512,
            vf_coef=0.50,
            learning_rate=1e-4,
            timesteps=100_000
        )
    },
    'Pendulum-v1': {
        'irl': IRL_Params(
            hidden_dims_p=(64,64),
            hidden_dims_d=(64,64),
            training_steps=30,
            learning_rate_p=2e-4,
            learning_rate_d=2e-4,
            weight_decay_p=0,
            weight_decay_d=0,
            num_transitions=300_000,
            num_eval_runs=10,
            batch_size=64,
            gt_ratio_p=0.33,
            gt_ratio_d=0.25,
            noise_dict={
                'dim': 'batch',
                'start': 1.00, 
                'final': 0.03, 
                'decay': 0.01, 
            }
        ),
        'rl': RL_Params(
            batch_size=64,
            clip_range=0.1,
            ent_coef=0.0004,
            gae_lambda=0.95,
            gamma=0.95,
            max_grad_norm=0.8,
            n_epochs=20,
            n_steps=256,
            vf_coef=0.50,
            learning_rate=5e-4,
            timesteps=150_000
        )
    },
    'Ant-v4': {
        'irl': IRL_Params(
            hidden_dims_p=(64,64),
            hidden_dims_d=(64,64),
            training_steps=30,
            learning_rate_p=2e-4,
            learning_rate_d=2e-4,
            weight_decay_p=0,
            weight_decay_d=0,
            num_transitions=300_000,
            num_eval_runs=10,
            batch_size=64,
            gt_ratio_p=0.33,
            gt_ratio_d=0.25,
            noise_dict={
                'dim': 'batch',
                'start': 1.00, 
                'final': 0.02, 
                'decay': 0.01,
            }
        ),
        'rl': RL_Params(
            batch_size=128,
            clip_range=0.1,
            ent_coef=0.0004,
            gae_lambda=0.92,
            gamma=0.98,
            max_grad_norm=0.8,
            n_epochs=20,
            n_steps=896,
            vf_coef=0.58,
            learning_rate=4e-5,
            timesteps=600_000
        )
    },
    'HalfCheetah-v4': {
        'irl': IRL_Params(
            hidden_dims_p=(64,64),
            hidden_dims_d=(64,64),
            training_steps=30,
            learning_rate_p=2e-4,
            learning_rate_d=2e-4,
            weight_decay_p=0,
            weight_decay_d=0,
            num_transitions=300_000,
            num_eval_runs=10,
            batch_size=64,
            gt_ratio_p=0.33,
            gt_ratio_d=0.25,
            noise_dict={
                'dim': 'batch',
                'start': 1.00, 
                'final': 0.02, 
                'decay': 0.01,
            }
        ),
        'rl': RL_Params(
            batch_size=64,
            clip_range=0.1,
            ent_coef=0.000401762,
            gae_lambda=0.92,
            gamma=0.98,
            max_grad_norm=0.8,
            n_epochs=20,
            n_steps=960,
            vf_coef=0.58096,
            learning_rate=1.6e-5,
            timesteps=1_000_000
        )
    },
    'LunarLander-v2': {
        'irl': IRL_Params(
            hidden_dims_p=(64,64),
            hidden_dims_d=(64,32),
            training_steps=50,
            learning_rate_p=2e-4,
            learning_rate_d=2e-4,
            weight_decay_p=0,
            weight_decay_d=0,
            num_transitions=300_000,
            num_eval_runs=20,
            batch_size=64,
            gt_ratio_p=0.33,
            gt_ratio_d=0.10,
            noise_dict={
                'dim': 'batch',
                'start': 1.00, 
                'final': 0.02, 
                'decay': 0.01,
            }
        ),
        'rl': RL_Params(
            batch_size=64,
            clip_range=0.2,
            ent_coef=0.01,
            gae_lambda=0.95,
            gamma=0.999,
            max_grad_norm=0.6,
            n_epochs=20,
            n_steps=960,
            vf_coef=0.50,
            learning_rate=3e-4,
            timesteps=500_000
        )
    },
    'MountainCar-v0': {
        'irl': IRL_Params(
            hidden_dims_p=(64,),
            hidden_dims_d=(64,),
            training_steps=40,
            learning_rate_p=3e-4,
            learning_rate_d=3e-4,
            weight_decay_p=0,
            weight_decay_d=0,
            num_transitions=100_000,
            num_eval_runs=10,
            batch_size=64,
            gt_ratio_p=0.10,
            gt_ratio_d=0.25,
            noise_dict={
                'dim': 'batch',
                'start': 1.00, 
                'final': 0.04, 
                'decay': 0.01,
            }
        ),
        'rl': RL_Params(
            timesteps=200_000
        )
    },
    'LimitHoldem-v0': {
        'irl': IRL_Params(
            hidden_dims_p=(64,),
            hidden_dims_d=(64,),
            training_steps=50,
            learning_rate_p=1e-3,
            learning_rate_d=1e-3,
            weight_decay_p=0,
            weight_decay_d=5e-3,
            num_transitions=300_000,
            num_eval_runs=1000,
            batch_size=256,
            gt_ratio_p=0.33,
            gt_ratio_d=0.00,
            noise_dict={
                'dim': 'batch',
                'start': 1.0, 
                'final': 0.1,
                'decay': 0.1,
            }
        )
    }
}
