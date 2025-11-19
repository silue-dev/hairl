import os
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
from collections import defaultdict
import envs.poker.limit_holdem
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from hairl import HAIRL
from envs.poker.data.preprocessing import get_data as get_poker_data
from utils.training import default_params, benchmarks, Timer
from utils.plotting import plot_learning_curves
import warnings
warnings.simplefilter('ignore', FutureWarning)

def get_expert(env: gym.Env, 
               retrain: bool = False,
               verbose: bool = True) -> PPO:
    """
    Returns the expert PPO model given the environment.

    Arguments
    ---------
    env     :  The Gym environment object.
    retrain :  Whether to retrain the model from scratch.
    verbose :  Whether to print training logs.

    Returns
    -------
    expert :  The best-performing PPO expert model.
    
    """
    # Retrieve environment name and path.
    env_name = env.envs[0].spec.id
    expert_path = f'src/trained/expert/{env_name.lower()}.zip'

    # Return None if the environment is poker (we use the dataset, not PPO).
    if 'LimitHoldem' in env_name:
        return None

    # Check if the expert model already exists.
    if not retrain and os.path.exists(expert_path):
        if verbose:
            print(f"Loading existing expert {env_name} model.")
        return PPO.load(expert_path, env=env) if env_name != 'MountainCar-v0' \
          else DQN.load(expert_path, env=env)

    # Set policy architecture.
    policy_kwargs = dict(
        log_std_init=-2,
        ortho_init=False,
        activation_fn=nn.LeakyReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    ) if env_name in ['Ant-v4', 
                      'HalfCheetah-v4', 
                      'LunarLander-v2'] \
      else None

    # Create the expert model.
    expert = PPO(
        policy='MlpPolicy',
        env=env,
        batch_size=default_params[env_name]['rl'].batch_size,
        clip_range=default_params[env_name]['rl'].clip_range,
        ent_coef=default_params[env_name]['rl'].ent_coef,
        gae_lambda=default_params[env_name]['rl'].gae_lambda,
        gamma=default_params[env_name]['rl'].gamma,
        max_grad_norm=default_params[env_name]['rl'].max_grad_norm,
        n_epochs=default_params[env_name]['rl'].n_epochs,
        n_steps=default_params[env_name]['rl'].n_steps,
        vf_coef=default_params[env_name]['rl'].vf_coef,
        policy_kwargs=policy_kwargs,
        learning_rate=default_params[env_name]['rl'].learning_rate,
        normalize_advantage=True,
        verbose=verbose
    ) if env_name != 'MountainCar-v0' \
      else DQN(
          policy='MlpPolicy',
          env=env,
          batch_size=128,
          buffer_size=10_000,
          exploration_final_eps=0.07,
          exploration_fraction=0.2,
          gamma=0.98,
          gradient_steps=8,
          learning_rate=0.004,
          learning_starts=1000,
          target_update_interval=600,
          train_freq=16,
          policy_kwargs=dict(net_arch=[256, 256]),
          verbose=verbose
      )

    # Evaluation environment
    eval_env = Monitor(gym.make(env_name))

    # Callback to save the best model as `expert_path`
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=os.path.dirname(expert_path),
        log_path=None,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # Train the model with the callback
    if verbose:
        print(f"Training expert model on {env_name} ... ")
    expert.learn(
        total_timesteps=default_params[env_name]['rl'].timesteps, 
        callback=eval_callback
    )

    # Rename the model
    best_path = os.path.join(os.path.dirname(expert_path), "best_model.zip")
    os.replace(best_path, expert_path) 

    # Reload the best model after training
    expert = PPO.load(expert_path, env=env) if env_name != 'MountainCar-v0' \
        else DQN.load(expert_path, env=env)
    if verbose:
        print(f"Best model saved as {expert_path}")

    return expert


def generate_expert_transitions(benchmark: str, 
                                verbose: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates transitions by the expert in a Gym environment.

    Arguments
    ---------
    env :  The Gym environment object.

    Returns
    -------
    states_tensor  :  A tensor of states.
    actions_tensor :  The tensor of actions performed in those states.

    """
    # Create environment.
    env = DummyVecEnv([lambda: Monitor(gym.make(benchmark))])

    # Get the expert model.
    expert = get_expert(env)
    
    # Retrieve environment name.
    env_name = env.envs[0].spec.id

    # Retrieve the number of transitions to generate.
    num_transitions = default_params[benchmark]['irl'].num_transitions

    # If the environment is poker, get poker data.
    if 'LimitHoldem' in env_name:
        states, actions = get_poker_data(num_transitions, verbose=verbose)
        states_tensor = torch.tensor(np.array(states))
        actions_tensor = torch.tensor(np.array(actions))
        expert_data = states_tensor, actions_tensor, None
        return expert, expert_data

    # Get action space type.
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Set up state and action storage.
    states = []
    actions = []
    rewards = []

    if verbose:
        print(f"Generating {env_name} trajectories ...")

    transition = 0
    while transition < num_transitions:
        obs = env.reset()
        done = False
        while not done and transition < num_transitions:
            # Act and observe.
            obs_tensor = torch.tensor(obs)
            action, _ = expert.predict(obs_tensor)
            next_obs, reward, done, _ = env.step(action)

            # Store all the data.
            states.append(obs[0])
            actions.append(action if discrete else action[0])
            rewards.append(reward[0])

            # Update the iteration variables.
            obs = next_obs
            transition += 1
    if verbose:
        print('')
    # Create dataset.
    states_tensor = torch.tensor(np.array(states))
    actions_tensor = torch.tensor(np.array(actions))
    rewards_tensor = torch.tensor(np.array(rewards))
    expert_data = states_tensor, actions_tensor, rewards_tensor

    return expert, expert_data

def train_irl(algo: str,
              env: gym.Env,
              expert: PPO,
              expert_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
              hidden_dims_p: tuple,
              hidden_dims_d: tuple,
              learning_rate_p: float,
              learning_rate_d: float,
              weight_decay_p: float,
              weight_decay_d: float,
              training_steps: int,
              num_transitions: int,
              num_eval_runs: int,
              batch_size: int,
              gt_ratio_p: float,
              gt_ratio_d: float,
              noise_dict: dict | bool,
              verbose: bool,
              save: bool,
              base_dir: str = '') -> dict[str, any]:
    """
    Trains the (Hybrid-)AIRL algorithm.
    
    Arguments
    ---------
    env             :  The Gym environment instance.
    expert          :  The expert model.
    expert_data     :  The dataset of the expert behavior (trajectories).
    hidden_dims_p   :  The hidden dimensions for the policy network.
    hidden_dims_d   :  The hidden dimensions for the discriminator network.
    learning_rate_p :  The learning rate for the policy network.
    learning_rate_d :  The learning rate for the discriminator network.
    weight_decay_p  :  The weight decay of the policy network.
    weight_decay_d  :  The weight decay of the discriminator network.
    training_steps  :  The number of training steps.
    num_transitions :  The number of transitions to generate.
    num_eval_runs   :  The number of runs to perform during an evaluation.
    batch_size      :  The batch size for training.
    gt_ratio_p      :  The ground truth ratio of the policy.
    gt_ratio_d      :  The ground truth ratio of the discriminator.
    verbose         :  The verbosity.
    save            :  Whether or not to save the trained model.

    Returns
    -------
    eval_performance :  The overall evaluation performance dictionary, which 
                        includes rewards, alignments, and action distributions.

    """
    # Initialize and train (Hybrid-)AIRL.
    hairl = HAIRL(algo=algo,
                    env=env,
                    expert=expert,
                    hidden_dims_p=hidden_dims_p,
                    hidden_dims_d=hidden_dims_d,
                    learning_rate_p=learning_rate_p,
                    learning_rate_d=learning_rate_d,
                    weight_decay_p=weight_decay_p,
                    weight_decay_d=weight_decay_d,
                    training_steps=training_steps,
                    num_transitions=num_transitions,
                    num_eval_runs=num_eval_runs,
                    batch_size=batch_size,
                    gt_ratio_p=gt_ratio_p,
                    gt_ratio_d=gt_ratio_d,
                    noise_dict=noise_dict,
                    verbose=verbose,
                    base_dir=base_dir)
    
    eval_performance = hairl.train(expert_data, save=save)
    return eval_performance

def run_trainings(benchmark: str, 
                  algo: str,
                  expert: PPO,
                  expert_data: tuple[torch.Tensor, 
                                     torch.Tensor, 
                                     torch.Tensor | None],
                  params: dict = default_params,
                  verbose: bool = True,
                  save: bool = True,
                  num_experiments: int = 25) -> None:
    """
    Runs the IRL training experiments for Gym benchmarks.

    """
    # Initialize performance storage.
    irl_phase_performances = defaultdict(list)

    # Run training.
    for _ in range(num_experiments):
        # Create environment.
        env = DummyVecEnv([lambda: Monitor(gym.make(benchmark))])

        # Train (Hybrid-)AIRL.
        irl_eval_performance = train_irl(
            algo=algo,
            env=env, 
            expert=expert,
            expert_data=expert_data,
            hidden_dims_p=params[benchmark]['irl'].hidden_dims_p,
            hidden_dims_d=params[benchmark]['irl'].hidden_dims_d,
            learning_rate_p=params[benchmark]['irl'].learning_rate_p,
            learning_rate_d=params[benchmark]['irl'].learning_rate_d,
            weight_decay_p=params[benchmark]['irl'].weight_decay_p,
            weight_decay_d=params[benchmark]['irl'].weight_decay_d,
            training_steps=params[benchmark]['irl'].training_steps,
            num_transitions=params[benchmark]['irl'].num_transitions,
            num_eval_runs=params[benchmark]['irl'].num_eval_runs,
            batch_size=params[benchmark]['irl'].batch_size,
            gt_ratio_p=params[benchmark]['irl'].gt_ratio_p if algo == 'hairl' else 0,
            gt_ratio_d=params[benchmark]['irl'].gt_ratio_d if algo == 'hairl' else 0,
            noise_dict=params[benchmark]['irl'].noise_dict if algo == 'hairl' else False,
            verbose=verbose,
            save=save
        )
        
        # Store performance.
        for metric, performance in irl_eval_performance.items():
            irl_phase_performances[metric].append(performance)
        
    # Compute and store mean and std of performance metrics across all experiments.
    for metric, performances in irl_phase_performances.items():
        # Compute.
        mean = np.mean(performances, axis=0)
        std = np.std(performances, axis=0)
        
        # Store.
        irl_phase_performances[metric] = {
            'mean': mean,
            'std': std
        }

    return dict(irl_phase_performances)

def main(save: bool = True, num_experiments: int = 25):
    # Run trainings.
    for benchmark in benchmarks:
        # Generate expert data.
        expert, expert_data = generate_expert_transitions(benchmark)

        # Train AIRL.
        airl_irl_performance = run_trainings(
            benchmark=benchmark, 
            algo='airl', 
            expert=expert,
            expert_data=expert_data,
            save=save,
            num_experiments=num_experiments
        )

        # Train H-AIRL.
        hairl_irl_performance = run_trainings(
            benchmark=benchmark, 
            algo='hairl', 
            expert=expert,
            expert_data=expert_data,
            save=save,
            num_experiments=num_experiments
        )

        # Save the learning curves for each benchmark.
        plot_learning_curves(benchmark=benchmark,
                             training_type='irl',
                             airl_performance=airl_irl_performance,
                             hairl_performance=hairl_irl_performance,
                             window_size=2,
                             save=save)
        
        print(f"Plotted curves for {benchmark}.")
        print(f"\nIRL Params: {default_params[benchmark]['irl'].to_dict()}")


if __name__ == '__main__':
    with Timer():
        main(save=True, num_experiments=20)
