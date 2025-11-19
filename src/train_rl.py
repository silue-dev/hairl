import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
from typing import Callable
from collections import defaultdict
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.irl_env import IRLGymEnv
from utils.training import SaveRewardsCallback
import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import tournament, reorganize
from utils.training import CustomLogger as Logger
from utils.training import RL_Params, default_params, benchmarks, Timer, run_tournament
from utils.plotting import plot_learning_curves

def create_irl_gym_env(base_env: gym.Env, 
                       reward_function: Callable) -> gym.Env:
    """
    Creates a vectorized IRLGymEnv environment. This is a Gym environment
    that has been augmented with a custom reward function that was learned
    using an IRL algorithm (AIRL or H-AIRL).

    Arguments
    ---------
    base_env        :  The base environment.
    reward_function :  The reward function to use in the environment.
    
    Returns
    -------
    The vectorized IRLGymEnv environment.

    """
    # Define a function to create the IRLGymEnv environment.
    def make_env() -> gym.Env:
        return Monitor(IRLGymEnv(env=base_env, 
                                 reward_function=reward_function))
    
    # Vectorize the environment using DummyVecEnv.
    vec_env = DummyVecEnv([make_env])
    return vec_env

def train_rl_gym(benchmark: str,
                 algo: str,
                 params: RL_Params = default_params,
                 verbose: bool = True,
                 base_dir: str = '') -> tuple[list, list]:

    # ---------- #
    # -- Algo -- #
    # ---------- #

    # Set up reward performance storage.
    rl_performance = {}

    # Get reward function.
    reward_function_path = f'src/trained/{base_dir}{algo}/{benchmark.lower()}_discriminator.model'
    reward_function = torch.load(reward_function_path, weights_only=False)

    # Create the IRL environment.
    irl_env = create_irl_gym_env(gym.make(benchmark), reward_function)

    # Initialize the callbacks to track rewards.
    irl_ppo_reward_callback = SaveRewardsCallback()

    # Set policy architecture.
    policy_kwargs = dict(
        log_std_init=-2,
        ortho_init=False,
        activation_fn=nn.LeakyReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    ) if benchmark in ['Ant-v4', 
                       'HalfCheetah-v4', 
                       'LunarLander-v2'] \
      else None
    
    # Create the PPO agent(s).
    irl_ppo_agent = PPO(
        policy='MlpPolicy',
        env=irl_env,
        batch_size=params[benchmark]['rl'].batch_size,
        clip_range=params[benchmark]['rl'].clip_range,
        ent_coef=params[benchmark]['rl'].ent_coef,
        gae_lambda=params[benchmark]['rl'].gae_lambda,
        gamma=params[benchmark]['rl'].gamma,
        max_grad_norm=params[benchmark]['rl'].max_grad_norm,
        n_epochs=params[benchmark]['rl'].n_epochs,
        n_steps=params[benchmark]['rl'].n_steps,
        vf_coef=params[benchmark]['rl'].vf_coef,
        policy_kwargs=policy_kwargs,
        learning_rate=params[benchmark]['rl'].learning_rate,
        verbose=verbose,
    ) if benchmark != 'MountainCar-v0' \
      else DQN(
          policy='MlpPolicy',
          env=irl_env,
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

    # Train the IRL-PPO agent.
    irl_ppo_agent.learn(total_timesteps=params[benchmark]['rl'].timesteps, 
                        callback=irl_ppo_reward_callback)
    
    # Store training performance.
    rl_performance['policy_rewards'] = irl_ppo_reward_callback.env_episode_rewards
    
    # ------------ #
    # -- Expert -- #
    # ------------ #

    # Also train the expert (if the algo is H-AIRL).
    if algo == 'airl':
        # Create the environment.
        base_env = gym.make(benchmark)

        # Create the PPO agent(s).
        expert_ppo_agent = PPO(
            policy='MlpPolicy',
            env=base_env,
            batch_size=params[benchmark]['rl'].batch_size,
            clip_range=params[benchmark]['rl'].clip_range,
            ent_coef=params[benchmark]['rl'].ent_coef,
            gae_lambda=params[benchmark]['rl'].gae_lambda,
            gamma=params[benchmark]['rl'].gamma,
            max_grad_norm=params[benchmark]['rl'].max_grad_norm,
            n_epochs=params[benchmark]['rl'].n_epochs,
            n_steps=params[benchmark]['rl'].n_steps,
            vf_coef=params[benchmark]['rl'].vf_coef,
            policy_kwargs=policy_kwargs,
            learning_rate=params[benchmark]['rl'].learning_rate,
            normalize_advantage=True,
            verbose=verbose
        ) if benchmark != 'MountainCar-v0' \
          else DQN(
              policy='MlpPolicy',
              env=base_env,
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

        # Initialize the callbacks to track rewards.
        expert_ppo_reward_callback = SaveRewardsCallback()
        
        # Train the expert PPO agent.
        expert_ppo_agent.learn(total_timesteps=params[benchmark]['rl'].timesteps, 
                               callback=expert_ppo_reward_callback)

        # Store training performance.
        rl_performance['expert_rewards'] = expert_ppo_reward_callback.rl_episode_rewards
    
    return rl_performance

def train_rl_card_agent(algo: str, 
                        payoff_weight: float = 0.0,
                        num_episodes: int = 5000,
                        num_eval_games: int = 1000,
                        eval_every: int = 100,
                        model_seed: int = 42,
                        env_seed: int = None,
                        save_dir: str = 'src/trained/rlcard/',
                        base_dir: str = '',
                        env_name: str = 'limit-holdem',
                        verbose: bool = True) -> None:
    # Hardware setup.
    device = torch.device('cpu')

    # Make the environment.
    env = rlcard.make(env_name, config={'seed': env_seed})

    # Initialize the agent.
    if algo == 'dqn':
        from envs.poker.agents.irl_agent import DQNAgent
        agent = DQNAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=[64,64],
            device=device,
            verbose=verbose
        )
    if algo == 'airl_dqn':
        from envs.poker.agents.irl_agent import IRL_DQNAgent
        agent = IRL_DQNAgent(
            # RL params.
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=[64,64],
            device=device,
            verbose=verbose,
            # IRL params.
            payoff_weight=payoff_weight,
            model_path='src/trained/airl/limitholdem-v0_discriminator.model',
        )
    if algo == 'hairl_dqn':
        from envs.poker.agents.irl_agent import IRL_DQNAgent
        agent = IRL_DQNAgent(
            # RL params.
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=[64,64],
            device=device,
            verbose=verbose,
            # IRL params.
            payoff_weight=payoff_weight,
            model_path=f'src/trained/{base_dir}hairl/limitholdem-v0_discriminator.model',
        )

    # Use a random agent as opponent.
    agents = [agent]
    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)

    # Set paths.
    log_path = save_dir + f'{base_dir}{algo}/logs'
    model_path = save_dir + f'{base_dir}{algo}/{algo}_{model_seed}.model'

    # Start training.
    if verbose:
        print(f"Training {algo} ... ")

    with Logger(log_dir=log_path, verbose=verbose) as logger:
        # Store best payoff.
        best_payoff = float('-inf')

        # Execute.
        training_payoffs = []
        for episode in range(num_episodes):
            if 'nfsp' in algo:
                agents[0].sample_episode_policy()
            
            # Generate data from the environment.
            trajectories, trajectory_payoffs = env.run(is_training=True)
            trajectories = reorganize(trajectories, trajectory_payoffs)
            
            # Feed the transitions into the agent (and train).
            for ts in trajectories[0]:
                agent.feed(ts)
            
            # Evaluate the payoff against a random agent.
            if episode % eval_every == 0:
                payoff = tournament(env, num_eval_games)[0] * 1000
                logger.log_performance(episode, payoff)
                training_payoffs.append(payoff)

                # Save the model if it performs well.
                if payoff > 0.8 * best_payoff:
                    torch.save(agent, model_path)
                    best_payoff = payoff if payoff > best_payoff \
                                                else best_payoff
                    if verbose:
                        print('model saved.')

    if verbose:
        print(f"Model {algo} (seed={model_seed}) trained and saved in {model_path}")
    
    return training_payoffs

def train_rl_card(algo: str, 
                  payoff_weight: float = 0.0,
                  num_episodes: int = 5000,
                  num_eval_games: int = 1000,
                  eval_every: int = 100,
                  model_seed: int = 42,
                  env_seed: int = None,
                  save_dir: str = 'src/trained/rlcard/',
                  env_name: str = 'limit-holdem',
                  verbose: bool = True,
                  base_dir: str = '') -> None:
    """
    Runs the training of the RL poker agent using RLCard.
    
    """
    # Set up performance storage.
    rl_performance = {}
    
    # Train agent.
    irl_training_payoffs = train_rl_card_agent(algo=f'{algo}_dqn',
                                               payoff_weight=payoff_weight,
                                               num_episodes=num_episodes,
                                               num_eval_games=num_eval_games,
                                               eval_every=eval_every,
                                               model_seed=model_seed,
                                               env_seed=env_seed,
                                               save_dir=save_dir,
                                               base_dir=base_dir,
                                               env_name=env_name,
                                               verbose=verbose)
    
    # Store performance.
    rl_performance['policy_rewards'] = irl_training_payoffs
    
    # Also train expert.
    expert_training_payoffs = train_rl_card_agent(algo='dqn',
                                                    payoff_weight=payoff_weight,
                                                    num_episodes=num_episodes,
                                                    num_eval_games=num_eval_games,
                                                    eval_every=eval_every,
                                                    model_seed=model_seed,
                                                    env_seed=env_seed,
                                                    save_dir=save_dir,
                                                    base_dir=base_dir,
                                                    env_name=env_name,
                                                    verbose=verbose)

    # Store performance.
    rl_performance['expert_rewards'] = expert_training_payoffs

    return rl_performance

def run_trainings(benchmark: str, 
                  algo: str,
                  verbose: bool = True,
                  num_experiments: int = 25) -> None:
    """
    Runs the RL training experiments.

    """
    # Initialize RL phase performance storage.
    rl_phase_performances = defaultdict(list)

    # Run training.
    for _ in range(num_experiments):
        # Train RL agent.
        rl_performance = train_rl_card(algo) if 'LimitHoldem' in benchmark else \
                         train_rl_gym(benchmark=benchmark, 
                                      algo=algo, 
                                      verbose=verbose)

        # Store performance.
        for metric, performance in rl_performance.items():
            rl_phase_performances[metric].append(performance)

    # Compute and store mean and std of performance metrics across all experiments.
    for metric, performances in rl_phase_performances.items():
        performance_length = min(len(performance) for performance in performances)
        trimmed_performances = [performance[:performance_length] for performance in performances]
        rl_phase_performances[metric] = {
            'mean': np.mean(trimmed_performances, axis=0),
            'std': np.std(trimmed_performances, axis=0)
        }

    return dict(rl_phase_performances)

def main(save: bool = True, num_experiments: int = 25):
    # Run trainings.
    for benchmark in benchmarks:
        # Train H-AIRL.
        hairl_rl_performance = run_trainings(
            benchmark=benchmark, 
            algo='hairl', 
            num_experiments=num_experiments
        )

        # Train AIRL.
        airl_rl_performance = run_trainings(
            benchmark=benchmark, 
            algo='airl', 
            num_experiments=num_experiments
        )

        # Save the learning curves for each benchmark.
        plot_learning_curves(benchmark=benchmark,
                             training_type='rl',
                             airl_performance=airl_rl_performance,
                             hairl_performance=hairl_rl_performance,
                             window_size=8,
                             save=save)
        
        print(f"Plotted curves for {benchmark}.")
        if benchmark != 'LimitHoldem-v0':
            print(f"\nRL Params: {default_params[benchmark]['rl'].to_dict()}")


if __name__ == '__main__':
    with Timer():
        main(save=True, num_experiments=20)

    run_tournament(algos=[f'hairl_dqn', 'dqn'], num_games=10_000, base_dir='', verbose=True)
    run_tournament(algos=[f'airl_dqn', 'dqn'], num_games=10_000, base_dir='', verbose=True)
