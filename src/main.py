import os
import json
import torch
import argparse
import numpy as np
import multiprocessing
import gymnasium as gym
import envs.poker.limit_holdem
from stable_baselines3 import PPO
from collections import defaultdict
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from train_irl import generate_expert_transitions, train_irl
from train_rl import train_rl_gym, train_rl_card
from utils.training import (
    run_tournament, 
    default_params,
    benchmarks
)

def run_trainings(benchmark: str, 
                  algo: str,
                  expert: PPO,
                  expert_data: tuple[torch.Tensor, 
                                     torch.Tensor, 
                                     torch.Tensor | None],
                  params: dict = default_params,
                  verbose: bool = False,
                  save: bool = True,
                  num_experiments: int = 20,
                  base_dir: str = '',
                  run_rl: bool = True) -> None:
    """
    Runs the IRL and RL training experiments for Gym benchmarks.

    """
    # Initialize RL phase performance storage.
    irl_phase_performances = defaultdict(list)
    rl_phase_performances = defaultdict(list)
    rl_tournaments = []

    # Run training.
    print('Training started ...')
    for i in range(num_experiments):
        print(f"Experiment {i + 1}:")

        # ------------------ #
        # -- IRL Training -- #
        # ------------------ #

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
            save=save,
            base_dir=base_dir
        )
        
        # Store performance.
        for metric, performance in irl_eval_performance.items():
            irl_phase_performances[metric].append(performance)
        
        # ----------------- #
        # -- RL Training -- #
        # ----------------- #
        if run_rl:
            # Train RL agent.
            rl_performance = train_rl_card(algo, 
                                           base_dir=base_dir, 
                                           verbose=verbose) if 'LimitHoldem' in benchmark else \
                                           \
                             train_rl_gym(benchmark=benchmark, 
                                          algo=algo, 
                                          verbose=verbose,
                                          base_dir=base_dir)

            # Store performance.
            for metric, performance in rl_performance.items():
                rl_phase_performances[metric].append(performance)

            # For poker, also perform tournaments.
            if 'LimitHoldem' in benchmark:
                if verbose:
                    print('Running poker tournament ... ')
                num_games = 1000
                rl_tournaments += [run_tournament(algos=[f'{algo}_dqn', 'dqn'], 
                                                  num_games=num_games,
                                                  base_dir=base_dir,
                                                  verbose=verbose) for _ in range(50)]
        
    # -- IRL Performance Processing -- #
    # -------------------------------- #
    
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

    if run_rl:
        # -- RL Performance Processing -- #
        # ------------------------------- #

        # Compute and store mean and std of performance metrics across all experiments.
        for metric, performances in rl_phase_performances.items():
            performance_length = min(len(performance) for performance in performances)
            trimmed_performances = [performance[:performance_length] for performance in performances]
            rl_phase_performances[metric] = {
                'mean': np.mean(trimmed_performances, axis=0),
                'std': np.std(trimmed_performances, axis=0)
            }

        # For poker, also process tournaments.
        if 'LimitHoldem' in benchmark and not base_dir:
            # Compute performance mean and std.
            rl_tournament_performance = {
                algo: {
                    'mean': np.mean(rl_tournaments, axis=0),
                    'std': np.std(rl_tournaments, axis=0) / np.sqrt(num_games)
                },
            }
            if verbose:
                print(f"Tournament result:", rl_tournament_performance)

            # Read existing logs.
            tournament_log_path = f'src/logs/{base_dir}tournaments/tournaments.json'
            with open(tournament_log_path, 'r') as f:
                try:
                    all_results = json.load(f)
                except json.JSONDecodeError:
                    all_results = []

            # Append and save.
            all_results.append(rl_tournament_performance)
            with open(tournament_log_path, 'w') as f:
                json.dump(all_results, f, indent=4)
    
    # -- Save Performance -- #
    # ---------------------- #
    if save:
        save_root = os.path.join('src', 'logs', 'main', benchmark.lower(), algo)
        phases_to_save = [('irl', irl_phase_performances)]
        if run_rl:
            phases_to_save.append(('rl', rl_phase_performances))
        for phase, phase_perf in phases_to_save:
            perf_dir = os.path.join(save_root, phase)
            os.makedirs(perf_dir, exist_ok=True)
            for metric, data in phase_perf.items():
                np.savez(
                    os.path.join(perf_dir, f'{metric}.npz'),
                    mean=data['mean'],
                    std =data['std']
                )

    return dict(irl_phase_performances), dict(rl_phase_performances)

def main(save: bool = True, 
         num_experiments: int = 20, 
         verbose: bool = False, 
         benchmark: str | None = None,
         run_rl: bool = True):

    # Pin PyTorch thread pools, honoring any explicit thread-limit env vars
    def _desired_threads():
        def _read_int(var):
            val = os.environ.get(var)
            if val is not None:
                try:
                    parsed = int(val)
                    if parsed > 0:
                        return parsed
                except ValueError:
                    pass
            return None

        for var in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
            parsed = _read_int(var)
            if parsed:
                return parsed

        # Local runs default to using most cores while keeping at least one free.
        return max(1, multiprocessing.cpu_count() - 1)

    n_threads = _desired_threads()
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(n_threads)

    # Get benchmark
    if benchmark is not None:
        selected_benchmarks = [benchmark]
        print(f"▶ Running benchmark: {benchmark}")
    else:
        selected_benchmarks = benchmarks
        print(f"▶ Running all benchmarks")

    # Run trainings.
    for bm in selected_benchmarks:
        # Generate expert data.
        expert, expert_data = generate_expert_transitions(bm)

        # Train AIRL.
        run_trainings(
            benchmark=bm, 
            algo='airl', 
            expert=expert,
            expert_data=expert_data,
            save=save,
            num_experiments=num_experiments,
            verbose=verbose,
            run_rl=run_rl
        )

        
        # Train H-AIRL.
        run_trainings(
            benchmark=bm, 
            algo='hairl', 
            expert=expert,
            expert_data=expert_data,
            save=save,
            num_experiments=num_experiments,
            verbose=verbose,
            run_rl=run_rl
        )

        print(f"\nIRL Params: {default_params[bm]['irl'].to_dict()}")
        print(f"\nRL Params: {default_params[bm]['rl'].to_dict()}"
              if 'LimitHoldem' not in bm else '')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--benchmark',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--skip-rl',
        action='store_true',
        help='Skip the RL fine-tuning phase (run IRL only).'
    )
    args = parser.parse_args()
    verbose = args.benchmark is None

    main(save=True,
         num_experiments=1,
         verbose=True,
         benchmark=args.benchmark,
         run_rl=not args.skip_rl)
