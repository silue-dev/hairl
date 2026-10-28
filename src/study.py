import argparse
import json
import os
import copy
import multiprocessing
import numpy as np
import torch
from main import generate_expert_transitions, run_trainings, benchmarks
from utils.training import default_params, Timer
from utils.studying import load_study_grid
from utils.plotting import plot_study

def run_single(param_index: int,
               last_N: int,
               benchmark: str,
               study_root: str = 'src/logs/study',
               num_experiments: int = 20,
               verbose: bool = False):
    # Load override config
    grid   = load_study_grid('src/utils/param_study_grid.json')
    config = grid[str(param_index)]['override']

    # Apply override to a fresh copy of default_params
    params = copy.deepcopy(default_params)
    irl_p  = params[benchmark]['irl']
    for k, v in config.items():
        if k == 'noise_dict':
            irl_p.noise_dict = v
        else:
            setattr(irl_p, k, v)

    # Generate expert data and run IRL + RL
    expert, expert_data = generate_expert_transitions(benchmark, verbose=verbose)
    with Timer():
        irl_perf, rl_perf = run_trainings(
            benchmark=benchmark,
            algo='hairl',
            expert=expert,
            expert_data=expert_data,
            params=params,
            save=True,
            num_experiments=num_experiments,
            base_dir=f'vsc/{param_index}/',
            verbose=verbose
        )

    # Save raw performance curves
    summary_dir = os.path.join(study_root, benchmark.lower(), str(param_index))
    perf_dir    = os.path.join(summary_dir, 'performance')
    os.makedirs(perf_dir, exist_ok=True)

    np.savez(os.path.join(perf_dir, 'irl.npz'),
             mean=irl_perf['policy_rewards']['mean'],
             std= irl_perf['policy_rewards']['std'])
    np.savez(os.path.join(perf_dir, 'rl.npz'),
             mean=rl_perf['policy_rewards']['mean'],
             std= rl_perf['policy_rewards']['std'])

    # Write summary JSON
    irl_m = irl_perf['policy_rewards']['mean']
    irl_s = irl_perf['policy_rewards']['std']
    rl_m  = rl_perf['policy_rewards']['mean']
    rl_s  = rl_perf['policy_rewards']['std']
    summary = {
        'param_index':    param_index,
        'benchmark':      benchmark,
        'override':       config,
        'final_mean_irl': float(np.mean(irl_m[-last_N:])),
        'final_std_irl':  float(np.mean(irl_s[-last_N:])),
        'final_mean_rl':  float(np.mean(rl_m[-last_N:])),
        'final_std_rl':   float(np.mean(rl_s[-last_N:]))
    }
    with open(os.path.join(summary_dir, 'results.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"✔ Completed run {param_index} for {benchmark} → {summary_dir}")


def run_collection(param_name: str,
                   benchmark: str,
                   last_N: int = 1):
    grid = load_study_grid('src/utils/param_study_grid.json')

    # Gather (idx, value) pairs for this parameter
    runs = []
    for idx_str, info in grid.items():
        ov = info['override']
        idx = int(idx_str)
        if param_name in ov:
            runs.append((idx, ov[param_name]))

    runs.sort(key=lambda x: x[1])

    # Run each index
    logs_root = os.path.join('src', 'logs', 'study', benchmark.lower(), param_name)
    for idx, val in runs:
        print(f"\n▶ [{benchmark} | {param_name}={val}] (idx={idx})")
        run_single(
            param_index=idx,
            last_N=last_N,
            benchmark=benchmark,
            study_root=os.path.join('src', 'logs', 'study', benchmark.lower())
        )

    # Overlay plots
    save_root = os.path.join('src', 'plots', 'study', benchmark.lower())
    plot_study(
        param_name=param_name,
        study_grid_path='src/utils/param_study_grid.json',
        logs_root=logs_root,
        window_size=2,
        save_root=save_root
    )
    print(f"\n✔ Overlay plots → {save_root}/{param_name}/")


if __name__ == '__main__':
    # Pin PyTorch thread pools
    n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()-2))
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(n_threads)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--benchmark',
        type=str,
        choices=benchmarks,
        default=benchmarks[0],
        help="Which Gym benchmark to run the study on"
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--param_index',
        type=int,
        help="HPC mode: run exactly one config index"
    )
    group.add_argument(
        '--param_name',
        type=str,
        choices=['gt_ratio_p', 'gt_ratio_d', 'noise_start', 'noise_final'],
        default='gt_ratio_p',
        help="Local mode: run & overlay all values (default: %(default)s)"
    )
    parser.add_argument('--last_N', type=int, default=10)
    args = parser.parse_args()

    # Run
    if args.param_index is not None:
        run_single(
            param_index=args.param_index,
            last_N=args.last_N,
            benchmark=args.benchmark
        )
    else:
        run_collection(
            param_name=args.param_name,
            benchmark=args.benchmark,
            last_N=args.last_N
        )
