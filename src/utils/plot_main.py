# plot_main.py
import os
import argparse
import numpy as np
from plotting import plot_learning_curves
from benchmark_config import benchmarks

def benchmark_choice(value: str) -> str:
    """Return the canonical benchmark name (case-insensitive)."""
    norm = value.lower()
    for benchmark in benchmarks:
        if benchmark.lower() == norm:
            return benchmark
    raise argparse.ArgumentTypeError(
        f"Unknown benchmark '{value}'. Expected one of: {', '.join(benchmarks)}"
    )

def phase_choice(value: str) -> str:
    """Validate and normalize the training phase name."""
    phases = {'irl', 'rl'}
    norm = value.lower()
    if norm not in phases:
        raise argparse.ArgumentTypeError("Phase must be 'irl' or 'rl'.")
    return norm

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate learning curve plots from stored performance logs.'
    )
    parser.add_argument(
        '-b', '--benchmark',
        nargs='+',
        type=benchmark_choice,
        metavar='BENCHMARK',
        help='Benchmark names to plot (default: all benchmarks).'
    )
    parser.add_argument(
        '-p', '--phase',
        nargs='+',
        type=phase_choice,
        metavar='PHASE',
        help="Training phases to plot: 'irl', 'rl', or both (default: both)."
    )
    return parser.parse_args()

LOGS_ROOT = 'src/logs/main'

def load_phase_perf(benchmark: str, algo: str, phase: str):
    """Returns a dict metric -> {'mean':…, 'std':…} by reading .npz files."""
    phase_dir = os.path.join(LOGS_ROOT, benchmark.lower(), algo, phase)
    perf = {}
    for fn in os.listdir(phase_dir):
        if not fn.endswith('.npz'):
            continue
        metric = fn[:-4]                # e.g. 'policy_rewards'
        data = np.load(os.path.join(phase_dir, fn))
        perf[metric] = {
            'mean': data['mean'],
            'std':  data['std']
        }
    return perf

def main():
    args = parse_args()
    selected_benchmarks = args.benchmark or benchmarks
    selected_phases = args.phase or ['irl', 'rl']
    for bm in selected_benchmarks:
        for phase in selected_phases:
            # Load both algos.
            airl_perf = load_phase_perf(bm, 'airl', phase)
            hairl_perf= load_phase_perf(bm, 'hairl', phase)

            # Pick a sensible window size.
            window = 2 if phase=='irl' else 8

            plot_learning_curves(
                benchmark=bm,
                training_type=phase,
                airl_performance=airl_perf,
                hairl_performance=hairl_perf,
                window_size=window,
                save=True,
                base_dir=f'main/{bm.lower()}/'
            )
            print(f"✔ Plotted {bm} [{phase}]")
    
if __name__ == '__main__':
    main()
