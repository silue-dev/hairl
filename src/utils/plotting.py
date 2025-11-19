import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curves(benchmark: str,
                         training_type: str, 
                         airl_performance: dict[str, np.ndarray], 
                         hairl_performance: dict[str, np.ndarray], 
                         window_size: int = 2,
                         save: bool = True,
                         base_dir: str = '') -> None:
    """
    This function creates and saves plots of the learning curves of agents, given that
    their rewards are provided.

    Arguments
    ---------
    benchmark          :  The name of the benchmark.
    training_type      :  The type of training done.
    airl_performance   :  The performance dictionary of airl.
    hairl_performance :  The performance dictionary of hairl.
    window_size        :  The size of the moving average window.

    """
    # Define dictionaries for reward data, alignment data, distribution data, and colors.
    rewards = {
        'AIRL': airl_performance.get('policy_rewards'), 
        'H-AIRL': hairl_performance.get('policy_rewards'), 
        'Expert': airl_performance.get('expert_rewards')
    }
    distributions = {
        'AIRL': airl_performance.get('policy_distributions'), 
        'H-AIRL': hairl_performance.get('policy_distributions'), 
        'Expert': airl_performance.get('expert_distributions')  
    }
    policy_alignments = {
        'AIRL': airl_performance.get('policy_alignments'), 
        'H-AIRL': hairl_performance.get('policy_alignments'), 
    }
    discriminator_accuracies = {
        'AIRL': airl_performance.get('discriminator_accuracies'), 
        'H-AIRL': hairl_performance.get('discriminator_accuracies'), 
    }
    colors = {
        'AIRL': '#83C266', 
        'H-AIRL': '#C26683', 
        'Expert': '#6683C2'
    }

    # Define a dictionary that brings together all the data.
    all_data = {
        'rewards': rewards,
        'distributions': distributions,
        'policy_alignments': policy_alignments,
        'discriminator_accuracies': discriminator_accuracies,
    }
    metric_to_axis = {
        'rewards': 'Reward',
        'distributions': 'Rate',
        'policy_alignments': 'Alignment (%)',
        'discriminator_accuracies': 'Accuracy (%)',
    }

    # Determine x-axis length.
    len_x = min(len(r['mean']) for r in rewards.values() if r)
    x = np.arange(len_x)
    
    # Smooth and plot AIRL rewards if they are provided.
    for metric, metric_data in all_data.items():
        # To plot distributions, we iterate over each column and create a new figure for each.
        if metric == 'distributions':
            if not None in metric_data.values():
                for col in range(list(metric_data.values())[0]['mean'].shape[-1]):
                    # Create a new figure for each column.
                    plt.figure(figsize=(11, 7))
                    do_plot = False

                    # Plot all algorithms together for this column.
                    for algo, metric_algo_data in metric_data.items():
                        if metric_algo_data and not np.isnan(metric_algo_data['mean']).any():
                            # Get smoothed mean and std data for the current column.
                            x_smooth, mean, std = get_smoothed_mean_and_std(
                                metric_algo_data['mean'][:, col], 
                                metric_algo_data['std'][:, col],
                                x, len_x, window_size
                            )

                            # Optionally, check if 'mean' contains data (if it might be empty).
                            if mean.size > 0:
                                do_plot = True
                                # Plot the data for the current algorithm.
                                plt.plot(x_smooth, mean, label=algo, color=colors[algo], linewidth=3)
                                plt.fill_between(x_smooth, mean - std, mean + std, color=colors[algo], alpha=0.3)

                    if do_plot:
                        # Add labels and legend only if something was plotted.
                        plt.xlabel('Training Steps', fontweight='bold', fontsize=28)
                        plt.ylabel(metric_to_axis[metric], fontweight='bold', fontsize=28)
                        plt.tick_params(axis='both', which='major', labelsize=26)
                        # plt.title(f'{training_type.upper()} Learning Curves on {benchmark} - Action {col}', fontweight='bold')
                        plt.grid(color='lightgrey', linewidth=0.5)
                        plt.legend(fontsize=28)
                        plt.tight_layout()

                        # Save or show the plot.
                        if save:
                            filename = f'src/plots/{base_dir}{training_type}/{metric}/{benchmark.lower()}_action{col}'
                            os.makedirs(os.path.dirname(filename), exist_ok=True)
                            plt.savefig(filename + '.pdf')
                        else:
                            plt.show()
                    plt.close()

        # For non-distribution metrics, plot all algorithms together on the same figure.
        else:
            plt.figure(figsize=(11, 7))
            do_plot = False

            for algo, metric_algo_data in metric_data.items():
                if metric_algo_data and not np.isnan(metric_algo_data['mean']).any():
                    # Get smoothed mean and std data for non-distribution metrics.
                    x_smooth, mean, std = get_smoothed_mean_and_std(
                        metric_algo_data['mean'], 
                        metric_algo_data['std'],
                        x, len_x, window_size
                    )
                    if benchmark.lower() == 'pendulum-v1':
                        # --- crop off the final third ---
                        crop_ratio = 0.65
                        cut_idx   = int(len(x_smooth) * crop_ratio)
                        x_smooth  = x_smooth[:cut_idx]
                        mean      = mean[:cut_idx]
                        std       = std[:cut_idx]
                    
                    # Check that data exists.
                    if mean.size > 0:
                        do_plot = True
                        # Plot the data for the current algorithm.
                        if algo == 'Expert' and training_type == 'rl': label = 'Env'
                        else: label = algo
                        plt.plot(x_smooth, mean, label=label, color=colors[algo], linewidth=3)
                        plt.fill_between(x_smooth, mean - std, mean + std, color=colors[algo], alpha=0.3)

            if do_plot:
                # Add labels and legend for non-distributions.
                plt.xlabel('Training Steps', fontweight='bold', fontsize=28)
                plt.ylabel(metric_to_axis[metric], fontweight='bold', fontsize=28)
                plt.tick_params(axis='both', which='major', labelsize=26)
                # plt.title(f'{training_type.upper()} Learning Curves on {benchmark}', fontweight='bold')
                plt.grid(color='lightgrey', linewidth=0.5)
                plt.legend(fontsize=28)
                plt.tight_layout()

                # Save or show the plot.
                if save:
                    filename = f'src/plots/{base_dir}{training_type}/{metric}/{benchmark.lower()}_{metric}'
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    plt.savefig(filename + '.pdf')
                else:
                    plt.show()
            plt.close()

def moving_average(y: np.ndarray, window_size: int) -> np.ndarray:
    """
    Applies a moving average to the data.

    Arguments
    ---------
    y           :  The y values of the data points.
    window_size :  The size of the moving average window.

    Returns
    -------
    The averaged y values.

    """
    return np.convolve(y, np.ones(window_size) / window_size, mode='valid')

def smooth_curve(x: np.ndarray, 
                 y: np.ndarray, 
                 window_size: int,
                 num_points: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    Smooths the given curve using a moving average followed by linear interpolation.

    Arguments
    ---------
    x           :  The x values of the data points.
    y           :  The y values of the data points.
    num_points  :  The number of points in the smoothed curve.
    window_size :  The size of the moving average window.

    Returns
    -------
    smooth_x :  The x values of the smoothed curve.
    smooth_y :  The y values of the smoothed curve.
    
    """
    # Apply moving average to y values to reduce noise.
    y_avg = moving_average(y, window_size)
    
    # Adjust x values to match the reduced length after moving average.
    x_avg = x[:len(y_avg)]

    # Create a new range of x values for a smoother curve.
    if len(x_avg) < 2:
        return x_avg, y_avg
    x_new = np.linspace(x_avg.min(), x_avg.max(), num_points)
    
    # Interpolate to get a smooth curve without requiring SciPy/OpenMP.
    y_smooth = np.interp(x_new, x_avg, y_avg)
    
    return x_new, y_smooth

def get_smoothed_mean_and_std(mean: np.ndarray, 
                              std: np.ndarray, 
                              x: np.ndarray, 
                              len_x: int, 
                              window_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooths the given mean and standard deviation values using a moving average 
    and linear interpolation.

    Arguments
    ---------
    mean        :  The mean values of the data points (numpy array).
    std         :  The standard deviation values of the data points (numpy array).
    x           :  The x values of the data points (numpy array).
    len_x       :  The length of the x values (integer).
    window_size :  The size of the moving average window (integer).

    Returns
    -------
    tuple : A tuple containing three numpy arrays:
            - The smoothed x values.
            - The smoothed mean values.
            - The smoothed standard deviation values.

    """
    # Ensure the mean and std have the correct length
    mean, std = mean[:len_x], std[:len_x]

    # Smooth the mean and standard deviation using the smooth_curve function
    x_smooth, mean_smooth = smooth_curve(x, mean, window_size)
    _, std_smooth = smooth_curve(x, std, window_size)

    return x_smooth, mean_smooth, std_smooth

def plot_study(
    benchmark: str,
    param_name: str,
    window_size: int = 2,
    study_grid_path: str = 'src/utils/param_study_grid.json',
):
    """
    Overlay OFAT runs for one parameter on a single plot, separately for IRL and RL.

    Arguments
    ---------
    benchmark: the Gym ID (e.g. "Pendulum-v1")
    param_name: one of 'gt_ratio_p', 'gt_ratio_d', 'noise_start', 'noise_final'
    window_size: smoothing window
    study_grid_path: path to the JSON grid definition

    """
    # Paths
    logs_root = os.path.join('src', 'logs', 'study', benchmark.lower())
    save_root = os.path.join('src', 'plots', 'study', benchmark.lower())

    # Load the grid
    with open(study_grid_path, 'r') as f:
        grid = json.load(f)

    # Pick out which runs vary this parameter
    runs = []
    for idx_str, info in grid.items():
        ov = info['override']
        idx = int(idx_str)
        if param_name in ov:
            runs.append((idx, ov[param_name]))
    runs.sort(key=lambda x: x[1])

    for phase in ['irl', 'rl']:

        # Identify file name:
        out_dir = os.path.join(save_root, param_name, phase)
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f'{param_name}_{phase}')

        loc = None
        crop = False
        if fname == 'src/plots/study/mountaincar-v0/gt_ratio_p/irl/gt_ratio_p_irl':
            loc = 'lower right'
        if fname == 'src/plots/study/mountaincar-v0/gt_ratio_d/rl/gt_ratio_d_rl':
            crop = True

        # styling defaults
        figsize     = (11, 7)
        xlabel_args = dict(fontweight='bold', fontsize=28)
        ylabel_args = dict(fontweight='bold', fontsize=28)
        tick_args   = dict(axis='both', which='major', labelsize=26)
        grid_args   = dict(color='lightgrey', linewidth=0.5)
        legend_args = dict(fontsize=20, loc=loc)

        # First pass: find the minimum available length
        min_len = None
        for idx, val in runs:
            perf_npz = os.path.join(logs_root, str(idx), 'performance', f'{phase}.npz')
            if os.path.exists(perf_npz):
                data = np.load(perf_npz)
                mean = data['mean']
                l = len(mean)
                if min_len is None or l < min_len:
                    min_len = l

        if min_len is None:
            print(f"⚠️ No valid runs for phase {phase}, skipping.")
            continue
        else:
            if crop:
                min_len = int(min_len * 0.7)

        plt.figure(figsize=figsize)

        # Second pass: plot all runs cropped to min_len
        for idx, val in runs:
            perf_npz = os.path.join(logs_root, str(idx), 'performance', f'{phase}.npz')
            if not os.path.exists(perf_npz):
                print(f"⚠️  Missing {perf_npz}, skipping")
                continue
            data = np.load(perf_npz)
            mean = data['mean'][:min_len]
            std  = data['std'][:min_len]
            x    = np.arange(len(mean))

            # adjust smoothing window for RL if desired
            w = window_size * (4 if phase == 'rl' else 1)
            x_s, m_s, s_s = get_smoothed_mean_and_std(mean, std, x, len(x), w)

            # rename params
            greeks = {
                'gt_ratio_p': r'$\alpha$',
                'gt_ratio_d': r'$\beta$',
                'noise_start': r'$\sigma_{\text{start}}$',
                'noise_final': r'$\sigma_{\text{final}}$',
            }

            plt.plot(x_s, m_s, label=f'{greeks[param_name]}={val}', linewidth=3)
            plt.fill_between(x_s, m_s - s_s, m_s + s_s, alpha=0.2)

        plt.xlabel('Training Steps', **xlabel_args)
        plt.ylabel('Reward', **ylabel_args)
        plt.tick_params(**tick_args)
        plt.grid(**grid_args)
        plt.legend(**legend_args)
        plt.tight_layout()

        # save
        plt.savefig(fname + '.pdf')
        plt.close()
