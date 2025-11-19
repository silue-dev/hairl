import json

# Import default parameters.
try:
    # When running as a script.
    from training import default_params
except ImportError:
    # When imported as a module.
    from utils.training import default_params

def generate_study_grid(save_path: str = 'src/utils/param_study_grid.json') -> None:
    """
    Build an OFAT grid over:
      - gt_ratio_p
      - gt_ratio_d
      - noise_dict.start  (tagged as 'noise_start')
      - noise_dict.final  (tagged as 'noise_final')
    Each override now contains both the specific key being varied
    (e.g. 'noise_start': v) and the complete 'noise_dict'.

    """
    # Pick a representative benchmark to fetch default noise settings.
    benchmark = next(iter(default_params))
    default_noise = default_params[benchmark]['irl'].noise_dict
    dim      = default_noise['dim']
    decay    = default_noise['decay']
    start0   = default_noise['start']
    final0   = default_noise['final']

    # Value ranges to sweep.
    gt_p_vals          = [0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
    gt_d_vals          = [0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
    noise_start_vals   = [0.00, 0.25, 0.50, 0.75, 1.00]
    noise_final_vals   = [0.00, 0.02, 0.08, 0.20, 0.50]

    grid = {}
    idx = 0

    # Sweep gt_ratio_p.
    for v in gt_p_vals:
        grid[idx] = {
            'override': {
                'gt_ratio_p': v
            }
        }
        idx += 1

    # Sweep gt_ratio_d.
    for v in gt_d_vals:
        grid[idx] = {
            'override': {
                'gt_ratio_d': v
            }
        }
        idx += 1

    # Sweep noise_dict.start (tag as 'noise_start').
    for v in noise_start_vals:
        nd = {
            'dim': dim,
            'start': v,
            'final': final0,
            'decay': decay
        }
        grid[idx] = {
            'override': {
                'noise_start': v,
                'noise_dict' : nd
            }
        }
        idx += 1

    # Sweep noise_dict.final (tag as 'noise_final').
    for v in noise_final_vals:
        nd = {
            'dim': dim,
            'start': start0,
            'final': v,
            'decay': decay
        }
        grid[idx] = {
            'override': {
                'noise_final': v,
                'noise_dict' : nd
            }
        }
        idx += 1

    # Save to JSON.
    with open(save_path, 'w') as f:
        json.dump(grid, f, indent=4)
    print(f"Saved OFAT study grid to {save_path} ({len(grid)} experiments)")


def load_study_grid(path: str = 'src/utils/param_study_grid.json') -> dict:
    """Load the OFAT study grid JSON."""
    with open(path, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    generate_study_grid()
