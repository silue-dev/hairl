import os
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def main(algo: str = 'hairl'):
    # --- CONFIG ---
    benchmark = "MountainCar-v0"
    model_path = f"src/trained/{algo}/{benchmark.lower()}_discriminator.model"
    grid_size = 200

    # --- LOAD ENV & MODEL ---
    env = gym.make(benchmark)
    obs_low, obs_high = env.observation_space.low, env.observation_space.high
    n_actions = env.action_space.n

    discriminator = torch.load(model_path, map_location="cpu", weights_only=False)
    discriminator.eval()

    # --- BUILD GRID OF STATES ---
    xs = np.linspace(obs_low[0], obs_high[0], grid_size)
    ys = np.linspace(obs_low[1], obs_high[1], grid_size)
    xx, yy = np.meshgrid(xs, ys)
    states = np.stack([xx.ravel(), yy.ravel()], axis=1)
    states_t = torch.from_numpy(states).float()

    # --- COMPUTE REWARDS FOR EACH ACTION ---
    all_rewards = np.zeros((n_actions, states.shape[0]), dtype=np.float32)
    with torch.no_grad():
        for a in range(n_actions):
            actions_oh = torch.zeros(states.shape[0], n_actions)
            actions_oh[:, a] = 1.0
            inp = torch.cat([states_t, actions_oh], dim=1)
            r = discriminator(inp).squeeze()
            all_rewards[a] = r.numpy()

    # --- STYLING DEFAULTS ---
    xlabel_args = dict(fontweight='bold', fontsize=24)
    ylabel_args = dict(fontweight='bold', fontsize=24)
    tick_args   = dict(axis='both', which='major', labelsize=20)
    grid_args   = dict(color='lightgrey', linewidth=0.5)

    # --- PLOT EACH ACTION'S VALUE FUNCTION ---
    action_names = ['L', 'N', 'R']
    for a, name in enumerate(action_names):
        reward_map = all_rewards[a].reshape((grid_size, grid_size))

        plt.figure(figsize=(11, 7))
        plt.pcolormesh(xx, yy, reward_map, cmap='viridis', shading='auto')
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label('Reward', fontweight='bold', fontsize=20)

        plt.xlabel("Position", **xlabel_args)
        plt.ylabel("Velocity", **ylabel_args)
        plt.tick_params(**tick_args)
        plt.grid(**grid_args)
        plt.tight_layout()

        # safe filename
        fn = name.replace(' ', '_')
        out_base = f'src/plots/study/mountaincar-v0/reward_function/{algo}/value_{fn}'
        os.makedirs(os.path.dirname(out_base), exist_ok=True)
        plt.savefig(out_base + '.pdf')
        plt.show()
        plt.close()

    # --- DECISION MAP: best action at each state ---
    best_actions = np.argmax(all_rewards, axis=0).reshape((grid_size, grid_size))

    plt.figure(figsize=(11, 7))
    cmap = ListedColormap(["red", "green", "blue"])
    plt.pcolormesh(xx, yy, best_actions, cmap=cmap, shading="auto")

    cbar = plt.colorbar(ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(action_names, fontweight='bold', fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    plt.xlabel("Position", **xlabel_args)
    plt.ylabel("Velocity", **ylabel_args)
    plt.tick_params(**tick_args)
    plt.grid(**grid_args)
    plt.tight_layout()

    out_base = f'src/plots/study/mountaincar-v0/reward_function/{algo}/best_reward'
    plt.savefig(out_base + '.pdf')
    plt.show()
    plt.close()


if __name__ == "__main__":
    main(algo='airl')