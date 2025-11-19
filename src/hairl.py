import os
import time
import psutil
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.utils.data import DataLoader
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv
from utils.training import TrajectoryDataset

class Policy(nn.Module):
    """
    The policy MLP for both discrete and continuous actions.

    Arguments
    ---------
    env         :  The environment instance.
    hidden_dims :  The dimensions of the hidden layers.
    discrete    :  Whether the action space is discrete.
    
    """
    def __init__(self, 
                 env: gym.Env, 
                 hidden_dims: tuple[int, ...], 
                 discrete: bool) -> None:
        super(Policy, self).__init__()

        # Environment attributes.
        self.discrete = discrete
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete \
                          else env.action_space.shape[0]
        
        # Store the action space bounds.
        if not self.discrete:
            self.actions_low = torch.tensor(env.action_space.low, dtype=torch.float32)
            self.actions_high = torch.tensor(env.action_space.high, dtype=torch.float32)

        # Network.
        self.layers = nn.ModuleList()
        last_dim = self.state_dim
        for hidden_dim in hidden_dims:
            if hidden_dim > 0:
                self.layers.append(nn.Linear(last_dim, hidden_dim))
                last_dim = hidden_dim
        
        # Output.
        if self.discrete:
            self.output_layer = nn.Linear(last_dim, self.action_dim)
        else:
            self.mean_layer = nn.Linear(last_dim, self.action_dim)
            self.log_std_layer = nn.Parameter(torch.zeros(self.action_dim))
    
    def forward(self, 
                states: torch.Tensor) -> tuple[torch.Tensor, 
                                               torch.Tensor | None]:
        """
        Computes the network output.

        Arguments
        ---------
        states :  The batch of input states.

        Returns
        -------
        A tuple containing action probabilities (for discrete actions) 
        or means and standard deviations (for continuous actions).

        """
        x = states.float()
        for layer in self.layers:
            x = F.leaky_relu(layer(x))

        if self.discrete:
            logits = self.output_layer(x)
            action_probs = F.softmax(logits, dim=-1)
            return action_probs, None
        else:
            logits = self.mean_layer(x)
            squashed_means = torch.sigmoid(logits)
            means = self.actions_low + (self.actions_high - self.actions_low) * squashed_means
            stds = torch.exp(self.log_std_layer)
            return means, stds
    
    def predict(self, 
                states: torch.Tensor) -> tuple[torch.Tensor, 
                                               torch.Tensor]:
        """
        Returns the predicted action given a state.

        Arguments
        ---------
        states :  The batch of input states.

        Returns
        -------
        A tuple containing action probabilities and log probabilities.

        """
        if self.discrete:
            action_probs, _ = self.forward(states)
            dist = Categorical(action_probs)
            best_actions = torch.argmax(action_probs, dim=-1)
            log_probs = dist.log_prob(best_actions)
            return action_probs, log_probs
        else:
            means, stds = self.forward(states)
            dist = Normal(means, stds)
            actions = means + stds * torch.randn_like(stds)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            return actions, log_probs
        
    def act(self, 
            states: torch.Tensor) -> tuple[torch.Tensor, 
                                           torch.Tensor]:
        """
        Returns the action taken given a state.

        Arguments
        ---------
        state :  The batch of input states.

        Returns
        -------
        actions   :  The actions to be performed.
        log_probs :  The log probabilities of the actions.

        """
        action_outputs, log_probs = self.predict(states)
        if self.discrete:
            dist = Categorical(action_outputs)
            actions = dist.sample()
        else:
            actions = action_outputs
        return actions, log_probs

    def log_prob(self, 
                 states: torch.Tensor, 
                 actions: torch.Tensor) -> torch.Tensor:
        """
        Computes the log probability of a given action.

        Arguments
        ---------
        states  :  The batch of input states.
        actions :  The batch of taken actions.

        Returns
        -------
        log_probs :  The log probabilities of the actions.

        """
        if self.discrete:
            action_probs, _ = self.forward(states)
            dist = Categorical(probs=action_probs)
            best_actions = torch.argmax(actions, dim=-1)
            log_probs = dist.log_prob(best_actions)
            return log_probs
        else:
            means, stds = self.forward(states)
            dist = Normal(means, stds)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            return log_probs

class Discriminator(nn.Module):
    """
    The discriminator MLP for adversarial training.

    Arguments
    ---------
    env         :  The environment instance.
    hidden_dims :  The dimensions of the hidden layers.
    discrete    :  Whether the action space is discrete.

    """
    def __init__(self, 
                 env: gym.Env, 
                 hidden_dims: tuple[int, ...], 
                 discrete: bool) -> None:
        super(Discriminator, self).__init__()

        # Environment attributes.
        self.discrete = discrete
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete \
                          else env.action_space.shape[0]

        # Network.
        self.layers = nn.ModuleList()
        last_dim = self.state_dim + self.action_dim
        for hidden_dim in hidden_dims:
            if hidden_dim > 0:
                self.layers.append(nn.Linear(last_dim, hidden_dim))
                last_dim = hidden_dim
        self.output_layer = nn.Linear(last_dim, 1)
        self.tanh = nn.Tanh()
    
    def forward(self, 
                states_actions: torch.Tensor) -> torch.Tensor:
        """
        Computes the raw network output. The network is our reward function,
        so the output is the raw reward signal, also called the energy.

        Arguments
        ---------
        states_actions :  The state-action input batch tensor.

        Returns
        -------
        reward :  The raw reward signal (i.e., energy) output of the network.

        """
        x = states_actions.float()
        for layer in self.layers:
            x = F.leaky_relu(layer(x))
        reward = self.output_layer(x)

        return reward
    
    def log_D(self, 
              states_actions: torch.Tensor, 
              policy_log_probs: torch.Tensor) -> torch.Tensor:
        """
        Computes the (log) discriminator output. It is the negative 
        log likelihood of the state-action input being authentic.

        Arguments
        ---------
        states_actions   :  The state-action input batch tensor.
        policy_log_probs :  The policy log probabilities.

        Returns                                            
        log_Ds :  The log discriminator outputs.

        """
        rewards = self.forward(states_actions).squeeze()
        
        # Apply the log-sum-exp trick for numerical stability.
        # Originally, log_D = reward - log(exp(reward) + exp(policy_log_prob)).
        max_terms = torch.max(rewards, policy_log_probs)
        log_Ds = rewards - (max_terms \
                          + torch.log(torch.exp(rewards - max_terms) \
                          + torch.exp(policy_log_probs - max_terms)))
        return log_Ds

class HAIRL:
    """
    The Hybrid Adversarial Inverse Reinforcement Learning (H-AIRL) model.

    Arguments
    ---------
    env             :  The Gym environment object.
    expert          :  The expert model.
    hidden_dims_p   :  Hidden dimensions for the policy network.
    hidden_dims_d   :  Hidden dimensions for the discriminator network.
    learning_rate_p :  Learning rate for the policy network.
    learning_rate_d :  Learning rate for the discriminator network.
    weight_decay_p  :  The weight decay of the policy network.
    weight_decay_d  :  The weight decay of the discriminator network.
    training_steps  :  Number of training steps.
    num_transitions :  Number of transitions to generate.
    num_eval_runs   :  The number of runs to perform during an evaluation.
    batch_size      :  Batch size for training.
    gt_ratio_p      :  The ground truth ratio of the policy.
    gt_ratio_d      :  The ground truth ratio of the discriminator.
    verbose         :  The verbosity.
    base_dir        :  The base directory to save the model into.
    
    """
    def __init__(self,
                 algo: str,
                 env: DummyVecEnv,
                 expert: BaseAlgorithm,
                 hidden_dims_p: tuple[int, ...],
                 hidden_dims_d: tuple[int, ...],
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
                 noise_dict: bool,
                 verbose: bool = True,
                 base_dir: str = '') -> None:
        
        # The algorithm is either AIRL or H-AIRL
        self.algo = algo

        # Get environment specification.
        self.env = env
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.benchmark = self.env.envs[0].spec.id
        self.benchmark_is_poker = 'LimitHoldem' in self.benchmark
        
        # Define policy and discriminator.
        self.policy = Policy(env, hidden_dims_p, self.discrete)
        self.discriminator = Discriminator(env, hidden_dims_d, self.discrete)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), 
                                           lr=learning_rate_p, 
                                           weight_decay=weight_decay_p)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), 
                                                  lr=learning_rate_d, 
                                                  weight_decay=weight_decay_d)

        # Define the expert model and data.
        self.expert = expert

        # H-AIRL specific parameters.
        self.gt_ratio_p = gt_ratio_p
        self.gt_ratio_d = gt_ratio_d
        self.noise_dict = noise_dict
        
        # Other parameters.
        self.training_steps = training_steps
        self.num_transitions = num_transitions
        self.num_eval_runs = num_eval_runs
        self.batch_size = batch_size
        self.verbose = verbose
        self.base_dir = base_dir

        # CPU tracking
        self._proc = psutil.Process(os.getpid())
        
    def train(self, 
              expert_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
              split_ratio: float = 0.8,
              save: bool = True) -> dict[str, any]:
        """
        Trains the H-AIRL model.

        Arguments
        ---------
        expert_data :  The dataset of the expert behavior (trajectories).
        split_ratio :  The ratio that defines the training to validation data split.
        save        :  Whether or not to save the trained model.

        Returns
        -------
        eval_performances :  The overall evaluation performance dictionary, which 
                             includes rewards, alignments, and action distributions.

        """
        # Print CPU info
        aff = os.sched_getaffinity(0) if hasattr(os, 'sched_getaffinity') else []
        if self.verbose:
            print(f"OS affinity cores: {sorted(aff)}  (count={len(aff)})")
            print(f"PyTorch threads (intra/inter): "
                f"{torch.get_num_threads()}/"
                f"{torch.get_num_interop_threads()}")
            print("-" * 60)

        # Kick off CPU‐time counters
        self._proc.cpu_times()         # initialize counters
        t0 = time.time()
        c0 = self._proc.cpu_times().user + self._proc.cpu_times().system

        if self.verbose:
            print(f"Training {self.algo.upper()} on {self.benchmark}:")

        # Store evaluation performance.
        policy_eval_rewards = []
        expert_eval_rewards = []
        policy_eval_distributions = []
        expert_eval_distributions = []
        policy_eval_alignments = []
        discriminator_eval_accuracies = []

        # Define state and action data.
        all_states, all_expert_actions, all_expert_rewards = expert_data
        reward_data_exists = True if all_expert_rewards is not None else False

        # Turn discrete actions into one-hot encodings.
        if self.discrete:
            all_expert_actions = F.one_hot(
                all_expert_actions.squeeze(), 
                num_classes=self.policy.action_dim
            ).float()

        # Split the data in training and validation sets.
        split_idx = int(len(all_states) * split_ratio)
        train_states, val_states = all_states[:split_idx], all_states[split_idx:]
        train_expert_actions, val_expert_actions = all_expert_actions[:split_idx], \
                                                   all_expert_actions[split_idx:]
        train_expert_rewards = all_expert_rewards[:split_idx] if reward_data_exists else None
        
        # Create the data loaders.
        train_dataset = TrajectoryDataset(train_states, train_expert_actions, train_expert_rewards)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        for step in range(self.training_steps):
            # Store discriminations.
            policy_discriminations = []
            expert_discriminations = []

            for states, expert_actions, expert_rewards in train_dataloader:
                self.policy.train()
                self.discriminator.train()

                # ---------------------------- #
                # -- Discriminator Training -- #
                # ---------------------------- #

                # Reset the discriminator's gradients.
                self.discriminator_optimizer.zero_grad()

                # Get policy action and compute log probabilities.
                with torch.no_grad():
                    policy_actions, policy_log_probs = self.policy.predict(states)
                    expert_log_probs = self.policy.log_prob(states, expert_actions)

                    if self.algo.lower() == 'hairl' and self.noise_dict:
                        # Define noise parameters.
                        eps = 1e-9
                        noise_dim = self.noise_dict['dim']
                        start_noise = self.noise_dict['start'] * (self.env.action_space.high.flatten()[0] \
                                                                  if not self.discrete else 1) + eps
                        final_noise = self.noise_dict['final'] * (self.env.action_space.high.flatten()[0] \
                                                                  if not self.discrete else 1) + eps

                        if noise_dim == 'batch':
                            # Compute decay parameters.
                            batch_size = states.shape[0]
                            b = (start_noise - final_noise) / (final_noise * batch_size)
                            k = self.noise_dict['decay']  # decay speed (smaller = slower decay)

                            # Compute noise factors.
                            batch_indices = torch.arange(batch_size).to(policy_actions.device)
                            noise_factors = start_noise / (1 + k * b * batch_indices)
                            perm = torch.randperm(batch_size).to(policy_actions.device)
                            noise_factors = noise_factors[perm]
                            
                            # Create noise.
                            noise = torch.randn_like(policy_actions) * noise_factors.view(-1, 1)

                        if noise_dim == 'time':
                            # Compute decay parameters.
                            b = (start_noise - final_noise) / (final_noise * self.training_steps)
                            k = self.noise_dict['decay']
                            noise_factor = start_noise / (1 + k * b * step)

                            # Create noise.
                            noise = torch.randn_like(policy_actions) * noise_factor
                        
                        # Add noise to policy actions.
                        policy_actions = policy_actions + noise
                        if self.discrete:
                            policy_actions = torch.clamp(policy_actions, min=1e-4)
                            policy_actions = policy_actions / policy_actions.sum(dim=1, keepdim=True)

                # Concatenate the states and actions to form transitions.
                policy_transitions = torch.cat([states, policy_actions], dim=-1)
                expert_transitions = torch.cat([states, expert_actions], dim=-1)

                # Compute the discriminator's guesses on the authenticity 
                # of the policy and expert transitions, in log form.
                log_Ds_policy = self.discriminator.log_D(policy_transitions, policy_log_probs)
                log_Ds_expert = self.discriminator.log_D(expert_transitions, expert_log_probs)

                # Compute discriminator loss.
                discriminator_adversarial_loss = - torch.mean(log_Ds_expert) \
                                                 - torch.mean(torch.log(
                                                     1 - torch.exp(log_Ds_policy).clamp(max=1-1e-6)
                                                   ))

                if self.algo.lower() == 'hairl' and reward_data_exists:
                    # Compute reward.
                    discriminator_expert_rewards = self.discriminator(expert_transitions)

                    # Use the H-AIRL approach.
                    discriminator_ground_truth_loss = F.mse_loss(discriminator_expert_rewards.flatten(), 
                                                                 expert_rewards, 
                                                                 reduction='mean')
                    
                    
                    # Compute the discriminator loss.
                    w = self.gt_ratio_d
                    discriminator_loss = discriminator_ground_truth_loss * w \
                                         + discriminator_adversarial_loss * (1 - w)

                else:
                    discriminator_loss = discriminator_adversarial_loss

                # Backpropagate the loss and update the discriminator.
                discriminator_loss.backward()
                self.discriminator_optimizer.step()

                # --------------------------------- #
                # -- Generator (policy) Training -- #
                # --------------------------------- #

                # Reset the policy's gradients.
                self.policy_optimizer.zero_grad()

                # Get the actions and log probabilities from the policy.
                policy_actions, policy_log_probs = self.policy.predict(states)

                # Compute the transitions.
                policy_transitions = torch.cat([states, policy_actions], dim=-1)

                # Compute the adversarial loss.
                policy_rewards = self.discriminator(policy_transitions)
                policy_adversarial_loss = - (policy_rewards.mean() - policy_log_probs.mean())

                # Compute the policy loss.
                policy_ground_truth_loss = torch.tensor(0)
                if self.algo.lower() == 'hairl':
                    # Use the H-AIRL approach, which uses the ground-truth.
                    policy_ground_truth_loss = F.mse_loss(policy_actions, expert_actions, reduction='sum')

                    # Compute policy loss.
                    w = self.gt_ratio_p
                    policy_loss = policy_ground_truth_loss * w + policy_adversarial_loss * (1 - w)
                else:
                    # Use the AIRL approach.
                    policy_loss = policy_adversarial_loss

                # Backpropagate the loss and update the policy.
                policy_loss.backward()
                self.policy_optimizer.step()

                # Store the actions and discriminations.
                policy_discriminations.append(torch.exp(log_Ds_policy))
                expert_discriminations.append(torch.exp(log_Ds_expert))

            # Evaluate and store the H-AIRL (and expert) performance so far.
            self.policy.eval()
            self.discriminator.eval()

            with torch.no_grad():
                # Compute policy predictions.
                val_policy_actions, val_policy_log_probs = self.policy.predict(val_states)

                # Compute transitions.
                val_policy_transitions = torch.cat([val_states, val_policy_actions], dim=-1)
                val_expert_transitions = torch.cat([val_states, val_expert_actions], dim=-1)

                # Compute discriminator outputs.
                val_policy_discriminations = torch.exp(
                    self.discriminator.log_D(val_policy_transitions, val_policy_log_probs)
                )
                val_expert_discriminations = torch.exp(
                    self.discriminator.log_D(val_expert_transitions, val_policy_log_probs)
                )

                # Run evaluation.
                eval_performance = self.evaluate(
                    step=step, 
                    discriminator_loss=discriminator_loss, 
                    policy_adversarial_loss=policy_adversarial_loss,
                    policy_ground_truth_loss=policy_ground_truth_loss,
                    policy_actions=policy_actions,
                    expert_actions=expert_actions,
                    val_policy_actions=val_policy_actions,
                    val_expert_actions=val_expert_actions,
                    policy_discriminations=torch.cat(policy_discriminations),
                    expert_discriminations=torch.cat(expert_discriminations),
                    val_policy_discriminations=val_policy_discriminations,
                    val_expert_discriminations=val_expert_discriminations
                )
            
                # Store evaluation rewards.
                policy_eval_rewards.append(eval_performance['policy_reward'])
                expert_eval_rewards.append(eval_performance['expert_reward'])
                policy_eval_distributions.append(eval_performance['policy_distribution'])
                expert_eval_distributions.append(eval_performance['expert_distribution'])
                policy_eval_alignments.append(eval_performance['policy_alignment'])
                discriminator_eval_accuracies.append(eval_performance['discriminator_accuracy'])

                # Save the model.
                if save:
                    self.save()
                elif self.verbose:
                    print('')
                self.env.close()
            
            # measure wall‐clock and CPU‐time since last report
            t1 = time.time()
            c1 = self._proc.cpu_times().user + self._proc.cpu_times().system
            wall = t1 - t0
            cpu_sec = c1 - c0
            cores_used = cpu_sec / wall
            if self.verbose:
                print(f"[Step {step+1:3d}] wall {wall:.2f}s, cpu {cpu_sec:.2f}s → cores {cores_used:.2f}.")
            # reset timers
            t0, c0 = t1, c1

        # Create performance dictionary.
        eval_performance = {
            'policy_rewards': policy_eval_rewards,
            'expert_rewards': expert_eval_rewards,
            'policy_distributions': policy_eval_distributions,
            'expert_distributions': expert_eval_distributions,
            'policy_alignments': policy_eval_alignments,
            'discriminator_accuracies': discriminator_eval_accuracies,
        }
        
        return eval_performance

    def evaluate(self, 
                 step: int, 
                 discriminator_loss: torch.Tensor, 
                 policy_adversarial_loss: torch.Tensor,
                 policy_ground_truth_loss: torch.Tensor,
                 policy_actions: torch.Tensor,
                 expert_actions: torch.Tensor,
                 val_policy_actions: torch.Tensor,
                 val_expert_actions: torch.Tensor,
                 policy_discriminations: torch.Tensor,
                 expert_discriminations: torch.Tensor,
                 val_policy_discriminations: torch.Tensor,
                 val_expert_discriminations: torch.Tensor) -> dict[str, any]:
        """
        Evaluates the current H-AIRL model.

        Arguments
        ---------
        step                       :  The current training step.
        discriminator_loss         :  The loss of the discriminator during training.
        policy_adversarial_loss    :  The adversarial loss of the policy during training.
        policy_ground_truth_loss   :  The ground truth loss of the policy during training.
        policy_actions             :  The policy's actions during training.
        expert_actions             :  The expert's actions during training.
        val_policy_actions         :  The policy's actions during validation.
        val_expert_actions         :  The expert's actions during validation.
        policy_discriminations     :  The discriminations on policy actions during training.
        expert_discriminations     :  The discriminations on expert actions during training.
        val_policy_discriminations :  The discriminations on policy actions during validation.
        val_expert_discriminations :  The discriminations on expert actions during validation.

        Returns
        -------
        eval_performance :  The overall evaluation performance dictionary, which 
                            includes rewards, alignments, and action distributions.

        """

        # Evaluate the policy.
        total_rewards = []
        for _ in range(self.num_eval_runs):
            obs = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                obs_tensor = torch.tensor(obs)
                action, _ = self.policy.act(obs_tensor)
                obs, reward, done, _ = self.env.step(np.array(action))
                total_reward += reward
            total_rewards.append(total_reward)
        avg_policy_reward = np.mean(total_rewards)
        err_policy_reward = np.std(total_rewards)
        err_policy_reward /= np.sqrt(self.num_eval_runs) if self.benchmark_is_poker \
                                                         else 1
        
        # Also evaluate the expert model to compare.
        avg_expert_reward = np.nan
        err_expert_reward = np.nan
        if not self.benchmark_is_poker:
            total_expert_rewards = []
            for _ in range(self.num_eval_runs):
                obs = self.env.reset()
                done = False
                total_expert_reward = 0
                while not done:
                    obs_tensor = torch.tensor(obs)
                    action, _ = self.expert.predict(obs_tensor)
                    obs, reward, done, _ = self.env.step(np.array(action))
                    total_expert_reward += reward
                total_expert_rewards.append(total_expert_reward)
            avg_expert_reward = np.mean(total_expert_rewards)
            err_expert_reward = np.std(total_expert_rewards)

        # Compute action alignment and distribution, if the action space is discrete. 
        policy_alignment = np.nan
        val_policy_alignment = np.nan
        val_policy_distribution = np.nan
        val_expert_distribution = np.nan

        if self.discrete:
            # Compute policy alignment during training.
            policy_indices = torch.argmax(policy_actions, dim=-1)
            expert_indices = torch.argmax(expert_actions, dim=-1)
            policy_alignment = (policy_indices == expert_indices).float().mean().item()

            # Compute policy alignment during validation.
            val_policy_indices = torch.argmax(val_policy_actions, dim=-1)
            val_expert_indices = torch.argmax(val_expert_actions, dim=-1)
            val_policy_alignment = (val_policy_indices == val_expert_indices).float().mean().item()
            
            # Compute policy and expert action distributions during validation.
            val_policy_distribution = torch.sum(val_policy_actions, dim=0) \
                             / torch.sum(val_policy_actions, dim=(0,1))
            val_expert_distribution = torch.sum(val_expert_actions, dim=0) \
                             / torch.sum(val_expert_actions, dim=(0,1))
            val_policy_distribution = val_policy_distribution.numpy().round(2)
            val_expert_distribution = val_expert_distribution.numpy().round(2)
        
        # Compute discriminator accuracy during training.
        discriminator_accuracy_on_policy = (
            policy_discriminations.round() == torch.zeros(policy_discriminations.shape)
        ).float().mean().item()
        discriminator_accuracy_on_expert = (
            expert_discriminations.round() == torch.ones(expert_discriminations.shape)
        ).float().mean().item()
        discriminator_accuracy = (discriminator_accuracy_on_policy \
                               +  discriminator_accuracy_on_expert) / 2

        # Compute discriminator accuracy during validation.
        val_discriminator_accuracy_on_policy = (
            val_policy_discriminations.round() == torch.zeros(val_policy_discriminations.shape)
        ).float().mean().item()
        val_discriminator_accuracy_on_expert = (
            val_expert_discriminations.round() == torch.ones(val_expert_discriminations.shape)
        ).float().mean().item()
        val_discriminator_accuracy = (val_discriminator_accuracy_on_policy \
                                   +  val_discriminator_accuracy_on_expert) / 2

        # Print.
        if self.verbose:
            step_str = f'Step {step + 1}, '
            d_loss_str = f'D tLoss: {discriminator_loss.item():.4f}, '
            p_loss_str = f'G tLoss: {policy_adversarial_loss.item():.4f} + {policy_ground_truth_loss.item():.4f}, '
            p_rew_str = f'G vReward: {avg_policy_reward:.2f} ± {err_policy_reward:.2f}, '
            e_rew_str = f'E vReward: {avg_expert_reward:.2f} ± {err_expert_reward:.2f}. '
            p_v_acc_str = f'G vAcc: {100 * val_policy_alignment:.2f}, ' if self.discrete else ''
            p_t_acc_str = f'G tAcc: {100 * policy_alignment:.2f}, ' if self.discrete else ''
            d_v_acc_str = f'D vAcc: {100 * val_discriminator_accuracy:.2f}, '
            d_t_acc_str = f'D tAcc: {100 * discriminator_accuracy:.2f}, '
            policy_distribution_str = f'G vDist {val_policy_distribution}, ' if self.discrete else ''
            expert_distribution_str = f'E vDist {val_expert_distribution}. ' if self.discrete else ''

            print(step_str, d_loss_str, p_loss_str, p_rew_str, e_rew_str, 
                  p_v_acc_str, p_t_acc_str, d_v_acc_str, d_t_acc_str, 
                  policy_distribution_str, expert_distribution_str, end='')

        # Create the performance dictionary.
        eval_performance = {
            'policy_reward': avg_policy_reward, 
            'expert_reward': avg_expert_reward, 
            'policy_distribution': val_policy_distribution, 
            'expert_distribution': val_expert_distribution,
            'policy_alignment': 100 * val_policy_alignment,
            'discriminator_accuracy': 100 * val_discriminator_accuracy
        }

        return eval_performance
        
    def save(self, path: str | None = None) -> None:
        """
        Saves the entire (Hybrid-)AIRL model object, including its policy and discriminator.

        Arguments
        ---------
        path :  The path where to save the model.

        """
        # Use default path if none is provided
        base_path = f'src/trained/{self.base_dir}{self.algo}/{self.benchmark.lower()}_' if not path else path

        # Make sure the directory exists
        dir_name = os.path.dirname(base_path)
        os.makedirs(dir_name, exist_ok=True)
        
        # Save the entire object using pickle via torch.save
        for model in ['policy', 'discriminator']:
            with open(base_path + f'{model}.model', 'wb') as f:
                if model == 'policy': torch.save(self.policy, f)
                if model == 'discriminator': torch.save(self.discriminator, f)
        if self.verbose:
            print('(model saved)')
