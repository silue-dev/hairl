# Hybrid Adversarial Inverse RL

This repository provides the implementation of H-AIRL (Hybrid Adversarial Inverse Reinforcement Learning).

* 📄 **Paper:** Published at ESANN 2026.
* 🔖 **DOI:** https://doi.org/10.14428/esann/2026.ES2026-114.
* 🔗 **arXiv:** https://arxiv.org/abs/2511.21356.

We benchmark H-AIRL against AIRL ([Fu et al., 2017](https://arxiv.org/abs/1710.11248)) on multiple [Gymnasium](https://gymnasium.farama.org) environments and on Limit Hold'em Poker using [RLCard](https://rlcard.org).

The poker data comprises 1v1 games from the [IRC Poker dataset](http://poker.cs.ualberta.ca/irc_poker_database.html).

## Getting Started

### 1. Create and activate the Conda environment  

```bash
conda create -n hairl python=3.11
conda activate hairl
pip install -r requirements.txt
```

### 2. Run experiments (IRL + RL)

```bash
python src/main.py
```

You can also specify benchmarks or skip components using flags, e.g.:

```bash
python src/main.py --benchmark LunarLander-v2 --skip-rl
```


### 3. Plot results

```bash
python src/utils/plot_main.py
```
