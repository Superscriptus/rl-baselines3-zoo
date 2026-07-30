<!-- SUPERSCRIPT BRANDING -->
<!-- Badges + logo in the style of Superscriptus/SuperScript (othneildrew Best-README-Template). -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Fork license: AGPL-3.0][license-agpl-shield]][license-agpl-url]
[![Upstream license: MIT][license-mit-shield]][license-mit-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/Superscriptus/SuperScript">
    <img src="documentation/images/logo.svg" alt="SuperScript logo" width="390" height="80">
  </a>

  <h3 align="center">SuperScript RL training fork</h3>

  <p align="center">
    Fork of RL Baselines3 Zoo used to train and evaluate the SuperScript RLD1 / RLD2 policies.
  </p>
</p>

---

<!-- MARKDOWN LINKS & IMAGES (SuperScript branding; reference-style) -->
[contributors-shield]: https://img.shields.io/github/contributors/Superscriptus/rl-baselines3-zoo.svg?style=for-the-badge
[contributors-url]: https://github.com/Superscriptus/rl-baselines3-zoo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Superscriptus/rl-baselines3-zoo.svg?style=for-the-badge
[forks-url]: https://github.com/Superscriptus/rl-baselines3-zoo/network/members
[stars-shield]: https://img.shields.io/github/stars/Superscriptus/rl-baselines3-zoo.svg?style=for-the-badge
[stars-url]: https://github.com/Superscriptus/rl-baselines3-zoo/stargazers
[issues-shield]: https://img.shields.io/github/issues/Superscriptus/rl-baselines3-zoo.svg?style=for-the-badge
[issues-url]: https://github.com/Superscriptus/rl-baselines3-zoo/issues
[license-agpl-shield]: https://img.shields.io/badge/Fork%20as%20a%20whole-AGPL--3.0-blue.svg?style=for-the-badge
[license-agpl-url]: LICENSE.AGPL-3.0
[license-mit-shield]: https://img.shields.io/badge/Upstream%20portions-MIT-yellow.svg?style=for-the-badge
[license-mit-url]: LICENSE

# SuperScript fork of RL Baselines3 Zoo

> **This is a fork** of [DLR-RM/rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
> (upstream state ~v2.2.0a4) used to train and evaluate the hierarchical RL
> team-selection policies (RLD1 / RLD2) for **SuperScript**, an agent-based
> model of team allocation in project-based organisations. It is a git
> submodule of [superscript-abm](https://github.com/Superscriptus/superscript-abm);
> see the top-level [OptimizationExperiments](https://github.com/Superscriptus/OptimizationExperiments)
> repository for the article-reproduction instructions.

**What this fork adds/changes vs upstream** (everything else below is the
unmodified upstream README):

- `hyperparams/ppo.yml`: tuned PPO configurations for the SuperScript
  Gymnasium environments (`RLD1-*`, `RLD2-*`, `RLD2_<N>-*`, `RLD2_50-*`,
  and legacy `SSEnv*` IDs), which is how article agents were trained, e.g.
  `python train.py --algo ppo --env RLD2_20-v1.62-nonlinear`.
- `rl_zoo3/import_envs.py`: imports `gym_superscript` so the custom env IDs
  are registered for `train.py` / `enjoy.py`.
- `rl_zoo3/exp_manager.py`: adds an `is_rld2()` helper (RLD2-specific eval
  handling; currently inert — upstream deterministic eval behaviour).
- `rl_zoo3/hyperparams_opt.py`: PPO Optuna search space tweaks (gamma,
  lr_schedule, ortho_init tunable) used for the project's hypertuning runs.
- `evaluate_rld1.py`, `evaluate_rld2.py`, `evaluate_rld2_array.py`,
  `evaluate_rls1_trained_agents.py` (repo root): project evaluation scripts
  comparing trained agents against GRASP-optimised baselines. They assume a
  sibling `../gym-superscript` checkout (the layout produced by a recursive
  clone of superscript-abm) and locally-available trained models under
  `logs/ppo/…` (not committed).
- `additional_requirements.txt`: extra deps used by the project workflow
  (sb3-contrib, optuna, wandb, confuse, pathos, …).

**Requirements not declared here:** `gym_superscript` (and transitively
`superscript_abm`) must be pip-installed in the same environment —
`rl_zoo3/import_envs.py` imports it unconditionally. The pinned environment
used for the article is `article_requirements.txt` in
OptimizationExperiments (Python 3.10, stable-baselines3 2.3.2,
gymnasium 0.29.1).

**License:** this fork **as a whole** is distributed under the GNU Affero
General Public License v3.0 ([LICENSE.AGPL-3.0](LICENSE.AGPL-3.0)). Portions
derived from [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
are Copyright (c) 2019 Antonin Raffin and licensed under the MIT License
([LICENSE](LICENSE), preserved unchanged from upstream); modifications and
additions are Copyright (C) 2025 Michael Christen
&lt;michael.christen@mobi.ch&gt; and licensed under AGPL-3.0. MIT is
compatible with the (A)GPL, permitting this combined licensing. Licensing
will be revisited before journal publication / API deployment — see
[Superscriptus/OptimizationExperiments#94](https://github.com/Superscriptus/OptimizationExperiments/issues/94).

The `rl-trained-agents` submodule pinned in `.gitmodules` is upstream's
collection of pretrained agents (DLR-RM); it is **not needed** for the
SuperScript workflow and can be skipped when initialising submodules.

---

<!-- [![pipeline status](https://gitlab.com/araffin/rl-baselines3-zoo/badges/master/pipeline.svg)](https://gitlab.com/araffin/rl-baselines3-zoo/-/commits/master) -->
![CI](https://github.com/DLR-RM/rl-baselines3-zoo/workflows/CI/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/rl-baselines3-zoo/badge/?version=master)](https://rl-baselines3-zoo.readthedocs.io/en/master/?badge=master)
[![coverage report](https://gitlab.com/araffin/rl-baselines3-zoo/badges/master/coverage.svg)](https://gitlab.com/araffin/rl-baselines3-zoo/-/commits/master) [![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)



# RL Baselines3 Zoo: A Training Framework for Stable Baselines3 Reinforcement Learning Agents

<img src="images/car.jpg" align="right" width="40%"/>

RL Baselines3 Zoo is a training framework for Reinforcement Learning (RL), using [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3).

It provides scripts for training, evaluating agents, tuning hyperparameters, plotting results and recording videos.

In addition, it includes a collection of tuned hyperparameters for common environments and RL algorithms, and agents trained with those settings.


We are **looking for contributors** to complete the collection!

Goals of this repository:

1. Provide a simple interface to train and enjoy RL agents
2. Benchmark the different Reinforcement Learning algorithms
3. Provide tuned hyperparameters for each environment and RL algorithm
4. Have fun with the trained agents!

This is the SB3 version of the original SB2 [rl-zoo](https://github.com/araffin/rl-baselines-zoo).

## Documentation

Documentation is available online: [https://rl-baselines3-zoo.readthedocs.io/](https://rl-baselines3-zoo.readthedocs.io)

## Installation

### Minimal installation

From source:
```
pip install -e .
```

As a python package:
```
pip install rl_zoo3
```

Note: you can do `python -m rl_zoo3.train` from any folder and you have access to `rl_zoo3` command line interface, for instance, `rl_zoo3 train` is equivalent to `python train.py`

### Full installation (with extra envs and test dependencies)

```
apt-get install swig cmake ffmpeg
pip install -r requirements.txt
```

Please see [Stable Baselines3 documentation](https://stable-baselines3.readthedocs.io/en/master/) for alternatives to install stable baselines3.

## Train an Agent

The hyperparameters for each environment are defined in `hyperparameters/algo_name.yml`.

If the environment exists in this file, then you can train an agent using:
```
python train.py --algo algo_name --env env_id
```

Evaluate the agent every 10000 steps using 10 episodes for evaluation (using only one evaluation env):
```
python train.py --algo sac --env HalfCheetahBulletEnv-v0 --eval-freq 10000 --eval-episodes 10 --n-eval-envs 1
```

More examples are available in the [documentation](https://rl-baselines3-zoo.readthedocs.io).


## Integrations

The RL Zoo has some integration with other libraries/services like Weights & Biases for experiment tracking or Hugging Face for storing/sharing trained models. You can find out more in the [dedicated section](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/integrations.html) of the documentation.

## Plot Scripts

Please the see [dedicated section](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/plot.html) of the documentation.

## Enjoy a Trained Agent

**Note: to download the repo with the trained agents, you must use `git clone --recursive https://github.com/DLR-RM/rl-baselines3-zoo`** in order to clone the submodule too.


If the trained agent exists, then you can see it in action using:
```
python enjoy.py --algo algo_name --env env_id
```

For example, enjoy A2C on Breakout during 5000 timesteps:
```
python enjoy.py --algo a2c --env BreakoutNoFrameskip-v4 --folder rl-trained-agents/ -n 5000
```

## Hyperparameters Tuning

Please the see [dedicated section](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/tuning.html) of the documentation.

## Custom Configuration

Please the see [dedicated section](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/config.html) of the documentation.

## Current Collection: 200+ Trained Agents!

Final performance of the trained agents can be found in [`benchmark.md`](./benchmark.md). To compute them, simply run `python -m rl_zoo3.benchmark`.

List and videos of trained agents can be found on our Huggingface page: https://huggingface.co/sb3

*NOTE: this is not a quantitative benchmark as it corresponds to only one run (cf [issue #38](https://github.com/araffin/rl-baselines-zoo/issues/38)). This benchmark is meant to check algorithm (maximal) performance, find potential bugs and also allow users to have access to pretrained agents.*

### Atari Games

7 atari games from OpenAI benchmark (NoFrameskip-v4 versions).

|  RL Algo |  BeamRider         | Breakout           | Enduro             |  Pong | Qbert | Seaquest           | SpaceInvaders      |
|----------|--------------------|--------------------|--------------------|-------|-------|--------------------|--------------------|
| A2C      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| PPO      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| DQN      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| QR-DQN   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

Additional Atari Games (to be completed):

|  RL Algo |  MsPacman   | Asteroids | RoadRunner |
|----------|-------------|-----------|------------|
| A2C      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| PPO      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| DQN      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| QR-DQN   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |


### Classic Control Environments

|  RL Algo |  CartPole-v1 | MountainCar-v0 | Acrobot-v1 | Pendulum-v1 | MountainCarContinuous-v0 |
|----------|--------------|----------------|------------|--------------------|--------------------------|
| ARS      | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| A2C      | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| PPO      | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| DQN      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | N/A                | N/A |
| QR-DQN   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | N/A                | N/A |
| DDPG     |  N/A |  N/A  | N/A | :heavy_check_mark: | :heavy_check_mark: |
| SAC      |  N/A |  N/A  | N/A | :heavy_check_mark: | :heavy_check_mark: |
| TD3      |  N/A |  N/A  | N/A | :heavy_check_mark: | :heavy_check_mark: |
| TQC      |  N/A |  N/A  | N/A | :heavy_check_mark: | :heavy_check_mark: |
| TRPO     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |


### Box2D Environments

|  RL Algo |  BipedalWalker-v3 | LunarLander-v2 | LunarLanderContinuous-v2 |  BipedalWalkerHardcore-v3 | CarRacing-v0 |
|----------|--------------|----------------|------------|--------------|--------------------------|
| ARS      |  | :heavy_check_mark: | | :heavy_check_mark: | |
| A2C      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| PPO      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| DQN      | N/A | :heavy_check_mark: | N/A | N/A | N/A |
| QR-DQN   | N/A | :heavy_check_mark: | N/A | N/A | N/A |
| DDPG     | :heavy_check_mark: | N/A | :heavy_check_mark: | | |
| SAC      | :heavy_check_mark: | N/A | :heavy_check_mark: | :heavy_check_mark: | |
| TD3      | :heavy_check_mark: | N/A | :heavy_check_mark: | :heavy_check_mark: | |
| TQC      | :heavy_check_mark: | N/A | :heavy_check_mark: | :heavy_check_mark: | |
| TRPO     | | :heavy_check_mark: | :heavy_check_mark: | | |

### PyBullet Environments

See https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs.
Similar to [MuJoCo Envs](https://gym.openai.com/envs/#mujoco) but with a ~free~ (MuJoCo 2.1.0+ is now free!) easy to install simulator: pybullet. We are using `BulletEnv-v0` version.

Note: those environments are derived from [Roboschool](https://github.com/openai/roboschool) and are harder than the Mujoco version (see [Pybullet issue](https://github.com/bulletphysics/bullet3/issues/1718#issuecomment-393198883))

|  RL Algo |  Walker2D | HalfCheetah | Ant | Reacher |  Hopper | Humanoid |
|----------|-----------|-------------|-----|---------|---------|----------|
| ARS      |  |  |  |  |  | |
| A2C      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| PPO      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| DDPG     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| SAC      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| TD3      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| TQC      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| TRPO     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |

PyBullet Envs (Continued)

|  RL Algo |  Minitaur | MinitaurDuck | InvertedDoublePendulum | InvertedPendulumSwingup |
|----------|-----------|-------------|-----|---------|
| A2C      | | | | |
| PPO      | | | | |
| DDPG     | | | | |
| SAC      | | | | |
| TD3      | | | | |
| TQC      | | | | |

### MuJoCo Environments

|  RL Algo |  Walker2d | HalfCheetah | Ant | Swimmer |  Hopper | Humanoid |
|----------|-----------|-------------|-----|---------|---------|----------|
| ARS      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |  |
| A2C      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| PPO      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| DDPG     |  |  |  |  |  | |
| SAC      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| TD3      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| TQC      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| TRPO      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  |

### Robotics Environments

See https://gym.openai.com/envs/#robotics and https://github.com/DLR-RM/rl-baselines3-zoo/pull/71

MuJoCo version: 1.50.1.0
Gym version: 0.18.0

We used the v1 environments.

|  RL Algo |  FetchReach | FetchPickAndPlace | FetchPush | FetchSlide |
|----------|-------------|-------------------|-----------|------------|
| HER+TQC  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |


### Panda robot Environments

See https://github.com/qgallouedec/panda-gym/.

Similar to [MuJoCo Robotics Envs](https://gym.openai.com/envs/#robotics) but with a ~free~ easy to install simulator: pybullet.

We used the v1 environments.

|  RL Algo |  PandaReach | PandaPickAndPlace | PandaPush | PandaSlide | PandaStack |
|----------|-------------|-------------------|-----------|------------|------------|
| HER+TQC | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |


### MiniGrid Envs

See https://github.com/Farama-Foundation/Minigrid.
A simple, lightweight and fast Gym environments implementation of the famous gridworld.

| RL Algo | Empty-Random-5x5   | FourRooms          | DoorKey-5x5        | MultiRoom-N4-S5    | Fetch-5x5-N2       | GoToDoor-5x5       | PutNear-6x6-N2     | RedBlueDoors-6x6   | LockedRoom         | KeyCorridorS3R1    | Unlock             | ObstructedMaze-2Dlh |
| ------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------- |
| A2C     |                    |                    |                    |                    |                    |                    |                    |                    |                    |                    |                    |                     |
| PPO     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  |
| DQN     |                    |                    |                    |                    |                    |                    |                    |                    |                    |                    |                    |                     |
| QR-DQN  |                    |                    |                    |                    |                    |                    |                    |                    |                    |                    |                    |                     |
| TRPO    |                    |                    |                    |                    |                    |                    |                    |                    |                    |                    |                    |                     |

There are 22 environment groups (variations for each) in total.


## Colab Notebook: Try it Online!

You can train agents online using [Colab notebook](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/rl-baselines-zoo.ipynb).

### Passing arguments in an interactive session

The zoo is not meant to be executed from an interactive session (e.g: Jupyter Notebooks, IPython), however, it can be done by modifying `sys.argv` and adding the desired arguments.

*Example*
```python
import sys
from rl_zoo3.train import train

sys.argv = ["python", "--algo", "ppo", "--env", "MountainCar-v0"]

train()
```


## Tests

To run tests, first install pytest, then:
```
make pytest
```

Same for type checking with pytype:
```
make type
```


## Citing the Project

To cite this repository in publications:

```bibtex
@misc{rl-zoo3,
  author = {Raffin, Antonin},
  title = {RL Baselines3 Zoo},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DLR-RM/rl-baselines3-zoo}},
}
```

## Contributing

If you trained an agent that is not present in the RL Zoo, please submit a Pull Request (containing the hyperparameters and the score too).

## Contributors

We would like to thank our contributors: [@iandanforth](https://github.com/iandanforth), [@tatsubori](https://github.com/tatsubori) [@Shade5](https://github.com/Shade5) [@mcres](https://github.com/mcres), [@ernestum](https://github.com/ernestum), [@qgallouedec](https://github.com/qgallouedec)
