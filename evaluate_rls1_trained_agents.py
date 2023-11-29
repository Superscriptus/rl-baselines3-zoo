# TODO: add runtime/training time saving to train_all_rls1.py (and any other training scripts).
# TODO: add saving and analysis of other solution features to this script (e.g. prob cpts, other state variables?)
# TODO: check obs normalisation warning.
# TODO: Try enjoy with stochastic policy for N=1000 and take best...

from gym_superscript.envs import SSEnvAllocateHardSkillsTest
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_zoo3.utils import create_test_env

from pathlib import Path
from superscript_abm.optimisation_decoupled import Solution
import random
import matplotlib.pyplot as plt
import pickle
import time
import numpy as np

N = 1  # number of times to try stochasitc solution
deterministic = True

if __name__ == '__main__':

    training_data_path = Path('../gym-superscript/gym_superscript//training_data/abm_states')
    prior_results = training_data_path / 'prior_optimisation_results/dataset_1_grasp_run_0/results.pickle'
    with open(prior_results, 'rb') as infile:
        prior_results = pickle.load(infile)

    grasp_results = dict(zip(prior_results['state_file'], prior_results['probability']))
    print(grasp_results['state_28.pickle'])

    delta = []
    times = []

    for state in grasp_results.keys():

        state_name = state.split('.')[0]
        print(state_name)
        try:
            start = time.time()
            env = create_test_env(
                env_id="SSEnvAllocateHardSkillsTest-%s" % state_name,
                hyperparams={
                            'n_envs': 2,
                            'n_timesteps': 400000,
                            'policy': 'MlpPolicy',
                            'env_wrapper': 'gymnasium.wrappers.FlattenObservation',
                            'gamma': 0.9
                    },
                env_kwargs={
                    'render_mode': 'None',
                    'env_config_flags': {
                        'produce_failed_state_flag': False,
                        'normalise_observation_state': False,
                        'include_previous_probability_in_observations': False,
                        'include_current_probability_in_observations': False,
                        'include_fail_flags_in_observations': False,
                        'reward_range': [0, 1]
                    }
                }
            )

            best_model = PPO.load(
                './logs/ppo/rls1_gpu_normalised_no_wandb/ppo/SSEnvAllocateHardSkillsTest-%s_1/best_model' % state_name,
            )

            best_score = 0
            for ni in range(N):

                obs = env.reset()
                done = False
                actions = []
                episode_reward = 0
                while not done:
                    action, lstm_states = best_model.predict(
                        obs,
                        deterministic=deterministic,
                    )
                    obs, reward, done, infos = env.step(action)

                    actions.append(action)
                    episode_reward += reward

                    if len(actions) > 35:
                        done = True

                if episode_reward > best_score:
                    best_score = episode_reward

            # print("RLS1: ", episode_reward)
            # print("GRASP: ", grasp_results[state])

            delta.append(
                best_score - grasp_results[state]
            )
            times.append((time.time() - start) / 60)

        except:
            pass

    print(times)
    plt.scatter(range(len(delta)), delta)
    plt.axhline(0)
    plt.xlabel('state')
    plt.ylabel('probability_RLS1 - probability_GRASP')
    plt.grid()
    plt.show()

    plt.hist(delta)
    plt.show()


