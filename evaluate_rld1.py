# TODO: get state file name from env for each. So can be plotted against RLS1 and GRASP.
# TODO: tidy up this script.
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
import yaml

N = 10  # number of times to try stochasitc solution
deterministic = False 

if __name__ == '__main__':

    training_data_path = Path('../gym-superscript/gym_superscript//training_data/abm_states')
    prior_results = training_data_path / 'prior_optimisation_results/dataset_1_grasp_run_0/results.pickle'
    with open(prior_results, 'rb') as infile:
        prior_results = pickle.load(infile)

    grasp_results = dict(zip(prior_results['state_file'], prior_results['probability']))
    print(grasp_results['state_28.pickle'])

    delta = []
    times = []
    rel = []

    with open('./hyperparams/ppo.yml', 'r') as infile:
        hyperparams = yaml.safe_load(infile)['RLD1-v1.214']

    env = create_test_env(
        env_id="RLD1-v1.214",
        env_kwargs={
            'mode': 'evaluation',
            'render_mode': 'None'
        },
        hyperparams=hyperparams
    )

    best_model = PPO.load(
        './logs/ppo/RLD1-v1.214_10/best_model',
    )

    # for state in grasp_results.keys():
    # all_states = env.previous_results['state_file']
    for si in range(100):

        # state_id = env.evaluation_state_id - 1
        # if state_id == -1:
        #     state_id = 99
        # state =
        # state_name = state.split('.')[0]
        # print(state_name)
        # try:
            start = time.time()
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

                    if len(actions) >= 35:
                        done = True

                if episode_reward > best_score:
                    best_score = episode_reward

            # print("RLS1: ", episode_reward)
            # print("GRASP: ", grasp_results[state])
            grasp_result = env.get_attr('previous_result_success_probability')
            delta.append(
                best_score #- grasp_result
            )
            rel.append(
                best_score[0] #/ grasp_result
            )
            times.append((time.time() - start) / 60)

        # except:
        #     pass

    print(times)
    print(best_score)
    print(np.mean(rel))
    print(np.mean(times))
    plt.scatter(range(len(delta)), delta)
    plt.axhline(1, c='k')
    plt.xlabel('state')
    plt.ylabel('probability_RLD1 / probability_GRASP')
    plt.title("Mean(RLD1/GRASP) = %.2f\n Mean runtime = %.2fs" % (np.mean(rel), 60*np.mean(times)))
    plt.grid()
    plt.show()

    # plt.hist(delta)
    # plt.show()


