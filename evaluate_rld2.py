# TODO: get state file name from env for each. So can be plotted against RLS1 and GRASP.
# TODO: tidy up this script.
from copy import deepcopy

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

N = 100  # number of times to try stochasitc solution
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
    best_prob = []
    grasp = []

    with open('./hyperparams/ppo.yml', 'r') as infile:
        hyperparams = yaml.safe_load(infile)['RLD2-v0']

    env = create_test_env(
        env_id="RLD2-v2.38",
        env_kwargs={
            'mode': 'evaluation',
            'render_mode': 'None'
        },
        hyperparams=hyperparams
    )

    best_model = PPO.load(
        # './logs/ppo/RLD2-v0_5/best_model',
        './logs/ppo/RLD2-v2.38_1/best_model',
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
            # [
            #     e.unwrapped.set_options({'reset_to_new_state': True})
            #     for e in env._get_target_envs(indices=None)
            # ]
            obs = env.reset()
            state_copy = deepcopy([
                e.unwrapped.state
                for e in env._get_target_envs(indices=None)
            ][0])
            for ni in range(N):
                # [
                #     e.unwrapped.set_options({'reset_to_new_state': False})
                #     for e in env._get_target_envs(indices=None)
                # ]
                [
                        e.unwrapped.set_options({'state': state_copy, 'workers': None})
                        for e in env._get_target_envs(indices=None)
                ]
                obs = env.reset()
                ps = [
                    e.unwrapped.previous_result_success_probability
                    for e in env._get_target_envs(indices=None)
                ][0]

                for e in env._get_target_envs(indices=None):
                    np.testing.assert_equal(e.unwrapped.state.state, state_copy.state)

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

                    assert (
                            ps ==
                            grasp_results[list(grasp_results.keys())[infos[0]['Evaluation state ID'] - 1]]
                    )

                if episode_reward > best_score:
                    best_score = episode_reward
                # if ni == 0:
                #     sid = infos[0]['Evaluation state ID']
                # else:
                #     if sid == 99:
                #         assert infos[0]['Evaluation state ID'] == 0
                #     else:
                #         print(ni, sid, infos[0]['Evaluation state ID'])
                #         assert sid == infos[0]['Evaluation state ID'] - 1
                # bs = [
                #     e.unwrapped.current_success_probability
                #     for e in env._get_target_envs(indices=None)
                # ][0]
                # print("BS: ", bs)
                #
                # if bs > best_score:
                #     best_score = bs

            state_file = list(grasp_results.keys())[infos[0]['Evaluation state ID'] - 1]
            # state_file = infos[0]['state_file']
            grasp.append(grasp_results[state_file])
            # print("RLS1: ", episode_reward)
            # print("GRASP: ", grasp_results[state])
            # grasp_result = env.get_attr('previous_result_success_probability')
            # delta.append(
            #     best_score #- grasp_result
            # )
            best_prob.append(
                best_score #/ grasp_result
            )
            rel.append(best_score / grasp_results[state_file])
            times.append((time.time() - start))
            # break
        # except:
        #     pass

    # print(times)
    # print(best_prob)
    # print(grasp)
    print(np.mean(rel))
    print(np.mean(times))

    # plt.plot(grasp, best_prob)
    # plt.hist(rel)
    # plt.show()

    # print(np.mean(rel))

    plt.scatter(range(len(rel)), rel)
    plt.axhline(1, c='k')
    plt.xlabel('state')
    plt.ylabel('probability_RLD2 / probability_GRASP')
    plt.title("Mean(RLD2/GRASP) = %.2f\n Mean runtime = %.2fs" % (np.mean(rel), np.mean(times)))
    plt.grid()
    plt.show()

    plt.scatter(grasp, rel)
    # plt.axhline(1, c='k')
    plt.xlabel('probability_GRASP')
    plt.ylabel('probability_RLD2 / probability_GRASP')
    # plt.title("Mean(RLD2/GRASP) = %.2f\n Mean runtime = %.2fs" % (np.mean(rel), np.mean(times)))
    plt.grid()
    plt.show()
    # plt.hist(delta)
    # plt.show()


