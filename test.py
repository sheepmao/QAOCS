import os
import torch
import numpy as np
from env import ABREnv
from PPO import PPO
from core import S_INFO, S_LEN, A_DIM

def test(checkpoint_path):
    ####### initialize environment hyperparameters ######
    env_name = "ABREnv"

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 300  # max timesteps in one episode
    test_episodes = 10  # number of episodes to test

    random_seed = 0  # set random seed if required (0 = no random seed)
    #####################################################

    env = ABREnv()

    # state space dimension
    state_dim = S_LEN * S_INFO

    # action space dimension
    action_dim = A_DIM

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, 0, 0, 0, 0, 0, has_continuous_action_space, 0)

    # load the trained model checkpoint
    checkpoint = torch.load(checkpoint_path)
    ppo_agent.policy.load_state_dict(checkpoint)

    print("--------------------------------------------------------------------------------------------")
    test_running_reward = 0

    for ep in range(1, test_episodes + 1):
        ep_reward = 0
        state = env.reset()
        state = state.flatten()

        for t in range(1, max_ep_len + 1):
            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            state = state.flatten()
            ep_reward += reward

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()
        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("--------------------------------------------------------------------------------------------")
    print("Test Episodes: {} \t\t Average Test Reward: {}".format(test_episodes, round(test_running_reward / test_episodes, 2)))
    print("============================================================================================")


if __name__ == '__main__':
    checkpoint_path = "PPO_preTrained/ABREnv/PPO_ABREnv_0_0.pth"  # replace with your trained model checkpoint path
    test(checkpoint_path)