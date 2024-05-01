import os
import torch
import numpy as np
from env import ABREnv
from PPO import PPO
from core import S_INFO, S_LEN, A_DIM
from torch.utils.tensorboard import SummaryWriter
def test(checkpoint_path):
    ####### initialize environment hyperparameters ######
    env_name = "ABREnv"

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 300  # max timesteps in one episode
    test_episodes = 10  # number of episodes to test

    random_seed = 0  # set random seed if required (0 = no random seed)
    #####################################################
    test_trace_folder = './test/'
    video_path = 'bigbuckbunny360p24.mp4'
    random_seed = 0
    writer = SummaryWriter()
    #####################################################
    env = ABREnv(trace_folder=test_trace_folder, video_path=video_path,\
                  random_seed=random_seed,writer=writer, test =True) 

    # state space dimension
    state_dim = S_LEN * S_INFO

    # action space dimension
    action_dim = A_DIM
    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4  # update policy every n timesteps
    K_epochs = 80  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor
    
    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, 0, 0, 0, 0, 0, has_continuous_action_space, 0)
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)
    # load the trained model checkpoint

    if os.path.exists(checkpoint_path):
        ppo_agent.load(checkpoint_path)
    # checkpoint = torch.load(checkpoint_path)
    # ppo_agent.policy.load_state_dict(checkpoint)

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