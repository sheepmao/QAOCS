import os
from datetime import datetime
import torch
import numpy as np
from env import ABREnv
from PPO import PPO
from env import S_INFO, S_LEN, A_DIM
import matplotlib.pyplot as plt


################################### Training ###################################
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "ABREnv"

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 300  # max timesteps in one episode
    max_training_timesteps = int(3e6)  # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e4)  # save model frequency (in num timesteps)

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

    random_seed = 0  # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    env = ABREnv()

    # state space dimension
    state_dim = S_LEN * S_INFO

    # action space dimension
    action_dim = A_DIM


    ################### checkpointing ###################
    run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    directory = directory + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)
    ppo_agent.load(checkpoint_path)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    time_step = 0


    state = env.reset()
    state = state.flatten()
    current_ep_reward = 0


    for t in range(1, max_ep_len + 1):
        print('timestep:', time_step)

        # select action with policy
        action = ppo_agent.select_action(state)
        state, reward, done, _ = env.step(action)
        print('reward:', reward)
        state = state.flatten()

        # saving reward and is_terminals
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)

        time_step += 1
        current_ep_reward += reward

        # update PPO agent
        if time_step % update_timestep == 0:
            ppo_agent.update()

        # if continuous action space; then decay action std of ouput action distribution
        if has_continuous_action_space and time_step % action_std_decay_freq == 0:
            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)



    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()
