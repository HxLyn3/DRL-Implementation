import gym
import time
import copy
from tqdm import tqdm

import torch
import numpy as np
import matplotlib.pyplot as plt

from TD3 import TD3

def main(env, agent, update_freq=4, max_episode_len=2000, n_episodes=10000, render=True):
    episode_actor_loss_list  = []           # loss of TD3'actor at each episode
    episode_critic_loss_list = []           # loss of TD3'critic at each episode
    episode_reward_list      = []           # reward at each episode

    pbar = tqdm(range(n_episodes))
    for it in pbar:
        # init
        obs = env.reset()
        actor_loss_list  = []
        critic_loss_list = []
        reward_list      = []
        # reset noise
        agent.reset_noise(ratio=(1-it/n_episodes))

        cnt = 0                             # counter of decision making
        while cnt < max_episode_len:
            if render: env.render()
            action = agent.policy(obs)                                  # agent makes decision
            obs_, reward, done, info = env.step(action)                 # environmental feedback
            dead = True if reward == -100 else False                    # whether agent is dead or not
            agent.store_experience(obs, action, reward, dead, obs_)     # agent stores exp
            cnt += 1

            # update parameters
            if cnt%update_freq == 0: 
                actor_loss, critic_loss = agent.update()
                actor_loss_list.append(actor_loss)
                critic_loss_list.append(critic_loss)

            reward_list.append(reward)
            obs = copy.deepcopy(obs_)
            if done: break

        if it%1000 == 0: agent.save("./td3_model.pkl")
        episode_reward_list.append(np.sum(reward_list))
        episode_actor_loss_list.append(np.mean(actor_loss_list))
        episode_critic_loss_list.append(np.mean(critic_loss_list))
        pbar.set_postfix(episode_reward='%.2f'%episode_reward_list[-1],
                         actor_loss='%.2f'%episode_actor_loss_list[-1],
                         critic_loss='%.2f'%episode_critic_loss_list[-1])
        pbar.update()

    return episode_actor_loss_list, episode_critic_loss_list, episode_reward_list

if __name__ == '__main__':
    # environment
    env = gym.make('BipedalWalkerHardcore-v3')
    env.reset()

    obs_dim = env.observation_space.shape[0]
    hid_dim = 256
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # agent
    agent = TD3(
        obs_dim             = obs_dim,         # dimension of observation
        hid_dim             = hid_dim,         # dimension of hidden layer
        act_dim             = act_dim,         # dimension of action
        max_action          = max_action,      # max absolute value of action
        replay_buffer_sz    = 1000000,         # size of experience replay buffer
        replay_start_sz     = 2000,            # size threshold to start training
        batch_sz            = 64,              # mini-batch size
        gamma               = 0.99,            # discount factor
        actor_lr            = 0.0001,          # learning rate of actor
        critic_lr           = 0.0001,          # learning rate of critic
        delay_freq          = 2,               # delay frequency of actor
        tau                 = 0.005,           # soft update rate for target network
        explore_noise       = 0.25,            # std of Gaussian noise for exploration
        target_policy_noise = 0.2,             # std of noise added to the action selected by target policy
        noise_clip          = 0.5,             # clip of target policy noise
        device              = device           # cpu/gpu
    )

    episode_actor_loss_list, episode_critic_loss_list, episode_reward_list = \
        main(env, agent, update_freq=1, max_episode_len=800, n_episodes=10000, render=False)
    env.close()

    # plot loss
    figsize = 8, 6
    figure, ax = plt.subplots(figsize=figsize)
    ax.plot(episode_actor_loss_list, label="actor")
    ax.plot(episode_critic_loss_list, label="critic")
    plt.xlabel("episode")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("./figs/loss.png", dpi=500)

    # plot episode reward
    figure, ax = plt.subplots(figsize=figsize)
    ax.plot(episode_reward_list)
    plt.xlabel("episode")
    plt.ylabel("episode reward")
    plt.savefig("./figs/reward.png", dpi=500)
