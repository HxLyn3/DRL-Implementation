import gym
import time
import copy
from tqdm import tqdm

import torch
import numpy as np
import matplotlib.pyplot as plt

from ddpg import DDPG

def main(env, agent, update_freq=4, max_episode_len=2000, n_episodes=10000, render=True):
    episode_actor_loss_list  = []           # loss of DDPG'actor at each episode
    episode_critic_loss_list = []           # loss of DDPG'critic at each episode
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
            agent.store_experience(obs, action, reward, done, obs_)     # agent stores exp
            cnt += 1

            # update parameters
            if cnt%update_freq == 0:
                actor_loss, critic_loss = agent.update()
                actor_loss_list.append(actor_loss)
                critic_loss_list.append(critic_loss)

            reward_list.append(reward)
            obs = copy.deepcopy(obs_)
            if done: break

        episode_reward_list.append(np.sum(reward_list))
        episode_actor_loss_list.append(np.mean(actor_loss_list))
        episode_critic_loss_list.append(np.mean(critic_loss_list))
        pbar.set_postfix(episode_reward='%.2f'%episode_reward_list[-1],
                         actor_loss='%.2f'%episode_actor_loss_list[-1],
                         critic_loss='%.2f'%episode_critic_loss_list[-1])
        pbar.update()

    agent.save("./ddpg_model.pkl")

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

if __name__ == '__main__':
    # environment
    env = gym.make('Pendulum-v0')
    env.reset()

    obs_dim = env.observation_space.shape[0]
    hid_dim = 64
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # agent
    agent = DDPG(
        obs_dim          = obs_dim,         # dimension of observation
        hid_dim          = hid_dim,         # dimension of hidden layer
        act_dim          = act_dim,         # dimension of action
        max_action       = max_action,      # max absolute value of action
        replay_buffer_sz = 10000,           # size of experience replay buffer
        replay_start_sz  = 200,             # size threshold to start training
        batch_sz         = 32,              # mini-batch size
        gamma            = 0.99,            # discount factor
        lr               = 0.0005,          # learning rate
        tau              = 0.005,           # soft update rate for target network
        explore_noise    = 0,               # std of Gaussian noise for exploration
        device           = device           # cpu/gpu
    )

    main(env, agent, update_freq=1, max_episode_len=100, n_episodes=1000, render=True)
    env.close()
