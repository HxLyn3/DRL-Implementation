import gym
import time
import copy
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from TD3 import TD3

def main(env, agent, n_episodes=1000, render=True):
    episode_reward_list = []            # reward at each episode
    test_frames         = []            # frames of test process in env

    pbar = tqdm(range(n_episodes))
    for it in pbar:
        obs = env.reset()
        reward_list      = []

        while True:
            if render: test_frames.append(env.render(mode='rgb_array'))
            action = agent.policy(obs, exploration=False)               # agent makes decision
            obs_, reward, done, info = env.step(action)                 # environmental feedback

            reward_list.append(reward)
            obs = copy.deepcopy(obs_)
            if done: break

        episode_reward_list.append(np.sum(reward_list))
        pbar.set_postfix(episode_reward='%.2f'%episode_reward_list[-1])
        pbar.update()

    print("Episode x %d\taverage episode reward: %.4f"%(n_episodes, np.mean(episode_reward_list)))

    # patch = plt.imshow(test_frames[0])
    # plt.axis('off')
    # def animate(i):
    #     patch.set_data(test_frames[i])
    # anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(test_frames), interval=5)
    # anim.save('./figs/test.gif', writer='PillowWriter', fps=30)

if __name__ == '__main__':
    # environment
    env = gym.make('BipedalWalkerHardcore-v3')
    env.reset()

    obs_dim = env.observation_space.shape[0]
    hid_dim = 256
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # agent
    agent = TD3(
        obs_dim             = obs_dim,         # dimension of observation
        hid_dim             = hid_dim,         # dimension of hidden layer
        act_dim             = act_dim,         # dimension of action
        max_action          = max_action,      # max absolute value of action
        replay_buffer_sz    = 1000000,         # size of experience replay buffer
        replay_start_sz     = 2000,            # size threshold to start training
        batch_sz            = 256,             # mini-batch size
        gamma               = 0.99,            # discount factor
        actor_lr            = 0.0001,          # learning rate of actor
        critic_lr           = 0.0001,          # learning rate of critic
        delay_freq          = 2,               # delay frequency of actor
        tau                 = 0.005,           # soft update rate for target network
        explore_noise       = 0.25,            # std of Gaussian noise for exploration
        target_policy_noise = 0.2,             # std of noise added to the action selected by target policy
        noise_clip          = 0.5,             # clip of target policy noise
        device              = "cpu"            # cpu/gpu
    )
    agent.load("./td3_model.pkl")

    main(env, agent, n_episodes=100, render=False)
    env.close()
