import gym
import time
import copy
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from dqn import DQN

def main(env, agent, n_episodes=1000):
    episode_reward_list = []        # reward at each episode
    test_frames         = []        # frames of test process in env

    cnt = 0                 # counter of decision making
    for it in tqdm(range(n_episodes)):
        obs = env.reset()
        reward_list = []

        while True:
            test_frames.append(env.render(mode='rgb_array'))
            action = agent.choose_action(obs, exploration=False)        # agent makes decision
            obs_, reward, done, info = env.step(action)                 # environmental feedback
            agent.store_experience(obs, action, reward, done, obs_)     # agent stores exp

            reward_list.append(reward)
            obs = copy.deepcopy(obs_)
            if done: break

        episode_reward_list.append(np.sum(reward_list))

    print("Episode x %d\taverage episode reward: %.4f"%(n_episodes, np.mean(episode_reward_list)))

    patch = plt.imshow(test_frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(test_frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(test_frames), interval=5)
    anim.save('./test.gif', writer='imagemagick', fps=30)

if __name__ == '__main__':
    # environment
    env = gym.make('MountainCar-v0')
    env.reset()

    obs_dim = env.observation_space.shape[0]
    hid_dim = 64
    n_actions = env.action_space.n

    # agent
    agent = DQN(
        obs_dim          = obs_dim,         # dimension of observation
        hid_dim          = hid_dim,         # dimension of hidden layer
        n_actions        = n_actions,       # card of action space
        replay_buffer_sz = 3000,            # size of experience replay buffer
        replay_start_sz  = 500,             # size threshold to start training
        batch_sz         = 32,              # mini-batch size
        gamma            = 0.99,            # discount factor
        lr               = 0.0005,          # learning rate
        epsilon_initial  = 0.01,            # initial ε in ε-greedy exploration
        epsilon_final    = 0.01             # final value of ε
    )
    agent.load("./dqn_model.pkl")

    main(env, agent, n_episodes=100)
    env.close()