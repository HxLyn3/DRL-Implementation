import gym
import time
import copy

import numpy as np
import matplotlib.pyplot as plt

from dqn import DQN

def main(env, agent, update_freq=4, target_update_freq=100, n_episodes=1000):
    episode_loss_list   = []        # loss of DQN at each episode
    episode_reward_list = []        # reward at each episode

    cnt = 0                         # counter of decision making
    for it in range(n_episodes):
        obs = env.reset()
        loss_list   = []
        reward_list = []

        while True:
            env.render()
            action = agent.choose_action(obs)                           # agent makes decision
            obs_, reward, done, info = env.step(action)                 # environmental feedback
            agent.store_experience(obs, action, reward, done, obs_)     # agent stores exp
            cnt += 1

            # update parameters
            if cnt%update_freq == 0: loss_list.append(agent.train())
            if cnt%target_update_freq == 0: agent.update_target()

            reward_list.append(reward)
            obs = copy.deepcopy(obs_)
            if done: break

        episode_reward_list.append(np.sum(reward_list))
        episode_loss_list.append(np.mean(loss_list))
        tmp = int(it/n_episodes*100)
        print("\r%d%%|"%tmp+"="*int(tmp/2)+">"+" "*(50-int(tmp/2))+"|%d/%d.   episode reward: %.4f  loss: %.4f"\
            %(it, n_episodes, episode_reward_list[-1], episode_loss_list[-1]), end='')

    agent.save("./dqn_model.pkl")

    # plot loss
    figsize = 8, 6
    figure, ax = plt.subplots(figsize=figsize)
    plt.plot(episode_loss_list)
    plt.xlabel("episode")
    plt.ylabel("loss")
    plt.savefig("./loss.png", dpi=500)

    # plot episode reward
    figure, ax = plt.subplots(figsize=figsize)
    plt.plot(episode_reward_list)
    plt.xlabel("episode")
    plt.ylabel("episode reward")
    plt.savefig("./reward.png", dpi=500)

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
        epsilon_initial  = 0.5,             # initial ε in ε-greedy exploration
        epsilon_final    = 0.01             # final value of ε
    )

    main(env, agent, update_freq=1, target_update_freq=200, n_episodes=5000)
    env.close()