import copy
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQN(object):
    """ a implement of Deep Q-Network """

    def __init__(self,
                 obs_dim,
                 hid_dim,
                 n_actions,
                 replay_buffer_sz=1000000,
                 replay_start_sz=50000,
                 batch_sz=32,
                 gamma=0.99,
                 lr=0.0001,
                 epsilon_initial=1,
                 epsilon_decay=0.999,
                 epsilon_final=0.1):
        """ init Deep Q-Network """

        self.obs_dim = obs_dim                                  # dimension of observation
        self.hid_dim = hid_dim                                  # dimension of hidden layer
        self.n_actions = n_actions                              # card of action space
        self.main_q_net = nn.Sequential(                        # main Q-Network
            nn.Linear(self.obs_dim, self.hid_dim), nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim), nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim), nn.ReLU(),
            nn.Linear(self.hid_dim, self.n_actions)
        ).to(device)
        self.target_q_net = copy.deepcopy(                      # target Q-Network, the same architecture as main Q-Net
            self.main_q_net).to(device)

        self.batch_sz = batch_sz                                # mini-batch size
        self.replay_buffer_sz = replay_buffer_sz                # size of experience replay buffer
        self.replay_start_sz = replay_start_sz                  # size threshold to start training
        self.replay_buffer = np.empty(                          # empty experience replay buffer, e = <s, a, r, done, s'>
            shape=(self.replay_buffer_sz, 2*self.obs_dim+3))
        self.exp_cnt = 0                                        # counter of existing exp in replay buffer

        self.lr = lr                                            # learning rate
        self.optimizer = Adam(                                  # Adam optimizer
            self.main_q_net.parameters(), lr=self.lr)

        self.gamma = gamma                                      # discount factor
        self.epsilon = epsilon_initial                          # initial ε in ε-greedy exploration
        self.epsilon_decay = epsilon_decay                      # decay factor of ε
        self.epsilon_min = epsilon_final                        # final value of ε

    def choose_action(self, obs, exploration=True):
        """ choose action """
        obs = torch.FloatTensor(obs).to(device)
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
        if exploration and np.random.uniform(0, 1) < self.epsilon:
            action = np.random.randint(self.n_actions)
        else: 
            action = np.argmax(self.main_q_net(obs).detach().cpu().numpy())
        return action

    def store_experience(self, obs, action, reward, done, obs_next):
        """ store experience into replay buffer """
        exp = np.concatenate([obs, [action], [reward], [done], obs_next])
        self.replay_buffer[self.exp_cnt%self.replay_buffer_sz, :] = exp
        self.exp_cnt += 1

    def train(self):
        """ train main DQN with a mini-batch of exp """
        batch_indices = np.random.choice(min(self.exp_cnt, self.replay_buffer_sz), self.batch_sz)
        batch_exp = self.replay_buffer[batch_indices, :]
        s    = torch.FloatTensor(batch_exp[:, :self.obs_dim]).to(device)                  # state
        a    = torch.LongTensor(batch_exp[:, self.obs_dim]).to(device)                    # action
        r    = torch.FloatTensor(batch_exp[:, self.obs_dim+1]).to(device)                 # reward
        done = torch.LongTensor(batch_exp[:, self.obs_dim+2]).to(device)                  # if done
        s_   = torch.FloatTensor(batch_exp[:, -self.obs_dim:]).to(device)                 # next state

        q_values  = self.main_q_net(s)[range(self.batch_sz), a]
        q_targets = r + self.gamma*(1-done)*torch.max(self.target_q_net(s_), dim=1)[0]
        loss = ((q_values-q_targets)**2).mean()

        # gradient diecent
        if self.exp_cnt > self.replay_start_sz:
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.main_q_net.parameters(), 0.5)
            self.optimizer.step()
        return float(loss)

    def update_target(self):
        """ update target Q-Network by copying parameters from main Q-Network"""
        for param, target_param in zip(self.main_q_net.parameters(), self.target_q_net.parameters()):
            target_param.data.copy_(param.data)

    def save(self, filename):
        """ save model """
        self.main_q_net.cpu()
        torch.save(self.main_q_net.state_dict(), filename)

    def load(self, filename):
        """ load model """
        self.main_q_net.load_state_dict(torch.load(filename))
        self.main_q_net.eval()
