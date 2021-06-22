import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class Actor(nn.Module):
    """ Actor Network """

    def __init__(self, obs_dim, hid_dim, act_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(obs_dim, hid_dim)
        self.layer2 = nn.Linear(hid_dim, hid_dim)
        self.layer3 = nn.Linear(hid_dim, act_dim)
        self.max_action = max_action

    def forward(self, obs):
        h1 = torch.tanh(self.layer1(obs))
        h2 = torch.tanh(self.layer2(h1))
        a = torch.tanh(self.layer3(h2))*self.max_action
        return a

class Critic(nn.Module):
    """ Critic Network """

    def __init__(self, obs_dim, hid_dim, act_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(obs_dim+act_dim, hid_dim)
        self.layer2 = nn.Linear(hid_dim, hid_dim)
        self.layer3 = nn.Linear(hid_dim, 1)

    def forward(self, obs, action):
        h1 = F.relu(self.layer1(torch.cat((obs, action), 1)))
        h2 = F.relu(self.layer2(h1))
        Q = self.layer3(h2)
        return Q

class DDPG(object):
    """ an implement of Deep Deterministic Policy Gradient (DDPG) """

    def __init__(self,
                 obs_dim,
                 hid_dim,
                 act_dim,
                 max_action=1,
                 replay_buffer_sz=1000000,
                 replay_start_sz=50000,
                 batch_sz=32,
                 gamma=0.99,
                 lr=0.0001,
                 tau=0.01,
                 explore_noise=0.25,
                 device=torch.device("cpu")):
        """ init DDPG """

        self.device = device                                    # cpu/gpu

        self.obs_dim = obs_dim                                  # dimension of observation
        self.hid_dim = hid_dim                                  # dimension of hidden layer
        self.act_dim = act_dim                                  # dimension of action
        self.max_action = max_action                            # max absolute value of action

        # actor network and its target
        self.actor = Actor(obs_dim, hid_dim, act_dim, max_action).to(device)
        self.target_actor = copy.deepcopy(self.actor)

        # critic network and its target
        self.critic = Critic(obs_dim, hid_dim, act_dim).to(device)
        self.target_critic = copy.deepcopy(self.critic)

        self.batch_sz = batch_sz                                # mini-batch size
        self.replay_buffer_sz = replay_buffer_sz                # size of experience replay buffer
        self.replay_start_sz = replay_start_sz                  # size threshold to start training

        # empty experience replay buffer, e = <s, a, r, done, s'>
        self.replay_buffer = np.empty(shape=(self.replay_buffer_sz, 2*self.obs_dim+act_dim+2))
        self.exp_cnt = 0                                        # counter of existing exp in replay buffer

        self.lr = lr                                            # learning rate

        # Adam optimizers to optimize actor and critic respectively
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)

        self.gamma = gamma                                      # discount factor
        self.tau = tau                                          # soft update rate for target network
        self.explore_noise = explore_noise                      # std of Gaussian noise for exploration
        self.noise_init = self.explore_noise                    # store initial value of noise

    def reset_noise(self, ratio):
        """ reset exploration noise """
        self.explore_noise = self.noise_init*ratio

    def policy(self, obs, exploration=True):
        """ decision making """
        obs = torch.FloatTensor(obs).to(self.device)
        action = self.actor(obs).detach().cpu().numpy()
        if exploration:
            action += np.random.normal(0, self.explore_noise*self.max_action, size=action.size)
        return action.clip(-self.max_action, self.max_action)

    def store_experience(self, obs, action, reward, done, obs_next):
        """ store experience into replay buffer """
        exp = np.concatenate([obs, action, [reward], [done], obs_next])
        self.replay_buffer[self.exp_cnt%self.replay_buffer_sz, :] = exp
        self.exp_cnt += 1

    def update(self):
        """ update DDPG with a mini-batch of exp """
        batch_indices = np.random.choice(min(self.exp_cnt, self.replay_buffer_sz), self.batch_sz)
        batch_exp = self.replay_buffer[batch_indices, :]
        s    = torch.FloatTensor(batch_exp[:, :self.obs_dim]).to(self.device)                           # state
        a    = torch.FloatTensor(batch_exp[:, self.obs_dim:self.obs_dim+self.act_dim]).to(self.device)  # action
        r    = torch.FloatTensor(batch_exp[:, self.obs_dim+self.act_dim]).to(self.device)               # reward
        done = torch.LongTensor(batch_exp[:, self.obs_dim+self.act_dim+1]).to(self.device)              # if done
        s_   = torch.FloatTensor(batch_exp[:, -self.obs_dim:]).to(self.device)                          # next state
        a_   = self.target_actor(s_)                                                                    # next action

        # calculate critic's loss (TD error)
        q_values = self.critic(s, a).view(self.batch_sz)
        q_targets = r + self.gamma*(1-done)*self.target_critic(s_, a_).detach().view(self.batch_sz)
        critic_loss = nn.MSELoss()(q_values, q_targets)
        # update critic
        if self.exp_cnt > self.replay_start_sz:
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

        # calculate actor's loss (negative of Q value)
        actor_loss = -self.critic(s, self.actor(s)).mean()
        # update actor
        if self.exp_cnt > self.replay_start_sz:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

        # soft target network update
        self._soft_update(self.target_critic, self.critic)
        self._soft_update(self.target_actor, self.actor)

        return [float(actor_loss), float(critic_loss)]

    def _soft_update(self, target_network, source_network):
        """ soft update """
        for target_param, param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(target_param.data*(1.0-self.tau)+param.data*self.tau)

    def save(self, filename):
        """ save model of actor """
        self.actor.cpu()
        torch.save(self.actor.state_dict(), filename)

    def load(self, filename):
        """ load model of actor """
        self.actor.load_state_dict(torch.load(filename))
        self.actor.eval()