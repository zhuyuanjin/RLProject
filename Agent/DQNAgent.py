from Agent.Agent import Agent
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import random


class DQNAgent(Agent):
    def __init__(self, env, gamma, memory_size, Network, eps_start, eps_end, target_net_update_freq, num_steps, batch_size):
        super(DQNAgent, self).__init__(env, gamma, memory_size)
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.target_net_update_freq = target_net_update_freq
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.policy_net = Network()
        self.target_net = Network()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        if self.use_cuda:
            self.policy_net = self.policy_net.cuda()
            self.target_net = self.target_net.cuda()
        self.opt = torch.optim.RMSprop(self.policy_net.parameters())

    def _update_target_net_hard(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print('Hard Copy!!!!')

    def _select_action(self, state):
        eps = self._get_eps()
        return self._epsilon_greedy(state, eps)

    def _epsilon_greedy(self, state, eps):
        if self.use_cuda:
            state = state.cuda()
        if np.random.uniform() < eps:
            action = torch.LongTensor([random.randrange(self.env.action_space.n)])
        else:
            action = self.policy_net(state).max(1)[1].data.cpu()
        return action

    def _backward(self):
        if len(self.memory) < self.batch_size:
            return
        state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = self._batch_sampler(self.batch_size)
        qvalues = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_qvalues = self.target_net(next_state_batch).max(1)[0]
        target_values = (self.gamma * next_qvalues.data * (1 - terminal_batch)) + reward_batch
        target_values = Variable(target_values).unsqueeze(1)
        loss_fn = nn.MSELoss()
        loss = loss_fn(qvalues, target_values)
        self.opt.zero_grad()
        loss.backward()
        if self.step > 1 and self.step % self.target_net_update_freq == 1:
            print('self.step %-5d, loss%-8f' % (self.step, loss))
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.opt.step()

    def _get_eps(self):
        eps = self.eps_start - (self.eps_start - self.eps_end) * min(1, self.step / (0.7 * self.num_steps))
        return eps

    def fit_model(self):
        while self.step < self.num_steps:
            for _ in range(self.target_net_update_freq):
                self._step_counter()
                self._roll_out()
                self._backward()
            self._update_target_net_hard()




