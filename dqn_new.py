import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from model.base import Agent
import numpy as np
import random


class DQNAgent(Agent):

    def __init__(self,
                 gamma,
                 memory_size,
                 policy_net,
                 target_net,
                 action_dim,
                 ):
        super(DQNAgent, self).__init__(gamma, memory_size)
        self.action_dim = action_dim
        self.policy_net = policy_net
        if self.use_cuda:
            self.policy_net = self.policy_net.cuda()
        self.opt = self.opt = torch.optim.RMSprop(self.policy_net.parameters())


    def select_action(self, state, eps):
        return self._epsilon_greedy(state, eps)

    def _epsilon_greedy(self, state, eps):
        if self.use_cuda:
            state = state.cuda()
        if np.random.uniform() < eps:
            action = torch.LongTensor([random.randrange(self.action_dim)])
        else:
            action = self.policy_net(state).max(1)[1].data.cpu()
        return action

    def get_loss(self, batch_size):
        state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = self.batch_sample(min(batch_size, len(self.memory)))
        qvalues = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_qvalues = self.policy_net(next_state_batch).max(1)[0]
        target_values = (self.gamma * next_qvalues * (1 - terminal_batch)) + reward_batch
        target_values = target_values.unsqueeze(1)
        # loss_fn = nn.MSELoss()
        # loss = loss_fn(qvalues, target_values.detach())
        loss = F.mse_loss(qvalues, target_values)
        return loss

    def backward(self, batch_size):
        if len(self.memory) < batch_size:
            return
        loss = self.get_loss(batch_size)
        self.opt.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.opt.step()

    def add_memory(self, *args):
        # add a sample of the (s, a, s', r, terminal) to the memory of the agent
        self.memory.push(*args)

    def save(self, filename):
        torch.save(self.policy_net, filename)


