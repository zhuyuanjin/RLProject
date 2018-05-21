import torch
from torch.autograd import Variable
from utils.ringmemory import ReplayMemory, Transition

class Agent(object):

    def __init__(self, gamma, memorty_size):
        self.gamma = gamma
        self.memory = ReplayMemory(memorty_size)
        self.use_cuda = torch.cuda.is_available()

    def backward(self, batch_size):
        raise NotImplementedError('not implement in bae class')

    def batch_sample(self, batch_size):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        reward_batch = torch.cat(batch.reward)
        action_batch = Variable(torch.cat(batch.action))
        next_state_batch = torch.cat(batch.next_state)
        terminal_batch = torch.cat(batch.terminal)
        if self.use_cuda:
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            action_batch = action_batch.cuda()
            next_state_batch = next_state_batch.cuda()
            terminal_batch = terminal_batch.cuda()
        return state_batch, action_batch, next_state_batch, reward_batch, terminal_batch


    def get_loss(self, batch_size):
        raise NotImplementedError('not implement in base class')



