import torch
from torch.autograd import Variable
from utils.RingMemory import ReplayMemory, Transition

class Agent:

    def __init__(self, env, gamma, memory_size):
        self.gamma = gamma
        self.step = 0
        self.use_cuda = torch.cuda.is_available()
        self.memory = ReplayMemory(memory_size)
        self.env = env
        self.state = self.env.reset()  ##which is in a type of np.ndarray
        self.state = Variable(torch.from_numpy(self.state)).unsqueeze(0).float()
        ##which is in a type of torch.FloatTensor
        self.is_done = False

    def _select_action(self, state):
        ##The return should be a LongTensor of size 1
        raise NotImplementedError("not implemented in base calss")

    def _roll_out(self):
        if self.is_done:
            self.state = self.env.reset()
            self.state = Variable(torch.from_numpy(self.state)).unsqueeze(0).float()
            self.is_done = False
        action = self._select_action(self.state)
        next_state, r, self.is_done, _ = self.env.step(action[0])
        reward = torch.Tensor([r])  ##which is a FloatTensor of size 1
        terminal = torch.Tensor([self.is_done])
        next_state = Variable(torch.from_numpy(next_state)).unsqueeze(0).float()
        self.memory.push(self.state, action, next_state, reward, terminal)
        self.state = next_state

    def _backward(self):
        raise NotImplementedError('not implemented in base class')

    def _batch_sampler(self, batch_size):
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

    def _step_counter(self):
        self.step += 1

    def fit_model(self):
        raise NotImplementedError('not implemented in base class')

    def _eval_model(self):
        raise NotImplementedError('not implemented in base class')

    def test_model(self):
        raise NotImplementedError('not implemented in base class')


