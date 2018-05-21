import torch
from collections import namedtuple
from torch import nn
import gym
import torch.nn.functional as F
import pdb

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminal'))


class A2CAgent(object):
    def __init__(self, model, gamma, test_freq, env, env_test):
        self.gamma = gamma
        self.model = model
        self.env_test = env_test
        self.test_freq = test_freq
        self.env = env
        self.state = torch.from_numpy(self.env.reset()).unsqueeze(0).float()
        self.done = False
        self.use_cuda = torch.cuda.is_available()
        self.duration = 0
        if self.use_cuda:
            self.model = self.model.cuda()
            self.state = self.state.cuda()
        self.opt_actor = torch.optim.Adam(self.model.parameters())
        self.opt_critic = torch.optim.Adam(self.model.parameters())

    def roll_out(self, nstep):
        rollout = []
        if self.done:
            self.state = torch.from_numpy(self.env.reset()).unsqueeze(0).float().cuda()
            self.done = False

        for _ in range(nstep):
            if self.done:
                #print(self.duration)
                self.duration = 0
                break
            self.duration += 1
            dist, _ = self.model(self.state)
            action = torch.multinomial(dist, 1).item()
            ob, r, self.done, _ = self.env.step(action)
            x, x_dot, theta, theta_dot = ob
            r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
            r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
            r = r1 + r2
            action = torch.Tensor([[action]]).long()
            next_state = torch.from_numpy(ob).unsqueeze(0).float().cuda()
            terminal = torch.Tensor([[self.done]])
            reward = torch.Tensor([[r]])
            rollout.append(Transition(self.state, action, next_state, reward, terminal))
            self.state = next_state
        return rollout

    def backward(self, nstep):
        loss_fn = nn.MSELoss()
        rollout = self.roll_out(nstep)
        batch = Transition(*zip(*rollout))
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        mask = torch.cat(batch.terminal)
        reward_batch = torch.cat(batch.reward)
        action_batch = torch.cat(batch.action)
        if self.use_cuda:
            mask = mask.cuda()
            reward_batch = reward_batch.cuda()
            action_batch = action_batch.cuda()
        _, next_value = self.model(next_state_batch)
        y = next_value * (1 - mask) * self.gamma + reward_batch
        dist, value = self.model(state_batch)
        td_error = loss_fn(value, y.detach())
        self.opt_critic.zero_grad()
        td_error.backward(retain_graph=True)
        self.opt_critic.step()
        prob = dist.gather(1, action_batch)
        actor_loss = ((value-y).detach() * prob.log()).mean() + 0.01 * (dist * dist.log()).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

    def fit_model(self, nstep, nepoch):
        for epoch in range(nepoch):
            self.backward(nstep)
            if epoch % self.test_freq == 0:
                self.test_model()

    def test_model(self):
        duration = 0
        done = False
        state = torch.from_numpy(self.env_test.reset()).unsqueeze(0).float().cuda()
        while not done:
            dist, _ = self.model(state)
            action = dist.max(1)[1].item()
            ob, r, done, _ = self.env_test.step(action)
            state = torch.from_numpy(ob).unsqueeze(0).float().cuda()
            duration += 1
        print('The episode has a duration of {}'.format(duration))






































