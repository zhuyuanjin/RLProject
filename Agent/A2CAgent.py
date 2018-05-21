from Agent import Agent
import torch



class A2CAgent(Agent):
    def __init__(self, env, PolicyNet, ValueNet):
        super(A2CAgent, self).__init__(env)
        self.policy_net = PolicyNet()
        self.value_net = ValueNet()
        if self.use_cuda:
            self.policy_net = self.policy_net.t.cuda()
            self.value_net = self.value_net.cuda()
        self.actor_opt = torch.optim.RMSprop(self.policy_net.parameters())
        self.critic_opt = torch.optim.RMSprop(self.value_net.parameters())

    def _select_action(self, state):
        if self.use_cuda:
            state = state.cuda()
        prob = self.policy_net(state)
        action = torch.multinomial(prob, 1).squeeze()  ##Is that suitable to use the sampling function "mulinamial"?
        return action


    def _value_iteration(self, state_batch, action_batch, next_state_batch, reward_batch, terminal_batch):
        next_values = self.value_net(next_state_batch)
        values = self.value_net(state_batch)
        td_error = reward_batch + self.gamma * next_values * (1 - terminal_batch) - values
        self.critic_opt.zero_grad()
        td_error.backward()
        #         for param in self.value_net.parameters():
        #             param.grad.data.clamp_(-1, 1)                ## Constraint the l_infty norm of the gradients
        self.critic_opt.step()

    def _actor_update(self, state_batch, action_batch, next_state_batch, reward_batch, terminal_batch)
        next_values = self.value_net(next_state_batch)
        values = self.value_net(state_batch)
        td_errot = reward_batch + self.gamma * next_values * (1 - terminal_batch) - values
        self.critic_opt.zero_grad()
        return
