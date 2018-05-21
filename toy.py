import torch
from torch.autograd import Variable
from model.dqn import DQNAgent
from network.CartPoleNet import CartPoleNet
import gym
from utils.utils import eps_decay

## Let's try a Toy-Model of CartPole
def main():
    policy_net = CartPoleNet()
    target_net = CartPoleNet()
    env = gym.make('CartPole-v1').unwrapped
    target_net_update_freq = 1000
    batch_size = 128
    num_steps = 10000
    memory_size = 4000
    gamma = 0.99
    eps_start = 0.99
    eps_end = 0.001
    action_dim = env.action_space.n
    epsParam = {
        'eps_start': eps_start,
        'eps_end': eps_end,
        'num_steps': num_steps
    }
    DQNParams = {
        'gamma': gamma,
        'memory_size': memory_size,
        'policy_net': policy_net,
        'target_net': target_net,
        'action_dim': action_dim
    }
    agent = DQNAgent(**DQNParams)
    step = 0
    duration = 0
    is_done = True
    while step < num_steps:
        if is_done == True:
            print('This episode has a duration of %-5d' % duration)
            duration = 0
            cur_state = env.reset()
            cur_state = Variable(torch.from_numpy(cur_state)).float().unsqueeze(0)
        step += 1
        duration += 1
        eps = eps_decay(step=step, **epsParam)
        action = agent.select_action(cur_state, eps)
        next_state, r, is_done, _ = env.step(action.item())
        x, x_dot, theta, theta_dot = next_state
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        r = r1 + r2
        next_state = Variable(torch.from_numpy(next_state)).float().unsqueeze(0)
        terminal = torch.Tensor([is_done])
        reward = torch.Tensor([r])
        agent.add_memory(cur_state, action, next_state, reward, terminal)
        cur_state = next_state
        agent.backward(batch_size)
        if step % target_net_update_freq == 0 and step > 1:
            agent.update_target_net_hard()
            print('Step %-5d, td_error %-10f, eps%-5f' % (step, agent.get_loss(memory_size), eps))


if __name__ == '__main__':
    main()





