import torch
from torch import nn
from model.dqn import DQNAgent
from network.PongNet import PongNet
import gym
from utils.utils import eps_decay, pongScreenCut, get_total_loss

## Let's try a Toy-Model of CartPole
def main():
    torch.cuda.set_device(1)
    policy_net = PongNet()
    target_net = PongNet()
    env = gym.make('Pong-v4').unwrapped
    target_net_update_freq = 1000
    batch_size = 128
    num_steps = 2000000
    memory_size = 4000
    gamma = 1
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
    is_done = True
    total_reward = 0
    while step < num_steps:
        if is_done == True:
            # print('This episode has a total reward of %-5d' % total_reward)
            env.reset()
            for _ in range(25):
                env.step(env.action_space.sample())
            last_screen = env.render('rgb_array')
            env.step(env.action_space.sample())
            cur_screen = env.render('rgb_array')
            total_reward = 0
        step += 1
        cur_state = pongScreenCut(cur_screen - last_screen)
        eps = eps_decay(step=step, **epsParam)
        action = agent.select_action(cur_state, eps)
        next_screen, r, is_done, _ = env.step(action[0])
        # x, x_dot, theta, theta_dot = next_state
        # r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        # r = r1 + r2
        total_reward += r
        if r != 0:
            is_done = True
        next_state = pongScreenCut(next_screen - cur_screen)
        terminal = torch.Tensor([is_done])
        reward = torch.Tensor([r])
        agent.add_memory(cur_state, action, next_state, reward, terminal)
        last_screen = cur_screen
        cur_screen = next_screen
        agent.backward(batch_size)
        if step % target_net_update_freq == 0 and step > 1:
            agent.update_target_net_hard()
            print('Step %-5d, td_error %-10f, eps%-5f' % (step, get_total_loss(agent, batch_size), eps))
            agent.save('param_dqn_pong')

if __name__ == '__main__':
    main()
