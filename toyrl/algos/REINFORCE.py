import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from toyrl.rl_utils import *

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    
class REINFORCE(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 device):
        self.policy_net = PolicyNet(state_dim, hidden_dim,
                                    action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.device = device

    def take_action(self, state):  # 根据动作概率分布随机采样
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.tensor([state_list[i]],
                                 dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G  # 每一步的损失函数
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降

class REINFORCE_test(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, device, ckpt_path):
        self.policy_net = PolicyNet(state_dim, hidden_dim,
                                    action_dim).to(device)
        self.policy_net.load_state_dict(torch.load(ckpt_path))
        self.device = device

    def take_action(self, state, take_max=True):  # 根据动作概率分布随机采样
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        if take_max:
            action = probs.argmax()
        else:
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
        return action.item()

def train_REINFORCE(save_image=True, save_ckpt=True):
    learning_rate = 1e-3
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env_name = "CartPole-v1"
    env = gym.make(env_name)
    env.reset()
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                state, info = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    # next_state, reward, _, _, _ = env.step(action)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    
    if save_image:
        episode_idx = [i for i in range(len(return_list))]
        episode_reward = return_list
        
        # 使用plot函数绘制点对
        plt.plot(episode_idx, episode_reward, '-o', markersize=1)  # 'o'表示以点的形式绘制

        # 可以添加标题和标签
        plt.title(f'Plot of REINFORCE')
        plt.xlabel('episode')
        plt.ylabel('reward')

        # 保存图像
        plt.savefig(f'REINFORCE.png')  # 保存为PNG格式的文件
    
    if save_ckpt:
        torch.save(agent.policy_net.state_dict(), 'REINFORCE_ckpt.pth')
    return return_list

def test_REINFORCE(ckpt_path="REINFORCE_ckpt.pth"):
    hidden_dim = 128
    device = torch.device("cuda") 

    env_name = 'CartPole-v1'
    env = gym.make(env_name, render_mode="human")
    random.seed(0)
    np.random.seed(0)
    env.reset(seed=0)
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCE_test(state_dim, hidden_dim, action_dim, device, ckpt_path)
    
    for episode in range(10):
        state, info = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            state = next_state
            done = terminated or truncated
        print(f"Episode {episode} Reward: {episode_reward}")
        
if __name__ == '__main__':
    # train_REINFORCE()
    test_REINFORCE()