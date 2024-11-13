import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from toyrl import rl_utils

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class A2C(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):
        super().__init__()
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)  # 使用Adam优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma  # 折扣因子
        self.device = device

    def take_action(self, state):  # 根据动作概率分布随机采样
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)  # 时序差分误差
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数

class A2C_test(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, device, ckpt_path):
        super().__init__()
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.load_state_dict(torch.load(ckpt_path))
        self.device = device

    def take_action(self, state, take_max=True):  # 根据动作概率分布随机采样
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        if take_max:
            action = probs.argmax()
        else:
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
        return action.item()

def train_A2C(save_image=True, save_ckpt=True):    
    actor_lr = 1e-3
    critic_lr = 1e-2
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
    agent = A2C(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)
    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)
    
    if save_image:
        episode_idx = [i for i in range(len(return_list))]
        episode_reward = return_list
        
        # 使用plot函数绘制点对
        plt.plot(episode_idx, episode_reward, '-o', markersize=1)  # 'o'表示以点的形式绘制

        # 可以添加标题和标签
        plt.title(f'Plot of A2C')
        plt.xlabel('episode')
        plt.ylabel('reward')

        # 保存图像
        plt.savefig(f'A2C.png')  # 保存为PNG格式的文件
    
    if save_ckpt:
        torch.save(agent.state_dict(), 'A2C_ckpt.pth')
    return return_list

def test_A2C(ckpt_path="A2C_ckpt.pth"):
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
    agent = A2C_test(state_dim, hidden_dim, action_dim, device, ckpt_path)
    
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
    # train_A2C()
    test_A2C()