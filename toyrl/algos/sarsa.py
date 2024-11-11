import time
import random
import matplotlib.pyplot as plt

class Env():
    def __init__(self, length, height):
        # define the height and length of the map
        self.length = length
        self.height = height
        # define the agent's start position
        self.x = 0
        self.y = 0

    def render(self, frames=50):
        for i in range(self.height):
            if i == 0: # cliff is in the line 0
                line = ['S'] + ['x']*(self.length - 2) + ['T'] # 'S':start, 'T':terminal, 'x':the cliff
            else:
                line = ['.'] * self.length
            if self.x == i:
                line[self.y] = 'o' # mark the agent's position as 'o'
            print(''.join(line))
        print('\033['+str(self.height+1)+'A')  # printer go back to top-left 
        time.sleep(1.0 / frames)

    def step(self, action):
        """4 legal actions, 0:up, 1:down, 2:left, 3:right"""
        change = [[0, 1], [0, -1], [-1, 0], [1, 0]]
        self.x = min(self.height - 1, max(0, self.x + change[action][0]))
        self.y = min(self.length - 1, max(0, self.y + change[action][1]))

        states = [self.x, self.y]
        reward = -1
        terminal = False
        if self.x == 0: # if agent is on the cliff line "SxxxxxT"
            if self.y > 0: # if agent is not on the start position 
                terminal = True
                if self.y != self.length - 1: # if agent falls
                    reward = -100
        return reward, states, terminal

    def reset(self):
        self.x = 0
        self.y = 0
        

class Sarsa_table():
    def __init__(self, length, height, actions=4, alpha=0.1, gamma=0.9, eps=0.1):
        self.table = [0] * actions * length * height # initialize all Q(s,a) to zero
        self.actions = actions
        self.length = length
        self.height = height
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def _index(self, a, x, y):
        """Return the index of Q([x,y], a) in Q_table."""
        return a * self.height * self.length + x * self.length + y

    def _epsilon(self):
        return self.eps
        # version for better convergence:
        # """At the beginning epsilon is 0.2, after 300 episodes decades to 0.05, and eventually go to 0."""
        # return 20. / (num_episode + 100)

    def take_action(self, x, y, num_episode):
        """epsilon-greedy action selection"""
        if random.random() < self._epsilon():
            return int(random.random() * 4)
        else:
            actions_value = [self.table[self._index(a, x, y)] for a in range(self.actions)]
            return actions_value.index(max(actions_value))

    def max_q(self, x, y):
        actions_value = [self.table[self._index(a, x, y)] for a in range(self.actions)]
        return max(actions_value)

    # def update(self, a, s0, s1, r, is_terminated):
    #     # both s0, s1 have the form [x,y]
    #     q_predict = self.table[self._index(a, s0[0], s0[1])]
    #     if not is_terminated:
    #         q_target = r + self.gamma * self.max_q(s1[0], s1[1])
    #     else:
    #         q_target = r
    #     self.table[self._index(a, s0[0], s0[1])] += self.alpha * (q_target - q_predict)
        
    def update(self, a0, a1, s0, s1, r, is_terminated):
        # both s0, s1 have the form [x,y]
        q_predict = self.table[self._index(a0, s0[0], s0[1])]
        if not is_terminated:
            # q_target = r + self.gamma * self.max_q(s1[0], s1[1])
            q_target = r + self.gamma * self.table[self._index(a1, s1[0], s1[1])]
        else:
            q_target = r
        self.table[self._index(a0, s0[0], s0[1])] += self.alpha * (q_target - q_predict)
        
def cliff_walk(cfg):
    env = Env(length=12, height=4)
    table = Sarsa_table(length=12, height=4, eps=cfg["eps"])
    logging = {}
    for num_episode in range(3000):
        # within the whole learning process
        episodic_reward = 0
        is_terminated = False
        s0 = [0, 0]
        while not is_terminated:
            # within one episode
            a0 = table.take_action(s0[0], s0[1], num_episode)
            r, s1, is_terminated = env.step(a0)
            a1 = table.take_action(s1[0], s1[1], num_episode)
            table.update(a0, a1, s0, s1, r, is_terminated)
            episodic_reward += r
            # env.render(frames=100)
            s0 = s1
            a0 = a1
        if num_episode % 20 == 0:
            print("Episode: {}, Score: {}".format(num_episode, episodic_reward))
        logging[num_episode] = episodic_reward
        env.reset()
    return logging

if __name__ == '__main__':
    cfg = {
        "eps": 0.1
    }
    logging = cliff_walk(cfg)
    episode_idx = list(logging.keys())
    episode_reward = list(logging.values())

    # 使用plot函数绘制点对
    plt.plot(episode_idx, episode_reward, '-o', markersize=1)  # 'o'表示以点的形式绘制

    # 可以添加标题和标签
    plt.title(f'Plot of Sarsa, eps={cfg["eps"]}')
    plt.xlabel('episode')
    plt.ylabel('reward')

    # 保存图像
    plt.savefig(f'sarsa_eps_{cfg["eps"]}.png')  # 保存为PNG格式的文件