import gym
import random
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from collections import deque
from torch.cuda.amp import GradScaler, autocast

class Config:
    def __init__(self):
        self.env_name = 'CartPole-v1'
        self.algo_name = 'DQN'
        self.train_eps = 500
        self.test_eps = 5
        self.max_steps = 100000
        self.epsilon_start = 0.95
        self.epsilon_end = 0.01
        self.epsilon_decay = 800
        self.lr = 0.001
        self.gamma = 0.9
        self.seed = random.randint(0, 100)
        self.batch_size = 64
        self.memory_capacity = 100000
        self.hidden_dim = 256 # 隐藏层维度
        self.target_update = 4 # 目标网络更新频率
        self.n_states = None
        self.n_actions = None
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def show(self):
        print('-' * 30 + '参数列表' + '-' * 30)
        for k, v in vars(self).items():
            print(k, '=', v)
        print('-' * 60)

class MLP(nn.Module):
    def __init__(self, n_states, n_actions, n_dims=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_states, n_dims)
        self.fc2 = nn.Linear(n_dims, n_dims)
        self.fc3 = nn.Linear(n_dims, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, transitions):
        self.buffer.append(transitions)

    # sequential=False表示随机采样，True表示序列采样
    def sample(self, batch_size, sequential = False):
        batch_size = min(batch_size, len(self.buffer))
        if sequential:
            rand_index = random.randint(0, len(self.buffer) - batch_size + 1)
            batch = [self.buffer[i] for i in range(rand_index, rand_index + batch_size)]
        else:
            batch = random.sample(self.buffer, batch_size)
        return zip(*batch)
    
    def size(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()

class DQN:
    def __init__(self, policy_net, target_net, memory, cfg):
        self.sample_count = 0 # 记录采样次数
        self.memory = memory
        self.policy_net = policy_net
        self.target_net = target_net
        self.cfg = cfg
        self.epsilon = cfg.epsilon_start
        self.optimizer = optim.Adam(policy_net.parameters(), lr=cfg.lr)
        self.loss = nn.MSELoss()
        self.scaler = GradScaler() # 混合精度训练

    @torch.no_grad()
    def sample_action(self, state):
        self.sample_count += 1
        self.epsilon = self.cfg.epsilon_end + (self.cfg.epsilon_start - self.cfg.epsilon_end) * \
                       np.exp(-1. * self.sample_count / self.cfg.epsilon_decay)
        if np.random.uniform(0, 1) > self.epsilon:
            # state是numpy数组，需要转换为tensor，unsqueeze是为了增加一个batch维度
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.cfg.device)
            # 选择Q值最大的动作，dim=1表示在行上取最大值，返回的是列索引
            action = self.policy_net(state).argmax(dim=1).item()
        else:
            action = np.random.choice(self.cfg.n_actions)
        return action
    
    @torch.no_grad()
    def predict_action(self, state):
        state = torch.tensor(np.array(state), device=self.cfg.device, dtype=torch.float32).unsqueeze(0)
        action = self.policy_net(state).argmax(dim=1).item()
        return action
    
    def update(self):
        # 如果记忆库中的样本数量小于batch_size，直接返回
        if self.memory.size() < self.cfg.batch_size:
            return
        # 从记忆库中采样一个batch的数据
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.cfg.batch_size)
        # 将数据转换为tensor
        state_batch = torch.tensor(state_batch, dtype=torch.float32).to(self.cfg.device)
        action_batch = torch.tensor(action_batch, dtype=torch.long).to(self.cfg.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).to(self.cfg.device)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32).to(self.cfg.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32).to(self.cfg.device)

        # autocast是混合精度训练的上下文管理器，可以自动完成前向传播和反向传播的精度转换
        with autocast():
            q_value = self.policy_net(state_batch).gather(1, action_batch)
            next_q_value = self.target_net(next_state_batch).max(dim=1)[0].detach()
            expect_q_value = reward_batch + self.cfg.gamma * next_q_value * (1 - done_batch)
            loss = F.mse_loss(q_value, expect_q_value.unsqueeze(1))

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward() # scale是将损失值放大，backward是反向传播,这里是自动完成精度转换
        self.scaler.step(self.optimizer)
        self.scaler.update()

# 创建环境和智能体
def env_agent_config(cfg):
    # env = gym.make(cfg.env_name, render_mode = "human")
    # human模式下，可以看到训练, 训练不需要渲染
    env = gym.make(cfg.env_name)
    print(f'观测空间 = {env.observation_space}')
    print(f'动作空间 = {env.action_space}')
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    cfg.n_states = n_states
    cfg.n_actions = n_actions
    # jit.script是TorchScript的一种形式，可以将Python代码转换为TorchScript代码，提高代码的运行效率
    policy_net = torch.jit.script(MLP(n_states, n_actions, cfg.hidden_dim).to(cfg.device)) 
    target_net = torch.jit.script(MLP(n_states, n_actions, cfg.hidden_dim).to(cfg.device)) 
    memory = ReplayBuffer(cfg.memory_capacity)
    agent = DQN(policy_net, target_net, memory, cfg)
    return env, agent

def train(env, agent, cfg):
    print('开始训练!')
    cfg.show()
    rewards, steps = [], []
    for i in range(cfg.train_eps):
        ep_reward, ep_step = 0.0, 0
        state, _ = env.reset(seed=cfg.seed)
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.sample_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.memory.push((state, action, reward, next_state, done))
            state = next_state
            agent.update()
            ep_reward += reward
            if done:
                break
        if i % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        rewards.append(ep_reward)
        steps.append(ep_step)
        print(f'回合:{i + 1}/{cfg.train_eps}, 奖励:{ep_reward:.3f}, 步数:{ep_step:d}. epsilon:{agent.epsilon:.3f}')
    print('完成训练!')
    return rewards, steps

def test(env, agent, cfg):
    print('开始测试!')
    rewards, steps = [], []
    for i in range(cfg.test_eps):
        ep_reward, ep_step = 0.0, 0
        state, _ = env.reset(seed=cfg.seed)
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.predict_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            ep_reward += reward
            if terminated or truncated:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        print(f'回合:{i + 1}/{cfg.test_eps}, 奖励:{ep_reward:.3f}')
    print('结束测试!')
    return rewards, steps

if __name__ == '__main__':
    cfg = Config()
    env, agent = env_agent_config(cfg)
    train_rewards, train_steps = train(env, agent, cfg)
    # 测试用human模式，可以看到训练效果
    env = gym.make(cfg.env_name, render_mode="human")
    test_rewards, test_steps = test(env, agent, cfg)
    env.close()