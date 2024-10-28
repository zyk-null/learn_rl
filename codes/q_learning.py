import torch
import numpy as np
import gym
import math
from collections import defaultdict

class Config:
    '''
    配置参数
    '''
    def __init__(self):
        self.env = 'CliffWalking-v0'  # 环境名称
        self.algorithm = 'Q-learning'  # 算法名称
        self.tranin_episodes = 400  # 训练的episode数目
        self.eval_episodes = 5 # 测试的episode数目
        self.max_steps = 200  # 每个episode的最大步数
        self.epsilon_start = 0.95  # epsilon的初始值
        self.epsilon_end = 0.01 # epsilon的最终值
        self.epsilon_decay = 300 # epsilon的衰减率
        self.gamma = 0.9  # 折扣因子
        self.lr = 0.1 # 学习率
        self.seed = 1 # 随机种子
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

class Q_learning:
    '''
    Q-learning算法
    '''
    def __init__(self, config, env):
        self.action_dim = env.action_space.n
        self.state_dim = env.observation_space.n
        self.lr = config.lr
        self.gamma = config.gamma
        self.epsilon = config.epsilon_start
        self.sample_count = 0
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        # Q表，使用defaultdict来初始化，当尝试访问Q中不存在的键时， 会自动调用 lambda 函数来创建一个包含 n_actions 个 0 的数组
        self.Q = defaultdict(lambda: torch.zeros(self.action_dim))

    def sample_action(self, state):
        '''
        采样动作，训练时用
        '''
        self.sample_count += 1
        # eplison递减，控制探索，随着训练次数的增加，逐渐减小探索概率
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               (1 - self.sample_count / self.epsilon_decay)
        # epsilon-greedy策略
        if np.random.uniform(0, 1) > self.epsilon:
            action = self.Q[str(state)].argmax().item() # 选择Q值最大的动作
        else:
            action = np.random.choice(self.action_dim) # 随机选择动作
        return action

    def predict_action(self, state):
        '''
        预测动作，测试时用
        '''
        action = self.Q[str(state)].argmax().item()
        return action
    
    def learn(self, state, action, reward, next_state, done):
        '''
        更新Q值
        '''
        predict_Q = self.Q[str(state)][action]
        if not done:
            target = reward + self.gamma * self.Q[str(next_state)].max()
        else:
            target = reward
        loss = (target - predict_Q).pow(2)
        self.Q[str(state)][action] += self.lr * (target - predict_Q)
        return loss
    
    def save(self, path):
        '''
        保存模型
        '''
        # 将 defaultdict 转换为普通字典
        regular_dict = {key: self.Q[key] for key in self.Q}
        torch.save(regular_dict, path)
        print(f"Save model to {path}")

    def load(self, path):
        '''
        加载模型
        '''
        regular_dict = torch.load(path)
        # 重新创建 defaultdict 并填充数据
        self.Q = defaultdict(lambda: torch.zeros(self.action_dim))
        for key, value in regular_dict.items():
            self.Q[key] = value
        print(f"Load model from {path}")

def train(cfg, env, agent):
    '''
    训练
    '''
    total_steps = 0
    for i in range(cfg.tranin_episodes):
        state = env.reset()
        total_reward = 0
        for t in range(cfg.max_steps):
            action = agent.sample_action(state) # 采样动作，需要探索
            next_state, reward, terminated, truncated, _ = env.step(action) # step函数返回的是下一个状态，奖励，是否结束，是否终止，调试信息, state可能是多种类型
            done = terminated or truncated
            loss = agent.learn(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            total_steps += 1
            if done:
                break
        if i % 10 == 0:
            print(f"Episode {i}, loss: {loss.item()}, total reward: {total_reward}")
    agent.save('./model/q_learning.pth')

def test(cfg, env, agent):
    '''
    测试
    '''
    agent.load('./model/q_learning.pth')
    total_reward = 0
    for i in range(cfg.eval_episodes):
        state = env.reset()
        for t in range(cfg.max_steps):
            action = agent.predict_action(state) # 预测动作，不需要探索
            next_state, reward, terminated, truncated, _ = env.step(action) # step函数返回的是下一个状态，奖励，是否结束，是否终止，调试信息
            done = terminated or truncated
            total_reward += reward
            state = next_state
            if done:
                break
    print(f"Average reward: {total_reward / cfg.eval_episodes}")

if __name__ == '__main__':
    cfg = Config()
    env = gym.make(cfg.env)
    agent = Q_learning(cfg, env)
    train(cfg, env, agent)
    env = gym.make(cfg.env, render_mode='human')
    test(cfg, env, agent)