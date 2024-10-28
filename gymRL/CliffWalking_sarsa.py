import gym
import math
import numpy as np
import torch
from collections import defaultdict


class Config:
    '''配置参数
    '''

    def __init__(self):
        self.env_name = 'CliffWalking-v0'  # 环境名称
        self.algo_name = 'SARSA'  # 算法名称
        self.train_eps = 400  # 训练回合数
        self.test_eps = 20  # 测试回合数
        self.max_steps = 200  # 每个回合最大步数
        self.epsilon_start = 0.95  # e-greedy策略中epsilon的初始值
        self.epsilon_end = 0.01  # e-greedy策略中epsilon的最终值
        self.epsilon_decay = 300  # e-greedy策略中epsilon的衰减率
        self.gamma = 0.9  # 折扣因子
        self.lr = 0.1  # 学习率
        self.seed = 1  # 随机种子
        if torch.cuda.is_available():  # 是否使用GPUs
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')


class SARSA(object):
    def __init__(self, n_states,
                 n_actions, cfg):
        self.n_actions = n_actions
        self.lr = cfg.lr  # 学习率
        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon_start
        self.sample_count = 0
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.Q_table = defaultdict(lambda: np.zeros(n_actions))  # 用嵌套字典存放状态->动作->状态-动作值（Q值）的映射，即Q表
        # Q表的state是字符串，action是整数，Q值是浮点数，为了节省内存，用嵌套字典存放Q表，嵌套字典的键是state，值是一个字典，字典的键是action，值是Q值
        # 这样就可以通过Q_table[state][action]来访问Q值，如果(state, action)没有访问过，Q值默认为0
        # 这里用defaultdict(lambda: np.zeros(n_actions))来实现嵌套字典，当访问Q_table[state]时，如果state不在Q_table中，会自动创建一个字典，字典的值是np.zeros(n_actions)

    def sample_action(self, state):
        ''' 采样动作，训练时用
        '''
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       math.exp(-1. * self.sample_count / self.epsilon_decay)  # epsilon是会递减的，这里选择指数递减
        # e-greedy 策略
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q_table[str(state)])  # 选择Q(s,a)最大对应的动作
        else:
            action = np.random.choice(self.n_actions)  # 随机选择动作
        return action

    def predict_action(self, state):
        ''' 预测或选择动作，测试时用
        '''
        action = np.argmax(self.Q_table[str(state)])
        return action

    def update(self, state, action, reward, next_state, terminated):
        Q_predict = self.Q_table[str(state)][action]
        if terminated:  # 终止状态
            Q_target = reward
        else:
            # Q_target = reward + self.gamma * np.max(self.Q_table[str(next_state)])
            # sarsa算法的更新公式
            next_action = self.sample_action(next_state)
            Q_target = reward + self.gamma * self.Q_table[str(next_state)][next_action]
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_predict)


def train(cfg, env, agent):
    print('开始训练！')
    print(f'环境:{cfg.env_name}, 算法:{cfg.algo_name}, 设备:{cfg.device}')
    rewards = []  # 记录奖励
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录每个回合的奖励

            # 动态控制渲染：在最后10回合或随机某些回合开启渲染
        if i_ep >= cfg.train_eps - 10:  # 只渲染最后10个回合
            env = gym.make(cfg.env_name, render_mode="human")
        else:
            env = gym.make(cfg.env_name)  # 无渲染

        state = env.reset(seed=cfg.seed)  # 重置环境,即开始新的回合
        
        while True:
            action = agent.sample_action(state)  # 根据算法采样一个动作
            next_state, reward, terminated, truncated, info = env.step(action)  # 与环境进行一次动作交互
            terminated = terminated or truncated
            agent.update(state, action, reward, next_state, terminated)  # Q学习算法更新
            state = next_state  # 更新状态
            ep_reward += reward
            if terminated:
                break
        rewards.append(ep_reward)
        if (i_ep + 1) % 20 == 0:
            print(f"回合：{i_ep + 1}/{cfg.train_eps}，奖励：{ep_reward:.1f}，Epsilon：{agent.epsilon:.3f}")
    print('完成训练！')
    return {"rewards": rewards}


def test(cfg, env, agent):
    print('开始测试！')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = []  # 记录所有回合的奖励
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # 记录每个episode的reward
        state = env.reset(seed=cfg.seed)  # 重置环境, 重新开一局（即开始新的一个回合）
        while True:
            action = agent.predict_action(state)  # 根据算法选择一个动作
            next_state, reward, terminated, truncated, info = env.step(action)  # 与环境进行一个交互
            terminated = terminated or truncated
            state = next_state  # 更新状态
            ep_reward += reward
            if terminated:
                break
        rewards.append(ep_reward)
        print(f"回合数：{i_ep + 1}/{cfg.test_eps}, 奖励：{ep_reward:.1f}")
    print('完成测试！')
    return {"rewards": rewards}


def env_agent_config(cfg, seed=1):
    '''创建环境和智能体
    '''
    # env = gym.make(cfg.env_name, render_mode = "human")
    env = gym.make(cfg.env_name)
    n_states = env.observation_space.n  # 状态维度
    n_actions = env.action_space.n  # 动作维度
    agent = SARSA(n_states, n_actions, cfg)
    return env, agent


# 获取参数
cfg = Config()
# 训练
env, agent = env_agent_config(cfg)
res_dic = train(cfg, env, agent)
# res_dic = test(cfg, env, agent)