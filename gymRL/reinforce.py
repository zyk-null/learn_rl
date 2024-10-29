import gym
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

class Config:
    def __init__(self):
        self.env_name = 'CartPole-v1'
        self.algo_name = 'REINFORCE'
        self.train_eps = 500
        self.test_eps = 5
        self.lr = 0.001
        self.gamma = 0.99
        self.seed = 42
        self.hidden_dim = 256
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def show(self):
        print('-' * 30 + '参数列表' + '-' * 30)
        for k, v in vars(self).items():
            print(k, '=', v)
        print('-' * 60)

class PolicyNetwork(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x
    
class REINFORCE:
    def __init__(self, policy_net, cfg):
        self.policy_net = policy_net
        self.cfg = cfg
        self.optimizer = optim.Adam(policy_net.parameters(), lr=cfg.lr)

    def sample_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.cfg.device)
        probs = self.policy_net(state) # 通过策略网络得到动作概率
        action = torch.multinomial(probs, 1).item() # 从多项分布中采样一个动作
        return action
    
    # 计算回报
    def compute_returns(self, rewards):
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.cfg.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.cfg.device)
        return (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # 更新策略网络
    def update_policy(self, log_probs, returns):
        # loss是负的期望回报
        # loss = sum(-log_prob * return)
        loss = -torch.sum(torch.stack(log_probs) * returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# 创建环境和智能体
def env_agent_config(cfg):
    env = gym.make(cfg.env_name)
    print(f'观测空间 = {env.observation_space}')
    print(f'动作空间 = {env.action_space}')
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    cfg.n_states = n_states
    cfg.n_actions = n_actions
    policy_net = PolicyNetwork(n_states, n_actions, cfg.hidden_dim).to(cfg.device)
    agent = REINFORCE(policy_net, cfg)
    return env, agent

def train(env, agent, cfg):
    print('开始训练!')
    cfg.show()
    rewards_all_episodes = []
    for i in range(cfg.train_eps):
        state, _ = env.reset(seed=cfg.seed)
        log_probs = []
        rewards = []
        for t in range(1, 1000):  # 限制步数避免无限循环
            action = agent.sample_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # 计算log_prob
            log_prob = torch.log(agent.policy_net(torch.tensor(state, dtype=torch.float32).to(cfg.device))[action])
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
            if done:
                break

        rewards_all_episodes.append(sum(rewards))
        returns = agent.compute_returns(rewards)
        agent.update_policy(log_probs, returns)

        print(f"Episode {i+1}/{cfg.train_eps}, Total Reward: {sum(rewards):.2f}")

    print('训练完成!')
    return rewards_all_episodes

def test(env, agent, cfg):
    print('开始测试!')
    rewards = []
    for i in range(cfg.test_eps):
        state, _ = env.reset(seed=cfg.seed)
        ep_reward = 0
        for t in range(1, 1000):
            action = agent.sample_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        rewards.append(ep_reward)
        print(f"Episode {i+1}/{cfg.test_eps}, Total Reward: {ep_reward:.2f}")
    print('测试结束!')
    return rewards

if __name__ == '__main__':
    cfg = Config()
    env, agent = env_agent_config(cfg)
    train_rewards = train(env, agent, cfg)
    env = gym.make(cfg.env_name, render_mode="human")
    test_rewards = test(env, agent, cfg)
    env.close()
