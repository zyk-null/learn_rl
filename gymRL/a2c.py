import gym
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

class Config:
    def __init__(self):
        self.env_name = 'CartPole-v1'
        self.train_eps = 500
        self.test_eps = 5
        self.max_steps = 1000
        self.lr = 0.001
        self.gamma = 0.99
        self.seed = np.random.randint(0, 100)
        self.hidden_dim = 256
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
    
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
        return F.softmax(self.fc2(x), dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, n_states, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class A2C:
    def __init__(self, policy_net, value_net, cfg):
        self.policy_net = policy_net
        self.value_net = value_net
        self.policy_optimizer = optim.Adam(policy_net.parameters(), lr=cfg.lr)
        self.value_optimizer = optim.Adam(value_net.parameters(), lr=cfg.lr)
        self.gamma = cfg.gamma
        self.cfg = cfg

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.cfg.device)
        probs = self.policy_net(state)
        action = torch.multinomial(probs, num_samples=1).item()
        return action, probs[:, action].item()
    
    def update(self, trajectories):
        policy_loss = []
        value_loss = []

        for states, actions, rewards in trajectories:
            G = 0
            returns = []
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)

            states = torch.tensor(states, dtype=torch.float32).to(self.cfg.device)
            actions = torch.tensor(actions, dtype=torch.int64).to(self.cfg.device)
            returns = torch.tensor(returns, dtype=torch.float32).to(self.cfg.device)

            values = self.value_net(states).squeeze()
            # 计算优势
            advantages = returns - values

            # policy loss
            policy_log_probs = torch.log(self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze())
            policy_loss.append(-policy_log_probs * advantages.detach())

            # value loss
            value_loss.append(F.mse_loss(values, returns))

        self.policy_optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss = torch.stack(value_loss).sum()
        value_loss.backward()
        self.value_optimizer.step()
        
def train(env, agent, cfg):
    print('开始训练!')
    cfg.show()
    rewards = []
    for i_ep in range(cfg.train_eps):
        state, _ = env.reset(seed=cfg.seed)
        ep_reward = 0
        states, actions, rewards_list = [], [], []
        
        for _ in range(cfg.max_steps):
            action, prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards_list.append(reward)
            ep_reward += reward
            state = next_state

            if done:
                break

        agent.update([(states, actions, rewards_list)])
        rewards.append(ep_reward)
        print(f'回合 {i_ep+1}/{cfg.train_eps}, 奖励: {ep_reward:.2f}')
    print('训练完成!')
    return rewards

def test(env, agent, cfg):
    print('开始测试!')
    rewards = []
    for i_ep in range(cfg.test_eps):
        state, _ = env.reset(seed=cfg.seed)
        ep_reward = 0
        for _ in range(cfg.max_steps):
            action, _ = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            state = next_state
            if done:
                break
        rewards.append(ep_reward)
        print(f'测试回合 {i_ep+1}/{cfg.test_eps}, 奖励: {ep_reward:.2f}')
    print('测试完成!')
    return rewards

if __name__ == "__main__":
    cfg = Config()
    env = gym.make(cfg.env_name)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    cfg.n_states = n_states
    cfg.n_actions = n_actions
    policy_net = PolicyNetwork(n_states, n_actions, cfg.hidden_dim).to(cfg.device)
    value_net = ValueNetwork(n_states, cfg.hidden_dim).to(cfg.device)
    agent = A2C(policy_net, value_net, cfg)
    train_rewards = train(env, agent, cfg)
    env = gym.make(cfg.env_name, render_mode="human")
    test_rewards = test(env, agent, cfg)
    env.close()
