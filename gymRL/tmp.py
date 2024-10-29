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
            advantages = returns - values  # 修改优势计算

            # 政策损失
            policy_log_probs = torch.log(self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze())
            policy_loss.append(-policy_log_probs * advantages.detach())

            # 价值损失
            value_loss.append(F.mse_loss(values, returns))

        self.policy_optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        self.value_optimizer.zero_grad()
        value_loss = torch.stack(value_loss).sum()
        value_loss.backward()
        self.value_optimizer.step()