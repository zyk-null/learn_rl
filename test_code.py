import torch

# 假设我们有一个简单的 Q-table，它是一个二维张量
# 这里我们假设有3个状态和2个可能的动作
Q_table = torch.tensor([
    [0.1, 0.4],  # 状态1对应的动作值
    [0.3, 0.2],  # 状态2对应的动作值
    [0.5, 0.7]   # 状态3对应的动作值
])

# 假设我们处于状态2
state = 1  # 状态索引从0开始

# 检索状态2的 Q 值
Q_values = Q_table[state]

# 找到具有最高 Q 值的动作的索引
action_index = Q_values.argmax().item()

print(f"在状态 {state} 下，具有最高 Q 值的动作索引是: {action_index}")