import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.fc(x)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.fc(x)

class GRPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, lambda_gae=0.95, kl_target=0.01, beta=1.0):
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=lr)
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.kl_target = kl_target  # GRPO的KL散度目标
        self.beta = beta  # KL惩罚系数
        self.mse_loss = nn.MSELoss()
    
    def select_action(self, state):
        # 确保状态是正确的形状 (4,)，添加容错处理
        state = np.ravel(state)  # 展平任意嵌套结构为1D数组
        if state.size < 4:
            state = np.pad(state, (0, 4 - state.size), mode='constant')
        elif state.size > 4:
            state = state[:4]
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()
    
    def compute_gae(self, rewards, values, dones, next_value):
        # 组内相对优势估计（GAE）：通过时序差分和指数加权平均计算优势值
        # 核心思想：利用相邻时间步的奖励和价值函数构建相对优势，体现组内比较特性
        advantages = []
        advantage = 0
        # 反向迭代计算每个时间步的优势值
        for t in reversed(range(len(rewards))):
            # 计算TD误差：当前奖励 + 未来价值估计 - 当前价值估计
            # 体现当前步与未来步的相对价值差异
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            # 指数加权累积优势，lambda控制历史信息的衰减率
            # 实现组内多步优势的相对加权组合
            advantage = delta + self.gamma * self.lambda_gae * advantage * (1 - dones[t])
            advantages.insert(0, advantage)
            next_value = values[t]
        return advantages
    
    def update(self, states, actions, old_log_probs, advantages, returns, values):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        # 归一化优势函数以稳定训练
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.detach()
        returns = returns.detach()
        
        # 计算当前策略的概率和对数概率
        probs = self.policy_net(states)
        dist = Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        
        # 计算策略比率
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # 计算KL散度
        # 计算正确的KL散度 (新策略 || 旧策略)
        kl_divergence = (new_log_probs.exp() * (new_log_probs - old_log_probs)).mean().item()
        
        # GRPO策略损失：线性优势估计 + KL惩罚
        # 添加熵正则化鼓励探索
        entropy = dist.entropy().mean()
        # 增加熵正则化系数增强探索
        policy_loss = -(ratio * advantages).mean() + self.beta * kl_divergence - 0.05 * entropy
        
        # 自适应调整beta系数
        if kl_divergence < self.kl_target / 1.5:
            self.beta /= 2
        elif kl_divergence > self.kl_target * 1.5:
            self.beta *= 2
        
        # 使用传入的values计算价值损失
        # 将列表转换为张量并调整维度
        values = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
        value_loss = self.mse_loss(values, returns)
        
        # 总损失
        total_loss = policy_loss + 0.5 * value_loss
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        entropy = dist.entropy().mean().item()
        return policy_loss.item(), value_loss.item(), kl_divergence, entropy

def train(env_name, agent, episodes=1000, max_steps=200):
    env = gym.make(env_name)
    total_rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()  # 解包observation和info元组
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        total_reward = 0
        
        for step in range(max_steps):
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # 合并终止状态
            state = next_state  # 更新状态为下一个状态
            
            # 存储轨迹信息
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(agent.value_net(torch.FloatTensor(state).unsqueeze(0)).item())
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # 计算GAE和回报
        next_value = agent.value_net(torch.FloatTensor(next_state).unsqueeze(0)).item() if not done else 0
        advantages = agent.compute_gae(rewards, values, dones, next_value)
        returns = [advantage + value for advantage, value in zip(advantages, values)]
        
        # 更新策略和价值网络
        policy_loss, value_loss, kl_divergence, entropy = agent.update(states, actions, log_probs, advantages, returns, values)
        
        total_rewards.append(total_reward)
        
        # 记录训练指标
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward={total_reward:.2f}, KL={kl_divergence:.4f}, Entropy={entropy:.4f}, ValueLoss={value_loss:.4f}")

        if episode % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, KL Divergence: {kl_divergence:.6f}")
    
    env.close()
    return total_rewards

def test(env_name, agent, episodes=10):
    env = gym.make(env_name)
    total_rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()  # 解包observation和info元组
        total_reward = 0
        done = False
        
        while not done:
            action, _ = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # 合并终止状态
            state = next_state  # 更新状态为下一个状态
            total_reward += reward
        
        total_rewards.append(total_reward)
        print(f"Test Episode {episode}, Reward: {total_reward}")
    
    avg_reward = np.mean(total_rewards)
    print(f"Test Average Reward: {avg_reward:.2f}")
    env.close()
    return avg_reward

if __name__ == "__main__":
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()
    
    agent = GRPOAgent(state_dim, action_dim)
    
    print("Training...")
    train_rewards = train(env_name, agent, episodes=500)
    
    print("Testing...")
    test_avg_reward = test(env_name, agent)
    
    # 简单绘图展示训练奖励曲线
    import matplotlib.pyplot as plt
    plt.plot(train_rewards)
    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("training_rewards.png")
    print("Training rewards plot saved as training_rewards.png")