import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random

class QNetwork(nn.Module):
    """Simple Q-network for approximating the action-value function"""
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        """Forward pass, returning Q-values for each action"""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Experience replay buffer for storing and sampling agent experiences"""
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """添加新经验到缓冲区"""
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self):
        """从缓冲区中随机采样一批经验"""
        experiences = random.sample(self.buffer, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return current size of the buffer"""
        return len(self.buffer)

class DQNAgent:
    """DQN agent implementing Deep Q-Network reinforcement learning algorithm"""
    def __init__(self, state_size, action_size, seed=42,
                 hidden_size=64, buffer_size=int(1e5), batch_size=64,
                 gamma=0.99, lr=5e-4, update_every=4):
        """
        参数:
            state_size (int): 状态空间大小
            action_size (int): 动作空间大小
            seed (int): 随机种子
            hidden_size (int): 神经网络隐藏层大小
            buffer_size (int): 经验回放缓冲区大小
            batch_size (int): 每次采样的批量大小
            gamma (float): 折扣因子
            lr (float): 学习率
            update_every (int): 目标网络更新频率
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.update_every = update_every

        # Q-network and target network
        self.qnetwork_local = QNetwork(state_size, action_size, hidden_size)
        self.qnetwork_target = QNetwork(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Experience replay buffer
        self.memory = ReplayBuffer(buffer_size, batch_size)
        # Time step counter for periodic target network updates
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """Process agent's step, store experience, and learn periodically"""
        # Add experience to replay buffer
        self.memory.add(state, action, reward, next_state, done)

        # 每update_every步学习一次
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples in buffer, perform learning
            if len(self.memory) > self.memory.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=0.0):
        """Select action based on current state (ε-greedy policy)
        Parameters:
            state (array_like): Current state
            eps (float): ε value for ε-greedy policy
        Returns:
            int: Selected action
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # ε-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """Update Q-network parameters using experience replay
        Parameters:
            experiences (Tuple[torch.Tensor]): Tuple containing (s, a, r, s', done)
        """
        states, actions, rewards, next_states, dones = experiences

        # Get target Q-values: max Q(s', a') from target network
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Calculate target Q-values: r + γ * max Q(s', a') * (1 - done)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get current Q-values: Q(s, a) from local network
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Calculate loss
        loss = nn.MSELoss()(Q_expected, Q_targets)
        # Minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model, tau=1e-3):
        """Soft update target network parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Parameters:
            local_model (PyTorch model): Local model whose weights will be copied
            target_model (PyTorch model): Target model whose weights will be updated
            tau (float): Soft update parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def print_policy(self, env):
        """Print current policy"""
        action_symbols = ['↑', '→', '↓', '←']  # up, right, down, left
        print("Optimal Policy:")
        for i in range(env.size):
            row = []
            for j in range(env.size):
                state = env._pos_to_state((i, j))
                state_tensor = torch.from_numpy(np.array([state])).float().unsqueeze(0)
                self.qnetwork_local.eval()
                with torch.no_grad():
                    q_values = self.qnetwork_local(state_tensor)
                action = np.argmax(q_values.cpu().data.numpy())
                row.append(action_symbols[action])
            print("\t".join(row))


def train_agent(env, agent, episodes=500, max_t=100, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Train DQN agent
    Parameters:
        env: Environment object implementing reset() and step() methods
        agent: DQNAgent instance
        episodes (int): Number of training episodes
        max_t (int): Maximum steps per episode
        eps_start (float): Initial exploration rate
        eps_end (float): Minimum exploration rate
        eps_decay (float): Exploration rate decay factor
    Returns:
        scores (list): Total reward per episode
    """
    scores = []                        # Store scores per episode
    eps = eps_start                    # Initialize exploration rate

    for i_episode in range(1, episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)  # Decay exploration rate

        # Print training progress every 100 episodes
        if i_episode % 100 == 0:
            print(f'Episode {i_episode}, Average Score: {np.mean(scores[-100:]):.2f}, Epsilon: {eps:.2f}')

    return scores

if __name__ == "__main__":
    # this is a simple grid world environment for demonstrating DQN training
    class SimpleGridEnv:
        """Simple grid world environment for demonstrating DQN training"""
        def __init__(self, size=4):
            self.size = size
            self.state_size = size * size
            self.action_size = 4  # up, right, down, left
            self.goal_pos = (size-1, size-1)
            self.reset()

        def reset(self):
            self.agent_pos = (0, 0)
            return np.array([self._pos_to_state(self.agent_pos)])

        def _pos_to_state(self, pos):
            return pos[0] * self.size + pos[1]

        def step(self, action):
            row, col = self.agent_pos
            # Actions: 0=up, 1=right, 2=down, 3=left
            if action == 0 and row > 0: row -= 1
            elif action == 1 and col < self.size-1: col += 1
            elif action == 2 and row < self.size-1: row += 1
            elif action == 3 and col > 0: col -= 1

            self.agent_pos = (row, col)
            state = self._pos_to_state(self.agent_pos)
            done = (self.agent_pos == self.goal_pos)
            reward = 100 if done else -1
            return np.array([state]), reward, done

    # Create environment and agent
    env = SimpleGridEnv(size=4)
    agent = DQNAgent(state_size=1, action_size=env.action_size)

    # train agent
    print("start tr...")
    scores = train_agent(env, agent, episodes=500)

    # Print final results
    print("Training completed! Average score over last 100 episodes: {:.2f}".format(np.mean(scores[-100:])))
    # print final policy
    agent.print_policy(env)