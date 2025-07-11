import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        action_probs = self.softmax(self.fc2(x))
        return action_probs

class Critic(nn.Module):
    def __init__(self, state_size, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        value = self.fc2(x)
        return value

class ActorCriticAgent:
    def __init__(self, state_size, action_size, lr_actor=1e-3, lr_critic=1e-3, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        # Initialize networks
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)

        # Optimizers
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Loss function for critic
        self.critic_loss_fn = nn.MSELoss()

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = self.actor(state)
        # Sample action from distribution，exploration
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def learn(self, state, action_log_prob, reward, next_state, done):
        # Convert to tensors
        state = torch.from_numpy(state).float().unsqueeze(0)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32)

        # Critic update
        value = self.critic(state)
        next_value = self.critic(next_state) if not done else torch.tensor([0.0])
        target = reward + self.gamma * next_value
        critic_loss = self.critic_loss_fn(value, target.detach())

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Actor update
        advantage = (target - value).detach()
        actor_loss = -action_log_prob * advantage

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        return critic_loss.item(), actor_loss.item()

    def print_policy(self, env):
        """Print current policy"""
        action_symbols = ['↑', '→', '↓', '←']  # up, right, down, left
        print("Optimal Policy:")
        for i in range(env.size):
            row = []
            for j in range(env.size):
                # Create one-hot representation of state
                state_idx = i * env.size + j
                state = np.zeros(env.size * env.size)
                state[state_idx] = 1
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                # Get action probabilities
                self.actor.eval()
                with torch.no_grad():
                    action_probs = self.actor(state_tensor)
                action = torch.argmax(action_probs).item()
                row.append(action_symbols[action])
            print("\t".join(row))

def train_agent(env, agent, episodes=500, max_t=100):
    scores = []
    for i_episode in range(1, episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action, action_log_prob = agent.act(state)
            next_state, reward, done = env.step(action)
            critic_loss, actor_loss = agent.learn(state, action_log_prob, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores.append(score)

        if i_episode % 100 == 0:
            print(f'Episode {i_episode}, Average Score: {np.mean(scores[-100:]):.2f}')

    return scores

def main():
    # Assuming a simple grid world environment similar to existing code
    # You may need to adjust state_size and action_size based on your actual environment
    state_size = 16  # 4x4 grid
    action_size = 4   # up, right, down, left

    # Create environment (replace with actual environment initialization)
    class SimpleEnv:
        def __init__(self):
            self.size = 4
            self.reset()
        def reset(self):
            self.agent_pos = (0, 0)
            return self._get_state()
        def step(self, action):
            i, j = self.agent_pos
            if action == 0: i -= 1  # up
            elif action == 1: j += 1  # right
            elif action == 2: i += 1  # down
            elif action == 3: j -= 1  # left
            i = max(0, min(i, self.size-1))
            j = max(0, min(j, self.size-1))
            self.agent_pos = (i, j)
            done = (i == self.size-1) and (j == self.size-1)
            reward = 1.0 if done else -0.01
            return self._get_state(), reward, done
        def _get_state(self):
            state = np.zeros(self.size * self.size)
            state[self.agent_pos[0] * self.size + self.agent_pos[1]] = 1
            return state

    env = SimpleEnv()
    agent = ActorCriticAgent(state_size, action_size)
    scores = train_agent(env, agent)

    # 打印最终策略
    print("Training completed! Printing optimal policy:")
    agent.print_policy(env)

if __name__ == '__main__':
    main()