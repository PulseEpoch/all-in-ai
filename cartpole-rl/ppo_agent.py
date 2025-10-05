import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from actor_critic import Actor, Critic

class PPOAgent:
    def __init__(self, state_size, action_size, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, update_epochs=10, batch_size=64, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.device = device

        # Initialize Actor and Critic networks
        self.actor = Actor(state_size, action_size, device=device)
        self.critic = Critic(state_size, device=device)
        self.old_actor = Actor(state_size, action_size, device=device)
        self.old_actor.load_state_dict(self.actor.state_dict())

        # Optimizers
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Experience buffer
        self.states = []
        self.actions = []
        self.action_log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.old_actor(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob.item()

    def store_transition(self, state, action, action_log_prob, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def compute_gae(self):
        # Compute GAE advantage estimation
        states_tensor = torch.tensor(np.array(self.states), dtype=torch.float32).to(self.device)
        next_states_tensor = torch.tensor(np.array(self.next_states), dtype=torch.float32).to(self.device)
        
        values = self.critic(states_tensor).detach().cpu().numpy().flatten()
        next_values = self.critic(next_states_tensor).detach().cpu().numpy().flatten()
        advantages = np.zeros_like(self.rewards)
        last_advantage = 0

        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * next_values[t] * (1 - self.dones[t]) - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * last_advantage
            advantages[t] = last_advantage

        returns = advantages + values
        return advantages, returns

    def learn(self):
        # Compute GAE advantages and returns
        advantages, returns = self.compute_gae()

        # Convert to tensors and move to device
        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.actions, dtype=torch.long).to(self.device)
        old_action_log_probs = torch.tensor(self.action_log_probs, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Multiple epochs updates
        for _ in range(self.update_epochs):
            # Randomly sample batches
            indices = torch.randperm(len(states)).to(self.device)
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_action_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Calculate action probabilities and value function of current policy
                action_probs = self.actor(batch_states)
                dist = torch.distributions.Categorical(action_probs)
                action_log_probs = dist.log_prob(batch_actions)
                values = self.critic(batch_states).squeeze()

                # Compute ratio and clipped objectives
                ratio = torch.exp(action_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Calculate critic loss
                critic_loss = nn.MSELoss()(values, batch_returns)

                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                self.optimizer_actor.step()

                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                self.optimizer_critic.step()

        # Update old policy network
        self.old_actor.load_state_dict(self.actor.state_dict())

        # Clear buffer
        self.states = []
        self.actions = []
        self.action_log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        return actor_loss.item(), critic_loss.item()