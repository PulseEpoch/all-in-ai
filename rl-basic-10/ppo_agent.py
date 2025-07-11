import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from actor_critic import Actor, Critic


class PPOAgent:
    def __init__(self, state_size, action_size, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, update_epochs=10, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.batch_size = batch_size

        # Reuse existing Actor and Critic networks
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
        self.old_actor = Actor(state_size, action_size)
        self.old_actor.load_state_dict(self.actor.state_dict())

        # Optimizers
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(
            self.critic.parameters(), lr=lr_critic)

        # Experience buffer
        self.states = []
        self.actions = []
        self.action_log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def act(self, state):
        # Reuse existing act method logic
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_probs = self.old_actor(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob.item()

    def store_transition(self, state, action, action_log_prob, reward, next_state, done):
        # New: Store trajectory data
        self.states.append(state)
        self.actions.append(action)
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def compute_gae(self, next_value):
        # New: Compute GAE advantage estimation
        values = self.critic(torch.tensor(
            np.array(self.states), dtype=torch.float32)).detach().numpy().flatten()
        next_values = self.critic(torch.tensor(
            np.array(self.next_states), dtype=torch.float32)).detach().numpy().flatten()
        advantages = np.zeros_like(self.rewards)
        last_advantage = 0

        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * \
                next_values[t] * (1 - self.dones[t]) - values[t]
            last_advantage = delta + self.gamma * \
                self.gae_lambda * (1 - self.dones[t]) * last_advantage
            advantages[t] = last_advantage

        returns = advantages + values
        return advantages, returns

    def learn(self):
        # New: PPO batch update logic
        next_state = torch.tensor(
            self.next_states[-1], dtype=torch.float32).unsqueeze(0)
        next_value = self.critic(next_state).detach(
        ).item() if not self.dones[-1] else 0

        # Compute GAE advantages and returns
        advantages, returns = self.compute_gae(next_value)

        # 转换为张量
        states = torch.tensor(np.array(self.states), dtype=torch.float32)
        actions = torch.tensor(self.actions, dtype=torch.long)
        old_action_log_probs = torch.tensor(
            self.action_log_probs, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-8)

        # Multiple epochs updates
        for _ in range(self.update_epochs):
            # Randomly sample batches
            indices = torch.randperm(len(states))
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
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
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

    def print_policy(self, env):
        # Reuse existing print policy method
        action_symbols = ['↑', '→', '↓', '←']
        print("Optimal Policy:")
        for i in range(env.size):
            row = []
            for j in range(env.size):
                state_idx = i * env.size + j
                state = np.zeros(env.size * env.size)
                state[state_idx] = 1
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                self.actor.eval()
                with torch.no_grad():
                    action_probs = self.actor(state_tensor)
                action = torch.argmax(action_probs).item()
                row.append(action_symbols[action])
            print("\t".join(row))

# Adjust training function to adapt to PPO batch updates


def train_ppo_agent(env, agent, episodes=500, max_t=100, update_timestep=200):
    scores = []
    timestep = 0
    for i_episode in range(1, episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            timestep += 1
            action, action_log_prob = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(
                state, action, action_log_prob, reward, next_state, done)
            state = next_state
            score += reward

            # Update PPO agent every update_timestep steps
            if timestep % update_timestep == 0:
                agent.learn()
                timestep = 0

            if done:
                break
        scores.append(score)

        if i_episode % 100 == 0:
            print(
                f'Episode {i_episode}, Average Score: {np.mean(scores[-100:]):.2f}')

    return scores


def main():
    state_size = 16  # 4x4 grid
    action_size = 4   # up, right, down, left

    # Use the same environment as the original code
    class SimpleEnv:
        def __init__(self):
            self.size = 4
            self.reset()

        def reset(self):
            self.agent_pos = (0, 0)
            return self._get_state()

        def step(self, action):
            i, j = self.agent_pos
            if action == 0:
                i -= 1  # up
            elif action == 1:
                j += 1  # right
            elif action == 2:
                i += 1  # down
            elif action == 3:
                j -= 1  # left
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
    agent = PPOAgent(state_size, action_size)
    scores = train_ppo_agent(env, agent)

    print("Training completed! Printing optimal policy:")
    agent.print_policy(env)


if __name__ == '__main__':
    main()
