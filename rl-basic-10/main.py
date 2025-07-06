import numpy as np
import random

# Simple Q-learning implementation for 4x4 grid world
class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=1.0):
        self.q_table = np.zeros((state_size, action_size))
        self.lr = learning_rate        # Learning rate
        self.gamma = discount_factor   # Discount factor for future rewards
        self.epsilon = epsilon         # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.q_table.shape[1]))  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def learn(self, state, action, reward, next_state):
        # Q-learning update rule
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
        self.q_table[state, action] = new_value

        # Decay epsilon
        # Adventure rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 4x4 Grid World Environment with exit
class GridWorldEnvironment:
    def __init__(self, size=4):
        self.size = size
        self.grid = [[0 for _ in range(size)] for _ in range(size)]
        self.start_pos = (0, 0)  # Top-left corner
        self.goal_pos = (size-1, size-1)  # Bottom-right corner (exit)
        self.reset()

    def reset(self):
        self.agent_pos = self.start_pos
        return self._pos_to_state(self.agent_pos)

    # 2D to 1D state conversion
    def _pos_to_state(self, pos):
        return pos[0] * self.size + pos[1]

    # 1D to 2D state conversion
    def _state_to_pos(self, state):
        return (state // self.size, state % self.size)

    def step(self, action):
        # Actions: 0=Up, 1=Right, 2=Down, 3=Left
        row, col = self.agent_pos

        # Move agent with boundary checking
        if action == 0 and row > 0:  # Up
            row -= 1
        elif action == 1 and col < self.size - 1:  # Right
            col += 1
        elif action == 2 and row < self.size - 1:  # Down
            row += 1
        elif action == 3 and col > 0:  # Left
            col -= 1

        self.agent_pos = (row, col)
        state = self._pos_to_state(self.agent_pos)
        done = (self.agent_pos == self.goal_pos)

        # Reward structure
        if done:
            reward = 100  # Large reward for finding exit
        else:
            reward = -1  # Small penalty for each step

        return state, reward, done

    def render(self):
        # Simple text visualization of grid world
        grid_str = "\n"
        for i in range(self.size):
            row_str = ""
            for j in range(self.size):
                if (i, j) == self.agent_pos:
                    row_str += " A "  # Agent
                elif (i, j) == self.goal_pos:
                    row_str += " G "  # Goal (Exit)
                else:
                    row_str += " . "  # Empty cell
            grid_str += row_str + "\n"
        print(grid_str)

# Training loop
if __name__ == "__main__":
    env = GridWorldEnvironment(size=4)  # 4x4 grid
    agent = QLearningAgent(state_size=env.size*env.size, action_size=4)  # 4 actions
    episodes = 500

    print("Training a Q-learning agent to find the exit in a 4x4 grid...")
    print("A = Agent, G = Goal (Exit)")

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 100:  # Prevent infinite loops
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            steps += 1

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"\nEpisode: {episode+1}, Epsilon: {agent.epsilon:.2f}, Total Reward: {total_reward}")
            env.render()  # Show final position

    print("\nTraining complete!\nFinal Q-table:\n", agent.q_table)
    print("\nExample path from start to exit:")
    state = env.reset()
    env.render()
    done = False
    while not done:
        action = np.argmax(agent.q_table[state])
        state, _, done = env.step(action)
        env.render()
