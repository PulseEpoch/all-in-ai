import numpy as np

class ValueIterationAgent:
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env  # Environment reference
        self.gamma = gamma  # Discount factor
        self.theta = theta  # Convergence threshold
        self.num_states = env.size * env.size  # Total number of states
        self.num_actions = 4  # Total number of actions: up, right, down, left
        self.V = np.zeros(self.num_states)  # State value function
        self.policy = np.zeros(self.num_states, dtype=int)  # Policy

    def value_iteration(self, max_iterations=1000):
        """Perform value iteration algorithm to update V values until convergence or max iterations"""
        iteration = 0
        while iteration < max_iterations:
            delta = 0
            # Iterate through all states
            for state in range(self.num_states):
                v = self.V[state]
                # Calculate Q-values for all possible actions
                q_values = np.zeros(self.num_actions)
                for action in range(self.num_actions):
                    # Simulate taking the action
                    pos = self.env._state_to_pos(state)
                    original_pos = self.env.agent_pos
                    self.env.agent_pos = pos
                    next_state, reward, _ = self.env.step(action)
                    # Restore environment state
                    self.env.agent_pos = original_pos
                    # Calculate Q-value
                    q_values[action] = reward + self.gamma * self.V[next_state]
                # Update V value to the maximum Q-value
                self.V[state] = np.max(q_values)
                delta = max(delta, np.abs(v - self.V[state]))
            # Print training progress
            print(f"Value Iteration - Iteration: {iteration}, Delta: {delta:.6f}")
            # Check for convergence
            if delta < self.theta:
                print(f"Converged after {iteration+1} iterations")
                break
            iteration += 1
        else:
            print(f"Reached maximum iterations ({max_iterations}) without convergence")
        # Extract optimal policy
        self._extract_policy()
        return self.V, self.policy

    def _extract_policy(self):
        """Extract optimal policy from converged V values"""
        for state in range(self.num_states):
            q_values = np.zeros(self.num_actions)
            for action in range(self.num_actions):
                # Simulate taking the action
                pos = self.env._state_to_pos(state)
                original_pos = self.env.agent_pos
                self.env.agent_pos = pos
                next_state, reward, _ = self.env.step(action)
                # Restore environment state
                self.env.agent_pos = original_pos
                q_values[action] = reward + self.gamma * self.V[next_state]
            # Select the action with the maximum Q-value as the policy
            self.policy[state] = np.argmax(q_values)

    def get_action(self, state):
        """Get action according to current policy"""
        return self.policy[state]

    def print_value_function(self):
        """Print state value function"""
        print("State Value Function V:")
        for i in range(self.env.size):
            row = []
            for j in range(self.env.size):
                state = self.env._pos_to_state((i, j))
                row.append(f"{self.V[state]:.2f}")
            print("\t".join(row))

    def print_policy(self, file=None):
        """Print policy to console or file"""
        action_symbols = ['↑', '→', '↓', '←']  # up, right, down, left
        print("Optimal Policy:", file=file)
        for i in range(self.env.size):
            row = []
            for j in range(self.env.size):
                state = self.env._pos_to_state((i, j))
                row.append(action_symbols[self.policy[state]])
            print("\t".join(row), file=file)

    def train(self, max_iterations=1000, save_path=None):
        """Main training method that runs value iteration and optionally saves the model"""
        print("Starting training...")
        self.value_iteration(max_iterations)
        print("Training completed!")
        
        # Print optimal policy
        self.print_policy()
        
        if save_path:
            self.save_model(save_path)
            # Save human-readable policy
            policy_text_path = save_path.replace('.npz', '_policy.txt')
            with open(policy_text_path, 'w') as f:
                self.print_policy(file=f)
            print(f"Model saved to {save_path}")
            print(f"Policy saved to {policy_text_path}")
        
        return self.V, self.policy

    def save_model(self, path):
        """Save the value function and policy to a numpy archive"""
        np.savez(path, V=self.V, policy=self.policy)


class SimpleEnv:
    """简单的网格环境，用于演示ValueIterationAgent"""
    def __init__(self, size=4):
        self.size = size
        self.agent_pos = (0, 0)  # 初始位置
    
    def _state_to_pos(self, state):
        """将状态转换为位置坐标"""
        return (state // self.size, state % self.size)
    
    def _pos_to_state(self, pos):
        """将位置坐标转换为状态"""
        return pos[0] * self.size + pos[1]
    
    def step(self, action):
        """执行动作，返回下一个状态、奖励和是否结束"""
        current_pos = self.agent_pos
        # 根据动作更新位置
        if action == 0:  # 上
            new_pos = (max(0, current_pos[0]-1), current_pos[1])
        elif action == 1:  # 右
            new_pos = (current_pos[0], min(self.size-1, current_pos[1]+1))
        elif action == 2:  # 下
            new_pos = (min(self.size-1, current_pos[0]+1), current_pos[1])
        elif action == 3:  # 左
            new_pos = (current_pos[0], max(0, current_pos[1]-1))
        else:
            new_pos = current_pos
        
        self.agent_pos = new_pos
        next_state = self._pos_to_state(new_pos)
        reward = 1 if new_pos == (self.size-1, self.size-1) else -0.1  # 到达终点奖励1，否则-0.1
        done = (new_pos == (self.size-1, self.size-1))
        
        return next_state, reward, done

if __name__ == "__main__":
    # create a simple grid(4x4) world environment for demonstrating ValueIterationAgent
    env = SimpleEnv(size=4)
    # Initialize and train ValueIterationAgent
    agent = ValueIterationAgent(env, gamma=0.9, theta=1e-6)
    # train and output the optimal policy
    agent.train(max_iterations=1000)