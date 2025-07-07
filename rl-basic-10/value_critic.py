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

    def value_iteration(self):
        """Perform value iteration algorithm to update V values until convergence"""
        while True:
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
            # Check for convergence
            if delta < self.theta:
                break
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

    def print_policy(self):
        """Print policy"""
        action_symbols = ['↑', '→', '↓', '←']  # up, right, down, left
        print("Optimal Policy:")
        for i in range(self.env.size):
            row = []
            for j in range(self.env.size):
                state = self.env._pos_to_state((i, j))
                row.append(action_symbols[self.policy[state]])
            print("\t".join(row))