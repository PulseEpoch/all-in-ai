# Core Content Explanation of main.py

## Overview
This file implements a Q-learning algorithm for an agent navigating a 4x4 grid world to reach a goal state throughout reinforcement learning.

## Key Components

### 1. GridWorldEnvironment
- Creates a 4x4 grid with an agent (A) and a goal (G)
- Handles agent movement and state transitions
- Provides reward system and rendering capability

### 2. QLearningAgent
- Implements Q-learning algorithm with an exploration-exploitation strategy (epsilon-greedy)
- Maintains a Q-table to store state-action values
- Updates Q-values using the Bellman equation

### 3. Training Process
- Runs for 500 episodes
- Resets environment at start of each episode
- Agent selects actions, receives rewards, and updates Q-table
- Prints progress every 100 episodes with current epsilon and total reward

### 4. Results
- Displays final Q-table after training
- Demonstrates an example path from start to goal using learned policy

## Key Variables
- `epsilon`: Controls exploration vs exploitation (starts high, decreases over time)
- `alpha`: Learning rate
- `gamma`: Discount factor for future rewards
- `q_table`: Stores state-action values for decision making