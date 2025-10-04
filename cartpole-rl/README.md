# CartPole RL with PPO

This project implements the Proximal Policy Optimization (PPO) algorithm to solve the classic CartPole reinforcement learning problem. The implementation uses PyTorch for neural network models and Gym for the environment simulation.

## Project Structure

- `actor_critic.py`: Contains the Actor and Critic neural network architectures
- `ppo_agent.py`: Implements the PPO agent with experience replay and policy optimization
- `main.py`: Main script for training and testing the agent with visualization
- `requirements.txt`: Required Python packages

## Features

- Implementation of Proximal Policy Optimization (PPO) algorithm
- Actor-Critic architecture for policy and value function approximation
- GAE (Generalized Advantage Estimation) for more stable training
- Visualization of the CartPole environment during testing
- Training progress tracking and score plotting
- Automatic model saving when the environment is solved

## Installation

```bash
cd cartpole-rl
pip install -r requirements.txt
```

## Usage

To train and test the PPO agent:

```bash
python main.py
```

This will:
1. Initialize the CartPole environment
2. Train the PPO agent until it solves the environment (average score >= 475 over 100 episodes)
3. Save the trained model weights
4. Run a demonstration of the trained agent with visualization
5. Generate a training scores plot

## Results

The agent typically solves the CartPole environment within 200-400 episodes. The training progress will be saved as `training_scores.png`, and the trained model weights will be saved as `ppo_actor.pth` and `ppo_critic.pth`.