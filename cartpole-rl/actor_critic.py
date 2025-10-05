import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64, device='cpu'):
        super(Actor, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=-1)
        # 将模型移到指定设备
        self.to(self.device)

    def forward(self, state):
        # 确保输入张量在正确的设备上
        state = state.to(self.device)
        x = torch.relu(self.fc1(state))
        action_probs = self.softmax(self.fc2(x))
        return action_probs

class Critic(nn.Module):
    def __init__(self, state_size, hidden_size=64, device='cpu'):
        super(Critic, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        # 将模型移到指定设备
        self.to(self.device)

    def forward(self, state):
        # 确保输入张量在正确的设备上
        state = state.to(self.device)
        x = torch.relu(self.fc1(state))
        value = self.fc2(x)
        return value