import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

# Ensure data directory exists
os.makedirs('../data', exist_ok=True)

# Data conversion
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load data set
train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Teacher model - CNN (from mnist_classifier-cnn-04)
class TeacherCNN(nn.Module):
    def __init__(self):
        super(TeacherCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 50 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Student model - Very Simple Linear Model
class StudentLinear(nn.Module):
    def __init__(self):
        super(StudentLinear, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Distillation Loss Function
class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=1)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, student_output, teacher_output, labels):
        # Soften teacher output
        soft_teacher_output = self.softmax(teacher_output / self.temperature)
        # Soften student output and calculate KL divergence
        soft_student_output = nn.LogSoftmax(dim=1)(student_output / self.temperature)
        distillation_loss = nn.KLDivLoss()(soft_student_output, soft_teacher_output) * (self.temperature ** 2)
        # Student hard loss
        student_loss = self.cross_entropy(student_output, labels)
        # Combine losses
        return self.alpha * distillation_loss + (1 - self.alpha) * student_loss

# Initialize models
teacher_model = TeacherCNN()
student_distill_model = StudentLinear()  # used for distillation training
student_alone_model = StudentLinear()   # used for standalone training

# Teacher model training function
def train_teacher(epochs=10):
    teacher_model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(teacher_model.parameters(), lr=0.01, momentum=0.5)
    teacher_losses = []
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = teacher_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f'Teacher Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        avg_loss = total_loss / len(train_loader)
        teacher_losses.append(avg_loss)
        print(f'==> Teacher Train Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return teacher_model, teacher_losses

# Define loss functions and optimizers
distill_criterion = DistillationLoss(temperature=1.5, alpha=0.3)
alone_criterion = nn.CrossEntropyLoss()

distill_optimizer = optim.SGD(student_distill_model.parameters(), lr=0.01, momentum=0.5)
alone_optimizer = optim.SGD(student_alone_model.parameters(), lr=0.01, momentum=0.5)

# Distillation training function
def train_distill(epochs, model, teacher_model, criterion, optimizer, train_loader, device):
    model.train()
    teacher_model.eval()  # Teacher is fixed
    distill_losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            student_output = model(data)
            with torch.no_grad():
                teacher_output = teacher_model(data)
            loss = criterion(student_output, teacher_output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f'Distill Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        avg_loss = total_loss / len(train_loader)
        distill_losses.append(avg_loss)
        print(f'==> Distill Train Epoch: {epoch+1} Average loss: {avg_loss:.4f}')
    return model, distill_losses

# Standalone training function
def train_alone(epochs, model, criterion, optimizer, train_loader, device):
    model.train()
    alone_losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f'Alone Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        avg_loss = total_loss / len(train_loader)
        alone_losses.append(avg_loss)
        print(f'==> Alone Train Epoch: {epoch+1} Average loss: {avg_loss:.4f}')
    return model, alone_losses

# Test function
def test(model, test_loader, device, model_name):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'{model_name} Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy

# Main function
if __name__ == '__main__':
    print("===== Starting Teacher Model Training ======")
    epochs = 5

    # Training teacher model
    teacher_model, teacher_losses = train_teacher(epochs=epochs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    teacher_model.to(device)
    teacher_accuracy = test(teacher_model, test_loader, device, "Teacher")

    student_distill_model.to(device)
    student_alone_model.to(device)
    # Training distilled student model
    print("===== Starting Distillation Training ======")
    student_distill_model, distill_losses = train_distill(epochs=epochs, model=student_distill_model, teacher_model=teacher_model, criterion=distill_criterion, optimizer=distill_optimizer, train_loader=train_loader, device=device)
    distill_accuracy = test(student_distill_model, test_loader, device, "Distillation Student")

    # Training standalone student model with same dataset
    print("\n===== Starting Standalone Training ======")
    student_alone_model, alone_losses = train_alone(epochs=epochs, model=student_alone_model, criterion=alone_criterion, optimizer=alone_optimizer, train_loader=train_loader, device=device)
    alone_accuracy = test(student_alone_model, test_loader, device, "Standalone Student")
    
    # save model weights
    torch.save(teacher_model.state_dict(), 'teacher_weights.pth')
    torch.save(student_distill_model.state_dict(), 'student_distill_weights.pth')
    torch.save(student_alone_model.state_dict(), 'student_alone_weights.pth')
    
    #  Compare accuracy
    print("\n===== Accuracy Comparison ======")
    print(f"Teacher Model Accuracy: {teacher_accuracy:.2f}%")
    print(f"Distillation Student Accuracy: {distill_accuracy:.2f}%")
    print(f"Standalone Student Accuracy: {alone_accuracy:.2f}%")
    print(f"Difference: {distill_accuracy - alone_accuracy:.2f}%")
    print("=================================")
    
    print('Model weights saved as teacher_weights.pth, student_distill_weights.pth and student_alone_weights.pth')
    
    # Plot loss comparison
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), teacher_losses, 'b-', marker='o', label='Teacher Model')
    plt.plot(range(1, epochs+1), distill_losses, 'r-', marker='s', label='Distillation Student')
    plt.plot(range(1, epochs+1), alone_losses, 'g-', marker='^', label='Standalone Student')
    plt.xlabel('Training Epoch')
    plt.ylabel('Average Loss')
    plt.title('Comparison of Training Losses')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('loss_comparison.png')
    plt.close()
    print('Loss comparison plot saved as loss_comparison.png')