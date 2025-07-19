import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.table import Table
from matplotlib.font_manager import FontProperties
import csv

# 设置中文字体（macOS兼容）
plt.rcParams["font.family"] = ["Heiti TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 生成线性样本数据
x = torch.unsqueeze(torch.linspace(0, 1, 100), dim=1)  # shape (100, 1)
y = 2 * x + 3 + 0.2 * torch.randn(x.size())  # 简单线性函数 y=2x+3 加噪声

test_x = x
test_y = y

class UnifiedLinearModel(nn.Module):
    def __init__(self, use_residual=False):
        super(UnifiedLinearModel, self).__init__()
        # 定义5个线性层
        self.linear1 = nn.Linear(1, 1)
        self.linear2 = nn.Linear(1, 1)
        self.linear3 = nn.Linear(1, 1)
        self.linear4 = nn.Linear(1, 1)
        self.linear5 = nn.Linear(1, 1)
        self.linear6 = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
        self.use_residual = use_residual
        
        # 所有层权重初始化为正态分布(0,1)，偏置初始化为0
        for m in [self.linear1, self.linear2, self.linear3, self.linear4, self.linear5, self.linear6]:
            nn.init.normal_(m.weight, mean=0.0, std=1.0)
            nn.init.zeros_(m.bias)
                
    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.sigmoid(x1)
        x2 = self.linear2(x1)
        x2 = self.sigmoid(x2)
        x3 = self.linear3(x2)
        x3 = self.sigmoid(x3)
        x4 = self.linear4(x3)
        x4 = self.sigmoid(x4)
        x5 = self.linear5(x4)
        x5 = self.sigmoid(x5)
        if self.use_residual:
            x5 = x2 + x5  # 残差连接：第二个linear到最后一个linear前
        x6 = self.linear6(x5)
        return x6

# 提取模型参数
def get_parameters(model):
    params = {}
    for name, param in model.named_parameters():
        params[name] = param.item()
    return params

# 初始化两个模型
residual_model = UnifiedLinearModel(use_residual=True)
no_residual_model = UnifiedLinearModel(use_residual=False)

# 保存初始权重
residual_initial_params = get_parameters(residual_model)
no_residual_initial_params = get_parameters(no_residual_model)

# 优化器和损失函数
criterion = nn.MSELoss()
residual_optimizer = torch.optim.SGD(residual_model.parameters(), lr=0.2)
no_residual_optimizer = torch.optim.SGD(no_residual_model.parameters(), lr=0.2)

# 训练两个模型并记录损失
epochs = 1000
residual_losses = []
no_residual_losses = []

print('开始训练...')
for epoch in range(epochs):
    # 训练残差模型
    residual_pred = residual_model(x)
    residual_loss = criterion(residual_pred, y)
    residual_optimizer.zero_grad()
    residual_loss.backward()
    residual_optimizer.step()
    residual_losses.append(residual_loss.item())
    
    # 训练非残差模型
    no_residual_pred = no_residual_model(x)
    no_residual_loss = criterion(no_residual_pred, y)
    no_residual_optimizer.zero_grad()
    no_residual_loss.backward()
    no_residual_optimizer.step()
    no_residual_losses.append(no_residual_loss.item())
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], ' +
              f'Residual Loss: {residual_loss.item():.4f}, ' +
              f'No Residual Loss: {no_residual_loss.item():.4f}')

# 计算权重变化量
def calculate_weight_changes(initial_params, final_params):
    changes = {}
    for name in initial_params:
        # 同时考虑权重和偏置的变化
        changes[name] = abs(final_params[name] - initial_params[name])
    return changes

residual_params = get_parameters(residual_model)
no_residual_params = get_parameters(no_residual_model)

# 计算权重变化量
residual_changes = calculate_weight_changes(residual_initial_params, residual_params)
no_residual_changes = calculate_weight_changes(no_residual_initial_params, no_residual_params)

# 绘制Loss对比曲线
plt.figure(figsize=(10, 6))
plt.plot(residual_losses, label='带残差连接')
plt.plot(no_residual_losses, label='无残差连接')
plt.title('训练损失对比')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('loss_comparison.png')
plt.close()

# 绘制权重变化量对比图
plt.figure(figsize=(14, 6))
layers = ['linear1.weight', 'linear1.bias', 'linear2.weight', 'linear2.bias',
          'linear3.weight', 'linear3.bias', 'linear4.weight', 'linear4.bias',
          'linear5.weight', 'linear5.bias', 'linear6.weight', 'linear6.bias']
residual_values = [residual_changes[layer] for layer in layers]
no_residual_values = [no_residual_changes[layer] for layer in layers]

x = np.arange(len(layers))
width = 0.35

plt.bar(x - width/2, residual_values, width, label='带残差连接', color='skyblue')
plt.bar(x + width/2, no_residual_values, width, label='无残差连接', color='lightgreen')

plt.xlabel('参数名称')
plt.ylabel('变化量 (绝对值)')
plt.title('各层权重和偏置变化量对比')
plt.xticks(x, layers, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('weight_change_comparison.png')
plt.close()

# 绘制模型预测对比图
plt.figure(figsize=(10, 6))
plt.scatter(test_x.data.numpy(), test_y.data.numpy(), label='真实数据')
plt.plot(test_x.data.numpy(), residual_model(test_x).data.numpy(), 'r-', lw=3, label='带残差连接')
plt.plot(test_x.data.numpy(), no_residual_model(test_x).data.numpy(), 'b--', lw=3, label='无残差连接')
plt.title('模型预测对比')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('prediction_comparison.png')
plt.close()

# 绘制权重对比表格
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

# 表格数据
layers = ['linear1.weight', 'linear1.bias', 'linear2.weight', 'linear2.bias',
          'linear3.weight', 'linear3.bias', 'linear4.weight', 'linear4.bias',
          'linear5.weight', 'linear5.bias', 'linear6.weight', 'linear6.bias']

# 创建表格数据
table_data = [['层名称', '带残差连接', '无残差连接']]
for layer in layers:
    table_data.append([
        layer,
        f'{residual_params[layer]:.6f}',
        f'{no_residual_params[layer]:.6f}'
    ])

# 创建表格字体属性
font = FontProperties(family=['Heiti TC', 'Arial Unicode MS'], size=10)

# 创建表格
table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.3, 0.35, 0.35]) 
table.auto_set_font_size(False)
table.set_fontsize(10)
table.set_fontsize(10)
table.scale(1.2, 1.5)  # 调整表格大小

plt.title('模型权重对比表', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('weight_comparison_table.png')
plt.close()

print('训练完成！生成的对比图表：')
print('- loss_comparison.png: 损失对比曲线')
print('- prediction_comparison.png: 模型预测对比')
print('- weight_comparison_table.png: 权重对比表格')
print('- model_architecture.png: 模型网络结构图')
print('- weight_change_comparison.png: 权重变化量对比图')