import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 配置参数
config = {
    'image_size': 32,  # 小尺寸图片以实现快速训练
    'channels': 3,     # RGB图片
    'batch_size': 64,
    'epochs': 50,
    'lr': 3e-4,
    'num_timesteps': 100,  # 扩散步骤
    'beta_start': 0.0001,
    'beta_end': 0.02,
    'device': 'cuda' if torch.cuda.is_available() else 'mps',
    'sampling_method': 'ddpm',  # 可选 'ddpm' 或 'ddim'
    'ddim_num_steps': 50,       # DDIM采样步数
    'ddim_eta': 0.0             # DDIM方差控制参数
}

# 确保输出目录存在
os.makedirs('results', exist_ok=True)

# 准备数据集 (使用CIFAR-10)
transform = transforms.Compose([
    transforms.Resize(config['image_size']),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(
    train_dataset, batch_size=config['batch_size'], shuffle=True)

# 定义U-Net模型的基本块
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        if up:
            self.conv1 = nn.Conv2d(2 * in_channels, out_channels, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.transform = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.bnorm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # 第一个卷积块
        h = self.bnorm1(self.relu(self.conv1(x)))
        # 时间嵌入
        time_emb = self.relu(self.time_mlp(t))
        # 广播时间嵌入到特征图
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        # 第二个卷积块
        h = self.bnorm2(self.relu(self.conv2(h)))
        # 下采样或上采样
        h = self.transform(h)
        return h

# 定义简单的U-Net模型
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.time_emb_dim = 32

        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.time_emb_dim),
            nn.ReLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

        # 下采样路径
        self.down1 = Block(in_channels, 64, self.time_emb_dim)
        self.down2 = Block(64, 128, self.time_emb_dim)
        self.down3 = Block(128, 256, self.time_emb_dim)

        # 中间层
        self.mid_conv = nn.Conv2d(256, 256, 3, padding=1)

        # 上采样路径
        self.up1 = Block(256, 128, self.time_emb_dim, up=True)
        self.up2 = Block(128, 64, self.time_emb_dim, up=True)
        self.up3 = Block(64, 64, self.time_emb_dim, up=True)

        # 输出层
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, t):
        # 时间嵌入
        t = t.unsqueeze(1)
        t = t.float()  # 转换为浮点类型以支持MPS设备
        t = self.time_mlp(t)

        # 下采样
        x1 = self.down1(x, t)
        x2 = self.down2(x1, t)
        x3 = self.down3(x2, t)

        # 中间层
        x_mid = self.mid_conv(x3)

        # 上采样
        x = self.up1(torch.cat([x_mid, x3], dim=1), t)
        x = self.up2(torch.cat([x, x2], dim=1), t)
        x = self.up3(torch.cat([x, x1], dim=1), t)

        # 输出
        return self.out(x)

# 定义噪声调度器
class NoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # 线性调度beta并移到指定设备
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(config['device'])
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # 确保所有参数都在正确的设备上
        self.alphas = self.alphas.to(config['device'])
        self.alphas_cumprod = self.alphas_cumprod.to(config['device'])
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(config['device'])
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(config['device'])

    def add_noise(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.num_timesteps, (batch_size,), device=config['device'])

# 定义DDPM模型
class DDPM:
    def __init__(self, model, noise_scheduler):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        self.loss_fn = nn.MSELoss()

    def train_step(self, x):
        self.model.train()
        self.optimizer.zero_grad()

        # 采样时间步
        t = self.noise_scheduler.sample_timesteps(x.shape[0]).to(config['device'])

        # 添加噪声
        x_noisy, noise = self.noise_scheduler.add_noise(x, t)

        # 预测噪声
        noise_pred = self.model(x_noisy, t)

        # 计算损失
        loss = self.loss_fn(noise_pred, noise)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def sample(self, batch_size, method=None, num_steps=None, eta=None):
        # 使用配置参数或默认值
        method = method or config['sampling_method']
        num_steps = num_steps or (config['ddim_num_steps'] if method == 'ddim' else config['num_timesteps'])
        eta = eta or config['ddim_eta']

        if method == 'ddim':
            return self.ddim_sample(batch_size, num_steps, eta)
        else:
            return self.ddpm_sample(batch_size)

    def ddpm_sample(self, batch_size):
        self.model.eval()

        # 从随机噪声开始
        x = torch.randn((batch_size, config['channels'], config['image_size'], config['image_size']),
                        device=config['device'])

        # 逐步去噪
        for t in reversed(range(0, self.noise_scheduler.num_timesteps)):
            with torch.no_grad():
                # 预测噪声
                noise_pred = self.model(x, torch.tensor([t] * batch_size, device=config['device']))

                # 计算当前alpha和beta
                beta_t = self.noise_scheduler.betas[t]
                alpha_t = self.noise_scheduler.alphas[t]
                alpha_cumprod_t = self.noise_scheduler.alphas_cumprod[t]

                # 计算均值和方差
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                # 去噪步骤
                x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * noise_pred) + torch.sqrt(beta_t) * noise

        # 归一化到[0, 1]
        x = (x.clamp(-1, 1) + 1) / 2
        return x

    def ddim_sample(self, batch_size, num_steps, eta):
        self.model.eval()

        # 计算DDIM采样步骤
        skip = self.noise_scheduler.num_timesteps // num_steps
        timesteps = list(range(0, self.noise_scheduler.num_timesteps, skip))[::-1]

        # 从随机噪声开始
        x = torch.randn((batch_size, config['channels'], config['image_size'], config['image_size']),
                        device=config['device'])

        for i, t in enumerate(timesteps):
            with torch.no_grad():
                # 预测噪声
                noise_pred = self.model(x, torch.tensor([t] * batch_size, device=config['device']))

                # 获取当前和下一个时间步的alpha累积乘积
                alpha_cumprod_t = self.noise_scheduler.alphas_cumprod[t]
                if i < len(timesteps) - 1:
                    t_next = timesteps[i+1]
                    alpha_cumprod_t_next = self.noise_scheduler.alphas_cumprod[t_next]
                else:
                    alpha_cumprod_t_next = torch.tensor(1.0, device=config['device'])

                # 计算DDIM步骤
                sigma = eta * torch.sqrt((1 - alpha_cumprod_t_next) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_next))
                sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)

                # 计算预测的原始图像
                x0_pred = (x - sqrt_one_minus_alpha_cumprod_t * noise_pred) / torch.sqrt(alpha_cumprod_t)
                x0_pred = torch.clamp(x0_pred, -1, 1)

                # 计算方向项
                dir_xt = torch.sqrt(1 - alpha_cumprod_t_next - sigma**2) * noise_pred

                # 随机噪声项
                if i < len(timesteps) - 1:
                    noise = torch.randn_like(x)
                    x = torch.sqrt(alpha_cumprod_t_next) * x0_pred + dir_xt + sigma * noise
                else:
                    # 最后一步不需要噪声
                    x = torch.sqrt(alpha_cumprod_t_next) * x0_pred + dir_xt

        # 归一化到[0, 1]
        x = (x.clamp(-1, 1) + 1) / 2
        return x

# 寻找最近的checkpoint

def find_latest_checkpoint():
    checkpoint_dir = 'results'
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    # 查找所有model_epoch_*.pth文件
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('model_epoch_') and filename.endswith('.pth'):
            try:
                epoch = int(filename.split('_')[-1].split('.')[0])
                checkpoints.append((epoch, os.path.join(checkpoint_dir, filename)))
            except ValueError:
                continue
    
    if not checkpoints:
        return None, 0
    
    # 按epoch排序，返回最新的
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1], checkpoints[0][0]

# 训练函数

def train(ddpm, train_loader, epochs, start_epoch=0):
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in progress_bar:
            images, _ = batch
            images = images.to(config['device'])

            # 训练步骤
            loss = ddpm.train_step(images)
            total_loss += loss

            # 更新进度条
            progress_bar.set_postfix({'loss': loss})

        # 打印平均损失
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {start_epoch + epoch + 1}/{start_epoch + epochs}, Average Loss: {avg_loss:.4f}')
        
        # 保存模型
        if (epoch + 1) % 10 == 0:
            torch.save(ddpm.model.state_dict(), f'results/model_epoch_{start_epoch + epoch + 1}.pth')

        # 每个epoch结束后生成一些样本
        if (epoch + 1) % 10 == 0:
            # 使用配置的采样方法生成样本
            samples = ddpm.sample(
                8,
                method=config['sampling_method'],
                num_steps=config['ddim_num_steps'] if config['sampling_method'] == 'ddim' else None,
                eta=config['ddim_eta'] if config['sampling_method'] == 'ddim' else None
            )
            # 保存样本图片
            grid = torchvision.utils.make_grid(samples, nrow=4)
            plt.figure(figsize=(10, 10))
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.axis('off')
            plt.savefig(f'results/samples_epoch_{epoch+1}_{config["sampling_method"]}.png')
            plt.close()

            # 保存模型
            torch.save(ddpm.model.state_dict(), f'results/model_epoch_{start_epoch + epoch + 1}.pth')

# 主函数
def main():
    # 初始化模型和噪声调度器
    model = SimpleUNet(in_channels=config['channels'], out_channels=config['channels']).to(config['device'])
    noise_scheduler = NoiseScheduler(
        num_timesteps=config['num_timesteps'],
        beta_start=config['beta_start'],
        beta_end=config['beta_end']
    )

    # 初始化DDPM
    ddpm = DDPM(model, noise_scheduler)

    # 查找最近的checkpoint
    checkpoint_path, start_epoch = find_latest_checkpoint()
    if checkpoint_path:
        print(f'Found checkpoint: {checkpoint_path}, starting from epoch {start_epoch+1}')
        ddpm.model.load_state_dict(torch.load(checkpoint_path, map_location=config['device']))
    else:
        print('No checkpoint found, starting from scratch')
        start_epoch = 0

    # 训练模型
    print(f'Starting training on {config["device"]}...')
    train(ddpm, train_loader, config['epochs'], start_epoch)

    # 生成最终样本
    print(f'Generating final samples with {config["sampling_method"]}...')
    final_samples = ddpm.sample(
        16,
        method=config['sampling_method'],
        num_steps=config['ddim_num_steps'] if config['sampling_method'] == 'ddim' else None,
        eta=config['ddim_eta'] if config['sampling_method'] == 'ddim' else None
    )
    grid = torchvision.utils.make_grid(final_samples, nrow=4)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.savefig(f'results/final_samples_{config["sampling_method"]}.png')
    plt.close()

    # 如果需要，也可以用另一种方法生成样本进行比较
    if config['sampling_method'] == 'ddpm':
        print('Generating comparison samples with ddim...')
        ddim_samples = ddpm.sample(16, method='ddim')
        grid = torchvision.utils.make_grid(ddim_samples, nrow=4)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.savefig('results/final_samples_ddim.png')
        plt.close()
    else:
        print('Generating comparison samples with ddpm...')
        ddpm_samples = ddpm.sample(16, method='ddpm')
        grid = torchvision.utils.make_grid(ddpm_samples, nrow=4)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.savefig('results/final_samples_ddpm.png')
        plt.close()

if __name__ == '__main__':
    main()