# 简单Diffusion模型实现

这个项目提供了一个简洁的Diffusion模型实现，同时支持DDPM (Denoising Diffusion Probabilistic Models)和DDIM (Denoising Diffusion Implicit Models)两种采样方法，用于随机生成图片。模型设计力求简洁明了，主要用于解释Diffusion模型的基本工作机制。

## 模型特点
- 基于PyTorch实现，同时支持DDPM和DDIM两种采样方法
- 使用简化的U-Net架构作为骨干网络
- 支持小尺寸图片训练，默认32x32像素以实现快速训练
- 训练数据使用CIFAR-10数据集
- 支持GPU、MPS等加速设备
- 不包含文本控制功能，仅用于随机生成图片

## 实现细节

### 模型架构
1. **噪声调度器**：实现了线性噪声调度策略，控制噪声的添加过程，所有参数均移至指定设备以支持GPU/MPS加速
2. **U-Net模型**：简化版U-Net，包含下采样路径、中间层和上采样路径
3. **时间嵌入**：将时间步信息嵌入到特征图中，支持浮点类型输入以兼容MPS设备
4. **训练和采样**：实现了标准的DDPM训练循环，同时支持DDPM和DDIM两种采样方法
   - DDPM采样：原始的Diffusion采样方法，需要完整的时间步数
   - DDIM采样：优化的采样方法，可以用更少的步数生成图片

### 训练配置
- 图片尺寸：32x32像素
- 训练批次：64
- 训练周期：50
- 学习率：3e-4
- 扩散步骤：1000
- 噪声调度：beta从0.0001线性增加到0.02
- 设备支持：自动检测CUDA或MPS，默认为CPU

### 采样配置
- 采样方法：可选择'ddpm'或'ddim'
- DDIM采样步数：可配置（默认50步）
- DDIM eta参数：控制采样随机性（0.0表示确定性采样，1.0表示完全随机）

## 使用方法

### 环境要求
- PyTorch
- torchvision
- matplotlib
- numpy
- tqdm

### 训练模型
1. 确保安装了所有依赖项
2. 运行main.py文件开始训练：
   ```bash
   python main.py
   ```
3. 训练过程中，模型会定期保存，并在results目录下生成样本图片

### 生成图片
训练完成后，模型会自动生成最终样本并保存到results目录。根据采样方法不同，生成的文件会有所区别：
- DDPM采样：final_samples_ddpm.png
- DDIM采样：final_samples_ddim.png

### 配置参数
可以通过修改main.py中的config字典来调整模型性能、训练速度和采样方式：
```python
config = {
    'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
    'batch_size': 64,
    'img_size': 32,
    'num_epochs': 50,
    'learning_rate': 3e-4,
    'num_timesteps': 1000,
    'beta_start': 0.0001,
    'beta_end': 0.02,
    'sampling_method': 'ddim',  # 可选 'ddpm' 或 'ddim'
    'ddim_num_steps': 50,       # DDIM采样步数
    'ddim_eta': 0.0             # DDIM eta参数
}
```

## 结果解释
训练过程中，模型会逐渐学习如何从随机噪声中生成逼真的图片。每个epoch结束后，你可以在results目录下查看生成的样本图片，观察模型的学习进展。DDIM采样通常可以在更少的步骤内生成与DDPM质量相当的图片。

## 注意事项
- 为了实现快速训练，模型使用了小尺寸图片和简化的网络架构
- 训练时间取决于硬件配置，在GPU上通常需要几十分钟到几小时
- MPS设备用户：代码已针对MPS设备进行优化，解决了类型不匹配问题
- 可以通过修改config字典中的参数来调整模型性能、训练速度和采样方式

这个实现主要用于教学目的，帮助理解Diffusion模型的基本原理以及DDPM和DDIM两种采样方法的区别。如果需要更高质量的图片生成，可以增加图片尺寸、使用更复杂的网络架构或增加训练周期。