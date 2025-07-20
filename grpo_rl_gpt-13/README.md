# GRPO强化学习算法简化实现

这个项目提供了GRPO (Generalized Policy Optimization) 算法的简化实现，用于解释其核心原理并包含完整的训练和测试流程。

## 算法原理

GRPO是一种基于策略梯度的强化学习算法，主要特点包括：
- 使用GAE (Generalized Advantage Estimation) 进行优势估计
- 通过KL散度控制策略更新步长
- 自适应调整KL惩罚系数beta
- 分离的策略网络和价值网络

与PPO的剪裁方法不同，GRPO使用KL散度作为策略更新的约束，通过线性优势估计和KL惩罚来构建损失函数。

## 代码结构

- `main.py`: 包含完整的GRPO实现，包括：
  - 策略网络和价值网络定义
  - GRPO代理类（含GAE计算和策略更新）
  - 训练和测试函数
  - CartPole环境演示
- `requirements.txt`: 项目依赖

## 使用方法

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行训练和测试
```bash
python main.py
```

程序将自动进行500轮训练，然后进行10轮测试，并生成训练奖励曲线图。

## 结果说明

训练过程中会输出平均奖励、策略损失、价值损失和KL散度等指标。训练完成后会生成`training_rewards.png`文件，展示奖励随训练轮次的变化。

测试阶段会输出每轮测试的奖励值和平均奖励，用于评估训练后策略的性能。