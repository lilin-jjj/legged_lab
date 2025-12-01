# G1深度编码器训练指南

本文档介绍如何使用改进的深度编码器训练G1双足机器人。该实现参考了extreme-parkour的深度图像处理方式，并针对双足机器人进行了适配。

## 主要改进

### 1. 深度图像处理架构

与原始的直接展平深度图像不同，新的实现使用了卷积神经网络来提取深度特征：

```python
# 原始方式：直接展平
depth_flat = depth_image.view(batch_size, -1)  # (batch_size, H*W)

# 改进方式：使用CNN提取特征
depth_features = depth_encoder(depth_image)  # (batch_size, 32)
```

### 2. 深度编码器结构

#### DepthOnlyFCBackbone
基础深度图像编码器，使用卷积层提取特征：

```
输入: (batch_size, 64, 64)
  ↓ Conv2d(1→32, k=5)
  ↓ MaxPool2d(k=2, s=2)
  ↓ ELU
  ↓ Conv2d(32→64, k=3)
  ↓ ELU
  ↓ Flatten
  ↓ Linear(50176→128)
  ↓ ELU
  ↓ Linear(128→32)
输出: (batch_size, 32)
```

#### RecurrentDepthBackbone
循环深度编码器，融合深度特征和本体感知信息：

```
深度图像 → DepthOnlyFCBackbone → 32维特征
                                    ↓
本体感知信息 ────────────────────→ Concat
                                    ↓
                              Combination MLP (32+n_proprio → 128 → 32)
                                    ↓
                                  GRU (512)
                                    ↓
                              Output MLP (512 → 32)
                                    ↓
                              深度特征 (32维)
```

**注意**：与extreme-parkour不同，双足机器人版本不输出偏航角(yaw)，因为双足机器人不需要这个信息。

### 3. 训练流程（Teacher-Student架构）

训练采用**蒸馏学习**方式，分为两个阶段：

#### 阶段1：标准PPO训练（Teacher网络）
使用完整观测（包括展平的深度图像4096维）训练主策略网络。

**Teacher观测**：
```
[本体感知(43维), 展平深度图像(4096维), 其他观测] → Teacher Actor → 动作
```

#### 阶段2：深度编码器蒸馏（Student网络）
定期进行深度编码器和深度actor的蒸馏训练：

**Student观测**：
```
深度图像(64x64) → 深度编码器 → 深度特征(32维)
[本体感知(43维), 深度特征(32维), 其他观测] → Student Actor → 动作
```

**训练步骤**：

1. **收集数据**：
   ```python
   # Teacher：使用完整观测（4096维深度图像）
   obs_teacher = [proprio, depth_image_flat, ...]
   actions_teacher = teacher_actor(obs_teacher)
   
   # Student：使用深度特征（32维）
   depth_features = depth_encoder(depth_image, proprio)
   obs_student = [proprio, depth_features, ...]
   actions_student = student_actor(obs_student)
   ```

2. **更新深度网络**：
   ```python
   # 深度actor损失：让student模仿teacher
   loss = ||actions_teacher - actions_student||²_2
   
   # 反向传播，同时更新depth_encoder和depth_actor
   loss.backward()
   optimizer.step()
   ```

3. **分离隐藏状态**：
   ```python
   depth_encoder.detach_hidden_states()
   ```

**关键点**：
- **Teacher网络**：主策略，使用高维观测（4096维深度图像）
- **Student网络**：深度策略，使用低维观测（32维深度特征）
- **目标**：让Student学会用低维特征达到与Teacher相同的性能
- **优势**：部署时只需Student网络，推理速度更快，观测维度更低

## 使用方法

### 1. 训练

#### 使用改进的训练脚本（推荐）

```bash
python /home/ymzz-tec/code/legged_lab/scripts/rsl_rl/train_g1_depth_improved.py \
  --task=LeggedLab-Isaac-Velocity-Rough-G1-Depth-v0 \
  --headless \
  --enable_cameras \
  --max_iterations 130000 \
  --num_envs 128 \
  --depth_training_interval 10 \
  --depth_steps_per_env 24
```

参数说明：
- `--depth_training_interval`: 每隔多少次主训练迭代进行一次深度编码器训练（默认10）
- `--depth_steps_per_env`: 每次深度编码器训练时每个环境的步数（默认24）

#### 使用原始训练脚本

```bash
python /home/ymzz-tec/code/legged_lab/scripts/rsl_rl/train_g1_depth.py \
  --task=LeggedLab-Isaac-Velocity-Rough-G1-Depth-v0 \
  --headless \
  --enable_cameras \
  --max_iterations 130000 \
  --num_envs 128
```

注意：原始脚本不包含深度编码器训练，仅使用展平的深度图像。

### 2. 推理

```bash
python /home/ymzz-tec/code/legged_lab/scripts/rsl_rl/play_g1_depth.py \
  --task=LeggedLab-Isaac-Velocity-Rough-G1-Depth-v0 \
  --checkpoint=/path/to/model.pt \
  --depth_checkpoint=/path/to/depth_encoder.pt \
  --use_depth_encoder \
  --num_envs 50
```

参数说明：
- `--checkpoint`: 主策略网络检查点路径
- `--depth_checkpoint`: 深度编码器检查点路径（可选，如果主检查点包含深度编码器权重）
- `--use_depth_encoder`: 启用深度编码器进行推理

## 配置文件

### 深度编码器配置

在 `legged_lab/rsl_rl/rl_cfg.py` 中定义：

```python
@configclass
class DepthEncoderCfg:
    use_depth_encoder: bool = True          # 是否使用深度编码器
    scandots_output_dim: int = 32           # 深度特征输出维度
    hidden_state_dim: int = 512             # GRU隐藏状态维度
    output_dim: int = 32                    # 最终输出维度
    learning_rate: float = 1.0e-4           # 学习率
    use_recurrent: bool = True              # 是否使用GRU
    num_frames: int = 1                     # 输入帧数
```

### G1深度PPO配置

在 `legged_lab/tasks/locomotion/velocity/config/g1/agents/rsl_rl_depth_ppo_cfg.py` 中定义：

```python
@configclass
class G1DepthPPORunnerCfg(G1RoughPPORunnerCfg):
    depth_encoder = DepthEncoderCfg(
        use_depth_encoder=True,
        scandots_output_dim=32,
        hidden_state_dim=512,
        output_dim=32,
        learning_rate=1.0e-4,
        use_recurrent=True,
        num_frames=1,
    )
    # ... 其他配置
```

## 代码结构

```
legged_lab/
├── source/legged_lab/legged_lab/
│   ├── rsl_rl/
│   │   ├── modules/
│   │   │   ├── __init__.py
│   │   │   └── depth_backbone.py          # 深度编码器网络
│   │   ├── depth_training_wrapper.py      # 深度训练包装器
│   │   ├── rl_cfg.py                      # 配置定义
│   │   └── __init__.py
│   └── tasks/locomotion/velocity/config/g1/
│       ├── agents/
│       │   ├── rsl_rl_depth_ppo_cfg.py    # G1深度PPO配置
│       │   └── __init__.py                # 环境注册
│       └── rough_depth_env_cfg.py         # 深度环境配置
└── scripts/rsl_rl/
    ├── train_g1_depth.py                  # 原始训练脚本
    ├── train_g1_depth_improved.py         # 改进的训练脚本
    └── play_g1_depth.py                   # 推理脚本
```

## 与extreme-parkour的区别

1. **移除偏航角输出**：双足机器人不需要偏航角信息，RecurrentDepthBackbone只输出32维深度特征
2. **本体感知维度**：G1双足机器人的本体感知维度约为43-48维，与四足机器人不同
3. **深度图像尺寸**：使用64x64的深度图像，而不是58x87
4. **训练策略**：简化了训练流程，专注于深度actor的蒸馏训练

## 性能优化建议

1. **调整深度训练间隔**：
   - 较小的间隔（如5-10）：更频繁的深度编码器更新，但训练时间更长
   - 较大的间隔（如20-50）：更快的训练，但深度编码器可能不够准确

2. **调整深度特征维度**：
   - 增加`scandots_output_dim`和`output_dim`可以提高表达能力，但增加计算量
   - 减小这些维度可以加快推理速度

3. **GRU隐藏状态维度**：
   - 默认512维，可以根据性能需求调整
   - 更大的隐藏状态可以捕获更多时序信息

## 故障排除

### 问题1：深度图像获取失败
```
[WARN] Could not get depth image from environment
```
**解决方案**：确保环境配置中包含深度相机，并且使用了正确的任务ID：
```bash
--task=LeggedLab-Isaac-Velocity-Rough-G1-Depth-v0
```

### 问题2：本体感知维度不匹配
```
RuntimeError: size mismatch
```
**解决方案**：检查并调整`DepthTrainingWrapper`中的`_get_proprio_dim`方法，确保返回正确的本体感知维度。

### 问题3：深度编码器权重未保存
**解决方案**：使用改进的训练脚本，它会自动保存深度编码器权重到单独的文件。

## 参考资料

- extreme-parkour: https://github.com/chengxuxin/extreme-parkour
- Isaac Lab文档: https://isaac-sim.github.io/IsaacLab/
- RSL-RL文档: https://github.com/leggedrobotics/rsl_rl

## 许可证

本实现基于extreme-parkour的深度处理方法，遵循BSD-3-Clause许可证。
