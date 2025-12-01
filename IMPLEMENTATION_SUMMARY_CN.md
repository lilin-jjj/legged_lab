# G1深度编码器实现总结

## 概述

成功将extreme-parkour的深度图像处理方式移植到G1双足机器人代码中，实现了基于CNN和GRU的深度编码器架构。

## 核心实现

### 1. 深度编码器模块 (`depth_backbone.py`)

#### DepthOnlyFCBackbone
- **功能**：使用CNN提取深度图像特征
- **输入**：64x64深度图像
- **输出**：32维深度特征向量
- **架构**：Conv2d → MaxPool → Conv2d → Flatten → FC → FC

#### RecurrentDepthBackbone
- **功能**：融合深度特征和本体感知，使用GRU处理时序信息
- **输入**：深度图像 + 本体感知信息
- **输出**：32维融合特征（不含偏航角，适配双足机器人）
- **架构**：DepthBackbone → Concat → MLP → GRU → MLP

#### StackDepthEncoder
- **功能**：处理多帧深度图像序列
- **用途**：备用，当前主要使用RecurrentDepthBackbone

### 2. 训练包装器 (`depth_training_wrapper.py`)

**DepthTrainingWrapper类**：
- 管理深度编码器和深度actor的创建
- 处理深度观测的前向传播
- 实现深度编码器的蒸馏训练
- 管理优化器和隐藏状态

**主要方法**：
```python
- process_depth_observation()  # 处理深度图像
- update_depth_encoder()       # 更新编码器（蒸馏）
- update_depth_actor()         # 更新深度策略
- detach_hidden_states()       # 分离GRU隐藏状态
- get_state_dict()             # 保存权重
- load_state_dict()            # 加载权重
```

### 3. 配置系统

#### DepthEncoderCfg (`rl_cfg.py`)
```python
@configclass
class DepthEncoderCfg:
    use_depth_encoder: bool = True
    scandots_output_dim: int = 32
    hidden_state_dim: int = 512
    output_dim: int = 32
    learning_rate: float = 1.0e-4
    use_recurrent: bool = True
    num_frames: int = 1
```

#### G1DepthPPORunnerCfg (`rsl_rl_depth_ppo_cfg.py`)
- 继承自G1RoughPPORunnerCfg
- 添加depth_encoder配置
- 设置实验名称为"g1_depth"

### 4. 训练脚本

#### train_g1_depth_improved.py
**改进的训练流程**：
1. 标准PPO训练主策略
2. 每隔N次迭代进行深度编码器训练
3. 深度编码器通过模仿学习训练
4. 自动保存深度编码器权重

**关键参数**：
- `--depth_training_interval`: 深度训练间隔（默认10）
- `--depth_steps_per_env`: 每次深度训练步数（默认24）

#### train_g1_depth.py
- 原始训练脚本（保留）
- 仅使用展平的深度图像
- 不包含深度编码器训练

### 5. 推理脚本

#### play_g1_depth.py
- 支持加载深度编码器权重
- 使用深度actor进行推理
- 提供性能统计

## 与extreme-parkour的关键差异

| 特性 | extreme-parkour (四足) | 本实现 (双足) |
|------|----------------------|-------------|
| 偏航角输出 | 输出2维偏航角 | 不输出偏航角 |
| 本体感知维度 | ~53维 | ~43-48维 |
| 深度图像尺寸 | 58x87 | 64x64 |
| 输出维度 | 32+2 (含yaw) | 32 (纯特征) |
| 训练集成 | 完整集成到runner | 使用wrapper分离 |

## 文件清单

### 新增文件

```
source/legged_lab/legged_lab/rsl_rl/
├── modules/
│   ├── __init__.py                          # ✓ 新建
│   └── depth_backbone.py                    # ✓ 新建 (200行)
├── depth_training_wrapper.py                # ✓ 新建 (250行)
├── rl_cfg.py                                # ✓ 修改 (添加DepthEncoderCfg)
└── __init__.py                              # ✓ 修改 (导出DepthEncoderCfg)

source/legged_lab/legged_lab/tasks/locomotion/velocity/config/g1/agents/
├── rsl_rl_depth_ppo_cfg.py                  # ✓ 新建 (60行)
└── __init__.py                              # ✓ 修改 (注册环境)

scripts/rsl_rl/
├── train_g1_depth_improved.py               # ✓ 新建 (300行)
└── play_g1_depth.py                         # ✓ 新建 (200行)

文档/
├── DEPTH_ENCODER_README_CN.md               # ✓ 新建 (详细文档)
├── QUICK_START_DEPTH_CN.md                  # ✓ 新建 (快速指南)
└── IMPLEMENTATION_SUMMARY_CN.md             # ✓ 新建 (本文件)
```

### 修改文件

```
source/legged_lab/legged_lab/rsl_rl/
├── rl_cfg.py                                # 添加DepthEncoderCfg类
└── __init__.py                              # 导出DepthEncoderCfg

source/legged_lab/legged_lab/tasks/locomotion/velocity/config/g1/agents/
└── __init__.py                              # 注册LeggedLab-Isaac-Velocity-Rough-G1-Depth-v0
```

## 使用流程

### 训练流程

```
1. 启动训练
   ↓
2. 创建环境和主策略网络
   ↓
3. 创建深度编码器和深度actor
   ↓
4. 主训练循环：
   ├─ 标准PPO训练 (每次迭代)
   └─ 深度编码器训练 (每N次迭代)
      ├─ 收集深度图像
      ├─ 生成teacher动作
      ├─ 生成student动作
      ├─ 计算模仿损失
      └─ 更新深度网络
   ↓
5. 保存检查点
   ├─ 主策略权重
   └─ 深度编码器权重
```

### 推理流程

```
1. 加载检查点
   ├─ 主策略权重
   └─ 深度编码器权重
   ↓
2. 获取观测
   ├─ 本体感知信息
   └─ 深度图像
   ↓
3. 深度编码器处理
   ├─ CNN提取特征
   ├─ 融合本体感知
   └─ GRU处理时序
   ↓
4. 深度actor生成动作
   ↓
5. 执行动作
```

## 技术要点

### 1. 维度适配

**本体感知维度计算**：
```python
# G1双足机器人
n_proprio = 43  # base_lin_vel(3) + base_ang_vel(3) + 
                # projected_gravity(3) + commands(4) + 
                # joint_pos(10) + joint_vel(10) + actions(10)
```

### 2. 深度图像处理

**从环境获取**：
```python
depth_image = env.scene.sensors["depth_camera"].data.output["distance_to_image_plane"]
depth_image = depth_image.view(batch_size, -1)  # 展平
```

**通过编码器处理**：
```python
depth_latent = depth_encoder(depth_image, proprioception)
```

### 3. 隐藏状态管理

**GRU隐藏状态**：
```python
# 每次训练步骤后分离
depth_encoder.detach_hidden_states()

# episode结束时重置
depth_encoder.reset_hidden_states(batch_size, device)
```

### 4. 蒸馏训练

**损失函数**：
```python
# 深度actor损失
loss = ||actions_teacher - actions_student||²_2

# 可选：深度特征损失
loss_feat = ||scandots_latent - depth_latent||²_2
```

## 性能优化

### 1. 计算效率

- **观测维度**：4096 → 32 (减少99%)
- **网络大小**：更小的MLP层
- **推理速度**：预期提升20-30%

### 2. 训练效率

- **并行训练**：深度编码器训练可与主训练并行
- **训练频率**：可调节深度训练间隔
- **批量大小**：支持大批量训练

### 3. 内存使用

- **GRU状态**：512维隐藏状态
- **特征缓存**：32维深度特征
- **总体**：相比展平深度图像显著降低

## 验证方法

### 1. 训练验证

检查训练日志：
```
Depth actor loss: 0.xxxx  # 应逐渐降低
```

### 2. 推理验证

对比性能：
```bash
# 使用深度编码器
python play_g1_depth.py --use_depth_encoder

# 不使用深度编码器
python play_g1_depth.py
```

### 3. 可视化验证

- 可视化深度特征的激活
- 对比teacher和student的动作
- 分析GRU隐藏状态的演化

## 已知限制

1. **本体感知维度**：当前使用固定值，需要根据实际配置调整
2. **深度图像获取**：依赖特定的传感器名称"depth_camera"
3. **训练集成**：使用wrapper而非完全集成到runner
4. **配置灵活性**：部分参数硬编码，可进一步配置化

## 后续改进方向

1. **自动维度检测**：自动从环境配置中获取本体感知维度
2. **多帧支持**：完善StackDepthEncoder的使用
3. **注意力机制**：在深度特征提取中加入注意力
4. **端到端训练**：探索联合训练主策略和深度编码器
5. **模型压缩**：量化和剪枝以加速部署

## 总结

本实现成功将extreme-parkour的深度图像处理方法适配到G1双足机器人，主要改进包括：

✓ 使用CNN提取深度特征，降低观测维度
✓ 使用GRU处理时序信息，提高鲁棒性
✓ 通过蒸馏训练，学习高效的深度策略
✓ 移除偏航角输出，适配双足机器人
✓ 提供完整的训练和推理脚本
✓ 详细的中文文档和使用指南

代码结构清晰，易于扩展和维护，为后续的研究和应用奠定了基础。
