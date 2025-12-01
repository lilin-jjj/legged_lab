# G1深度编码器快速开始指南

## 快速开始

### 1. 使用改进的深度编码器训练（推荐）

```bash
cd /home/ymzz-tec/code/legged_lab

# 训练G1双足机器人，使用深度编码器
python scripts/rsl_rl/train_g1_depth_improved.py \
  --task=LeggedLab-Isaac-Velocity-Rough-G1-Depth-v0 \
  --headless \
  --enable_cameras \
  --max_iterations 130000 \
  --num_envs 128 \
  --depth_training_interval 10 \
  --depth_steps_per_env 24
```

### 2. 使用原始方式训练（仅展平深度图像）

```bash
python scripts/rsl_rl/train_g1_depth.py \
  --task=LeggedLab-Isaac-Velocity-Rough-G1-Depth-v0 \
  --headless \
  --enable_cameras \
  --max_iterations 130000 \
  --num_envs 128
```

### 3. 推理测试

```bash
# 使用深度编码器进行推理
python scripts/rsl_rl/play_g1_depth.py \
  --task=LeggedLab-Isaac-Velocity-Rough-G1-Depth-v0 \
  --checkpoint=logs/rsl_rl/g1_depth/model_130000.pt \
  --depth_checkpoint=logs/rsl_rl/g1_depth/depth_encoder_130000.pt \
  --use_depth_encoder \
  --num_envs 50
```

## 主要改进点

### 1. 深度图像处理

**原始方式**：
```python
# 直接展平深度图像 (64x64 = 4096维)
depth_flat = depth_image.view(batch_size, -1)
obs = torch.cat([proprio, depth_flat], dim=-1)  # 非常高维
```

**改进方式**：
```python
# 使用CNN提取深度特征 (32维)
depth_features = depth_encoder(depth_image)  # 64x64 -> 32
obs = torch.cat([proprio, depth_features], dim=-1)  # 低维且更有表达力
```

### 2. 时序信息处理

**改进方式使用GRU处理时序信息**：
```python
# 深度特征 + 本体感知 -> 融合 -> GRU -> 输出
depth_latent = recurrent_depth_encoder(depth_image, proprioception)
```

这样可以：
- 捕获时序依赖关系
- 平滑深度特征
- 提高鲁棒性

### 3. 蒸馏训练

通过模仿学习，让深度策略学习主策略的行为：
```python
# Teacher: 使用完整观测（包括展平深度图像）
actions_teacher = main_policy(full_obs)

# Student: 仅使用深度特征
actions_student = depth_policy(obs_with_depth_features)

# 损失：让student模仿teacher
loss = ||actions_teacher - actions_student||²
```

## 文件结构

```
已创建的新文件：
├── source/legged_lab/legged_lab/rsl_rl/
│   ├── modules/
│   │   ├── __init__.py                    # 模块初始化
│   │   └── depth_backbone.py              # ✓ 深度编码器网络
│   ├── depth_training_wrapper.py          # ✓ 深度训练包装器
│   └── rl_cfg.py                          # ✓ 添加了DepthEncoderCfg
├── tasks/locomotion/velocity/config/g1/agents/
│   ├── rsl_rl_depth_ppo_cfg.py            # ✓ G1深度PPO配置
│   └── __init__.py                        # ✓ 更新了环境注册
└── scripts/rsl_rl/
    ├── train_g1_depth_improved.py         # ✓ 改进的训练脚本
    └── play_g1_depth.py                   # ✓ 推理脚本

文档：
├── DEPTH_ENCODER_README_CN.md             # ✓ 详细文档
└── QUICK_START_DEPTH_CN.md                # ✓ 快速开始指南
```

## 关键参数说明

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_iterations` | 130000 | 总训练迭代次数 |
| `--num_envs` | 128 | 并行环境数量 |
| `--depth_training_interval` | 10 | 深度编码器训练间隔 |
| `--depth_steps_per_env` | 24 | 每次深度训练的步数 |

### 深度编码器配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `scandots_output_dim` | 32 | 深度特征维度 |
| `hidden_state_dim` | 512 | GRU隐藏状态维度 |
| `output_dim` | 32 | 最终输出维度 |
| `learning_rate` | 1e-4 | 深度编码器学习率 |
| `use_recurrent` | True | 是否使用GRU |

## 预期效果

使用深度编码器后，相比直接展平深度图像：

1. **观测维度降低**：从4096维降低到32维
2. **训练效率提升**：更小的网络，更快的训练
3. **泛化能力增强**：CNN提取的特征更具泛化性
4. **时序信息利用**：GRU捕获时序依赖

## 下一步

1. **调参优化**：
   - 调整`depth_training_interval`找到最佳训练频率
   - 调整深度特征维度平衡性能和效率
   - 调整GRU隐藏状态维度

2. **可视化分析**：
   - 可视化深度特征的激活
   - 分析深度编码器的学习曲线
   - 对比使用和不使用深度编码器的性能

3. **部署优化**：
   - 导出深度编码器为ONNX格式
   - 量化加速推理
   - 集成到实际机器人系统

## 常见问题

**Q: 训练时间会增加多少？**
A: 深度编码器训练是额外的，但由于每10次迭代才训练一次，总体增加约10-20%的训练时间。

**Q: 可以不使用GRU吗？**
A: 可以，设置`use_recurrent=False`即可使用纯CNN编码器，但可能损失时序信息。

**Q: 深度编码器可以用于其他机器人吗？**
A: 可以，只需调整本体感知维度和深度图像尺寸即可。

**Q: 如何验证深度编码器是否工作？**
A: 查看训练日志中的`depth_actor_loss`，应该逐渐降低。

## 技术支持

如有问题，请参考：
- 详细文档：`DEPTH_ENCODER_README_CN.md`
- extreme-parkour源码：`/home/ymzz-tec/code/dog/extreme-parkour`
- Isaac Lab文档：https://isaac-sim.github.io/IsaacLab/
