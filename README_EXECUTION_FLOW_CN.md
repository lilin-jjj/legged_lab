# 训练流程完整梳理 - 总结文档

## 📚 已生成的文档列表

我为你生成了 **3 份详细的中文文档**，分别用于不同的查询需求：

### 1️⃣ **EXECUTION_FLOW_CN.md** - 最详细的完整流程
**适合：** 想要全面理解整个训练过程的人

包含内容：
- 六个执行阶段的详细说明（参数解析 → 模拟器启动 → 配置加载 → 环境创建 → Runner 构建 → 训练循环）
- 每个阶段内的所有关键步骤
- 传感器数据流详解
- 完整的函数调用图
- 关键 MDP 函数列表
- 总结表格

**快速定位：**
- 想了解 gym.make() 如何创建环境？ → 看"第三阶段：环境创建"
- 想了解传感器如何被使用？ → 看"传感器详细说明"部分
- 想了解每次 step 发生了什么？ → 看"第五阶段：训练循环"

---

### 2️⃣ **EXECUTION_FLOW_QUICK_GUIDE_CN.md** - 快速查阅版
**适合：** 想快速了解执行流程、或作为学习的导览

包含内容：
- 一句话总结
- 清晰的执行流程图（ASCII 艺术）
- 按调用顺序的关键函数表
- 传感器在高频循环中的用途
- 常见问题 Q&A

**快速定位：**
- 快速看执行流程？ → 看"执行流程图（简洁版）"
- 按顺序列出的函数？ → 看"按调用顺序列出的关键函数"
- 为什么有多个环境？ → 看"常见问题"

---

### 3️⃣ **FUNCTION_CALL_INDEX_CN.md** - 函数索引手册
**适合：** 需要查找特定函数的位置、功能、输入输出的人

包含内容：
- 按执行顺序的完整函数列表（每个函数的位置、功能、输入输出、内部细节）
- 环境创建（gym.make）的内部详解
- env.step() 的完整执行步骤（包含所有传感器数据流）
- 传感器生命周期
- 函数调用频率表

**快速定位：**
- 想知道 env.step() 内部发生了什么？ → 看"env.step(action) 内部执行流程"
- 想看特定函数的功能？ → 看"按执行顺序的完整函数列表"
- height_scanner 的完整生命周期？ → 看"传感器数据的完整生命周期"

---

## 🎯 按你的问题来选择文档

### "训练脚本执行的完整顺序是什么？"
→ 推荐：**EXECUTION_FLOW_QUICK_GUIDE_CN.md**（看"执行流程图"）
或 **FUNCTION_CALL_INDEX_CN.md**（看"按执行顺序的完整函数列表"）

### "gym.make() 创建环境时发生了什么？"
→ 推荐：**FUNCTION_CALL_INDEX_CN.md**（看"env.step(action) 内部执行流程"中的【14.1-14.7】部分）

### "height_scanner 和 contact_forces 如何被使用？"
→ 推荐：**FUNCTION_CALL_INDEX_CN.md**（看"【⭐】传感器 1" 和 "【⭐】传感器 2" 部分）

### "每个 env.step() 内部在做什么？"
→ 推荐：**FUNCTION_CALL_INDEX_CN.md**（看"env.step(action) 内部执行流程"）

### "Actor 和 Critic 网络在哪里被使用？"
→ 推荐：**FUNCTION_CALL_INDEX_CN.md**（看"runner.learn() 中的【步骤 A】和【步骤 C】"）

### "为什么有 6 个不同的环境 ID（Rough/Flat/Scandots）？"
→ 推荐：前面回答的 `__init__.py` 中的注释

---

## 🔍 快速查找指南

### 如果你想知道...

**"哪些函数在训练中被频繁调用？"**
```
查看: FUNCTION_CALL_INDEX_CN.md 最后的 "总结：函数调用频率" 表格
```

**"观测（observation）是如何生成的？"**
```
查看: EXECUTION_FLOW_CN.md 的 "第五阶段" → "5.2 训练循环内每次迭代"
     中的 "【环节 A4】观测生成"
```

**"奖励是如何计算的？"**
```
查看: EXECUTION_FLOW_CN.md 的 "第五阶段" → "5.2 训练循环内每次迭代"
     中的 "【环节 A5】奖励计算"
或
查看: FUNCTION_CALL_INDEX_CN.md 的 "env.step(action) 内部执行流程"
     中的 "【21.5】奖励计算"
```

**"contact_forces 传感器的字段有哪些？"**
```
查看: FUNCTION_CALL_INDEX_CN.md 的 【⭐ 传感器 2】contact_forces
```

**"height_scanner 传感器的工作原理是什么？"**
```
查看: FUNCTION_CALL_INDEX_CN.md 的 【⭐ 传感器 1】height_scanner
或
查看: EXECUTION_FLOW_CN.md 的 "传感器详细说明" → "RayCaster / RayCasterArray"
```

**"为什么 env.step() 会调用那么多函数？"**
```
查看: FUNCTION_CALL_INDEX_CN.md 的 "env.step(action) 内部执行流程"
      (包括物理模拟、传感器更新、事件触发、观测、奖励、终止、重置 7 个步骤)
```

---

## 🎓 学习建议

### 初级：快速了解流程（30 分钟）
1. 阅读：**EXECUTION_FLOW_QUICK_GUIDE_CN.md** → "执行流程图"
2. 浏览：**EXECUTION_FLOW_QUICK_GUIDE_CN.md** → "按调用顺序的关键函数表"
3. 了解：为什么会有 "物理模拟 → 传感器更新 → 观测计算 → 奖励计算 → 终止判断"

### 中级：深入理解传感器与奖励（1 小时）
1. 阅读：**FUNCTION_CALL_INDEX_CN.md** → "env.step(action) 内部执行流程"
2. 重点关注：
   - 【21.2】传感器更新（height_scanner & contact_forces）
   - 【21.4】观测生成（如何从传感器获取观测）
   - 【21.5】奖励计算（如何从传感器计算奖励）
   - 【21.6】终止判断（如何从传感器判断终止）

### 高级：完全掌握代码执行流（2-3 小时）
1. 完整阅读：**EXECUTION_FLOW_CN.md** 的所有章节
2. 深入研究：**FUNCTION_CALL_INDEX_CN.md** 的所有细节
3. 对应查看源代码：
   - 观测计算：`legged_lab/.../mdp/observations.py`
   - 奖励计算：`legged_lab/.../mdp/rewards.py`
   - 终止判断：`legged_lab/.../mdp/terminations.py`
   - 环境配置：`legged_lab/.../velocity_env_cfg.py`

---

## 📋 核心概念速览

### 为什么要分成 4 个 Manager？
- **ObservationManager**：负责把物理数据（位置、速度、传感器）转换为神经网络的输入
- **RewardManager**：负责把物理数据转换为学习信号（奖励）
- **TerminationManager**：负责判断这个环节是否失败
- **CommandManager**：负责生成目标速度（让机器人学会追踪）

### 传感器为什么有不同的 update_period？
- **height_scanner**: 0.02s（每 4 个物理步）- 机器人运动相对缓慢，不需要那么高的频率
- **contact_forces**: 0.005s（每 1 个物理步）- 脚步接触/离开的时间很短，需要精准捕捉

### 为什么要用 RayCaster 而不是 LiDAR？
- RayCaster 是射线投射，计算快、数据简洁
- 用网格化射线可以得到 2D 高度图（height_scan_ch），类似于 2D 摄像头图像
- 这种表示对卷积网络友好

### 为什么 contact_forces 要记录 history？
- 当前接触力可能波动大，历史数据可以提供时序信息
- 例如可以计算过去 3 个时间步中的最大接触力，更稳定

---

## 🔑 三个最重要的函数

### 1. env.step(action)
**做的事：**
- 运行物理模拟
- 更新所有传感器
- 计算观测、奖励、终止信号
- 返回 (obs, reward, done, info)

**调用频率：** 983 亿次（最频繁！）
**位置：** isaaclab 库（外部）

### 2. runner.learn()
**做的事：**
- 主训练循环
- 收集经验 → 计算优势 → 更新网络

**调用频率：** 1 次（但内部调用 env.step 数十亿次）
**位置：** rsl_rl 库（外部）

### 3. ObservationManager.compute() / RewardManager.compute() / TerminationManager.compute()
**做的事：**
- 在每个 env.step() 内被调用
- 通过调用多个小的 callback 函数（ObsTerm、RewTerm、DoneTerm）计算观测、奖励、终止

**调用频率：** 每 step 1 次（983 亿次）
**位置：** isaaclab 库（外部）

---

## 💡 理解传感器数据流的关键

### height_scanner 的数据流
```
世界中的地面
     ↓ (射线投射)
RayCaster.data.ray_hits_w [4096, 128, 3]
     ↓ (高度计算：sensor_z - ray_z)
mdp.height_scan() 返回 [4096, 128]
     ↓ (作为观测)
Actor/Critic 网络
     ↓ (学会通过地形识别)
策略能够避开障碍
```

### contact_forces 的数据流
```
物理引擎中的接触
     ↓
ContactSensor.data.net_forces_w_history [4096, 3, num_bodies, 3]
     ↓
多个 Reward/Termination 回调使用：
├─ feet_slide()         ← 用来惩罚滑移
├─ feet_air_time()      ← 用来奖励腾空
├─ feet_gait()          ← 用来奖励步态对齐
├─ undesired_contacts() ← 用来惩罚异常接触
└─ illegal_contact()    ← 用来判断失败
     ↓
网络学会了控制脚步接触/离开的时序
```

---

## 🎬 一个完整的训练循环示意

```
Iteration 0:

step 0: env.step(action_0) 
  → 物理 → height_scanner 更新 → contact_forces 更新 
  → obs_0, reward_0, done_0

step 1: env.step(action_1)
  → obs_1, reward_1, done_1

... (重复 24 步)

step 23: env.step(action_23)
  → obs_23, reward_23, done_23

[收集完数据]

计算 advantages 和 returns

FOR epoch in 5 epochs:
  FOR mini_batch in 4 batches:
    actor_loss = ... (基于新策略)
    value_loss = ... (基于新价值)
    loss.backward()
    optimizer.step()

保存 model_0.pt (如果 iteration % 100 == 0)

Iteration 1:
[重复上面的过程，共 10000 次迭代]
```

---

## 📖 文件位置速查

```
legged_lab/
├─ scripts/rsl_rl/
│  └─ train.py                    ← 训练入口脚本
├─ source/legged_lab/legged_lab/
│  └─ tasks/locomotion/velocity/
│     ├─ __init__.py              ← Gym 任务注册
│     ├─ velocity_env_cfg.py       ← 环境配置（传感器、MDP 配置）
│     ├─ mdp/
│     │  ├─ observations.py        ← 观测计算（包括 height_scan）
│     │  ├─ rewards.py             ← 奖励计算（包括 feet_slide 等）
│     │  └─ terminations.py        ← 终止判断（包括 illegal_contact）
│     └─ config/g1/
│        ├─ __init__.py            ← G1 任务注册和说明 ⭐
│        ├─ rough_env_cfg.py        ← G1 粗糙地形配置
│        ├─ flat_env_cfg.py         ← G1 平坦地形配置
│        ├─ scandots_rough_env_cfg.py ← G1 Scandots 配置
│        └─ agents/
│           └─ rsl_rl_ppo_cfg.py    ← PPO 算法参数配置
```

---

## ✅ 总结

你现在拥有：

1. **EXECUTION_FLOW_CN.md** - 完整的流程说明（最详细）
2. **EXECUTION_FLOW_QUICK_GUIDE_CN.md** - 快速查阅版（有流程图）
3. **FUNCTION_CALL_INDEX_CN.md** - 函数索引和详细实现（最具体）

这 3 份文档配合使用，可以：
- ✅ 快速理解整个训练流程
- ✅ 深入了解每个函数的作用
- ✅ 清楚地看到传感器数据如何被使用
- ✅ 查询任何关键函数的位置和功能

**建议：** 先看 QUICK_GUIDE，再根据需要深入阅读其他两份文档。

