# 执行流程详解：train.py 完整调用链

## 命令行执行
```bash
python scripts/rsl_rl/train.py --task=LeggedLab-Isaac-Velocity-Rough-G1-v0 --headless
```

---

## 第一阶段：参数解析与模拟器启动

### 1.1 命令行参数解析
```
文件: scripts/rsl_rl/train.py（第 14-36 行）
```

**调用链：**
```
parser = argparse.ArgumentParser(...)
  ├─ parser.add_argument("--task", ...)       # 添加 task 参数
  ├─ parser.add_argument("--headless", ...)   # 添加 headless 参数
  ├─ parser.add_argument("--num_envs", ...)   # 添加并行环境数参数
  └─ ...其他参数...

cli_args.add_rsl_rl_args(parser)              # 添加 rsl-rl 专用参数
AppLauncher.add_app_launcher_args(parser)     # 添加 Isaac Sim 启动参数

args_cli, hydra_args = parser.parse_known_args()  # 解析 CLI 参数
```

**关键参数值（你的命令）：**
- `args_cli.task = "LeggedLab-Isaac-Velocity-Rough-G1-v0"`
- `args_cli.headless = True`
- `args_cli.num_envs = None`（使用配置文件中的默认值 4096）

---

### 1.2 启动 Isaac Sim 模拟器

```
文件: scripts/rsl_rl/train.py（第 43-45 行）
```

**调用链：**
```
AppLauncher(args_cli)
  ├─ 初始化 Omniverse 应用（Isaac Sim）
  ├─ 创建场景和物理引擎
  ├─ 如果 headless=True，不显示 GUI
  └─ 返回 app_launcher 对象

simulation_app = app_launcher.app  # 获取模拟器应用对象
```

**执行结果：**
- Isaac Sim 模拟器后台启动
- 物理引擎准备好

---

### 1.3 检查 rsl-rl 版本

```
文件: scripts/rsl_rl/train.py（第 48-62 行）
```

**调用链：**
```
importlib.metadata.version("rsl-rl-lib")  # 获取已安装版本
version.parse(installed_version)          # 解析版本号
# 如果版本 < 3.1.1，提示用户更新
```

---

## 第二阶段：配置加载与环境创建

### 2.1 @hydra_task_config 装饰器触发

```
文件: scripts/rsl_rl/train.py（第 107 行）
```

**调用链：**
```
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
```

这个装饰器会：
1. 根据 `args_cli.task = "LeggedLab-Isaac-Velocity-Rough-G1-v0"` 查找配置
2. 加载 Hydra 配置文件
3. 注入 `env_cfg` 和 `agent_cfg` 到 main 函数

---

### 2.2 配置文件查找过程

**Hydra 配置查找链：**
```
"LeggedLab-Isaac-Velocity-Rough-G1-v0"
  ↓ (根据这个 task id 查找)
  ↓
legged_lab/tasks/locomotion/velocity/config/g1/__init__.py
  ↓
gym.register() 中找到对应的 entry_point 和 env_cfg_entry_point
  ↓
env_cfg_entry_point = "...rough_env_cfg:G1RoughEnvCfg"
  ↓
加载配置类: G1RoughEnvCfg
  ├─ 继承自 LocomotionVelocityRoughEnvCfg
  ├─ 在 velocity_env_cfg.py 中定义
  └─ 包含: robot, height_scanner, contact_forces 等配置

rsl_rl_cfg_entry_point = "...rsl_rl_ppo_cfg:G1RoughPPORunnerCfg"
  ↓
加载配置类: G1RoughPPORunnerCfg
  ├─ 继承自 RslRlOnPolicyRunnerCfg
  ├─ 在 rsl_rl_ppo_cfg.py 中定义
  └─ 包含: actor_hidden_dims, critic_hidden_dims, algorithm 参数等
```

---

### 2.3 main() 函数内的配置调整

```
文件: scripts/rsl_rl/train.py（第 109-126 行）
```

**调用链：**
```python
def main(env_cfg, agent_cfg):
    # 【步骤 1】使用 CLI 参数覆盖配置
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    #          ↑ 根据命令行参数更新 agent 配置
    
    # 【步骤 2】覆盖环境的 num_envs
    if args_cli.num_envs is not None:  # 如果命令行指定了 num_envs
        env_cfg.scene.num_envs = args_cli.num_envs
    else:
        # 使用配置文件默认值（G1RoughEnvCfg 中设置的 4096）
        env_cfg.scene.num_envs = env_cfg.scene.num_envs
    
    # 【步骤 3】覆盖最大迭代次数
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    else:
        agent_cfg.max_iterations = agent_cfg.max_iterations  # 使用配置文件值
    
    # 【步骤 4】设置随机种子
    env_cfg.seed = agent_cfg.seed
    
    # 【步骤 5】设置设备（GPU/CPU）
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
```

---

### 2.4 创建日志目录

```
文件: scripts/rsl_rl/train.py（第 128-138 行）
```

**调用链：**
```python
log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
# 结果: "logs/rsl_rl/g1_rough"

log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# 结果: "2025-11-25_14-30-45"

if agent_cfg.run_name:
    log_dir += f"_{agent_cfg.run_name}"
    
log_dir = os.path.join(log_root_path, log_dir)
# 最终: "logs/rsl_rl/g1_rough/2025-11-25_14-30-45"

env_cfg.log_dir = log_dir  # 传给环境配置
```

---

## 第三阶段：环境创建（关键！）

### 3.1 gym.make() 创建环境

```
文件: scripts/rsl_rl/train.py（第 150 行）
```

**调用链：**
```python
env = gym.make(
    args_cli.task,  # "LeggedLab-Isaac-Velocity-Rough-G1-v0"
    cfg=env_cfg,    # G1RoughEnvCfg 配置对象
    render_mode="rgb_array" if args_cli.video else None
)
```

**发生的事：**

1️⃣ Gym 查找注册表
```
gym.register 中找到:
  id="LeggedLab-Isaac-Velocity-Rough-G1-v0"
  entry_point="isaaclab.envs:ManagerBasedRLEnv"
  kwargs={
    "env_cfg_entry_point": "...rough_env_cfg:G1RoughEnvCfg",
    ...
  }
```

2️⃣ 创建 ManagerBasedRLEnv 实例
```
from isaaclab.envs import ManagerBasedRLEnv
env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=...)
```

3️⃣ ManagerBasedRLEnv 初始化过程（外部库，但执行的关键步骤）
```
ManagerBasedRLEnv.__init__()
  ├─ 【3.1.1】加载配置
  ├─ 【3.1.2】创建场景（Scene）
  ├─ 【3.1.3】初始化传感器
  ├─ 【3.1.4】初始化 Manager（观测、奖励、终止条件等）
  ├─ 【3.1.5】重置环境到初始状态
  └─ 返回可用的环境对象
```

#### 3.1.1 场景创建细节

```
env_cfg.scene (MySceneCfg 类型)
  ├─ terrain: 粗糙地形生成器
  │  └─ ROUGH_TERRAINS_CFG: 生成随机坡度、障碍
  │
  ├─ robot: 机器人资产
  │  └─ G1_27DOF_CFG: G1 机器人配置（27 个自由度）
  │
  ├─ height_scanner: RayCaster 高度扫描传感器
  │  ├─ prim_path="{ENV_REGEX_NS}/Robot/waist_yaw_link"  # 安装位置
  │  ├─ offset=(0.0, 0.0, 20.0)  # 相对位置（向上 20m，模拟从高处向下扫）
  │  ├─ pattern_cfg=GridPatternCfg(resolution=0.1, size=[1.6, 1.0])  # 网格扫描
  │  └─ mesh_prim_paths=["/World/ground"]  # 扫描地面
  │
  └─ contact_forces: ContactSensor 接触力传感器
     ├─ prim_path="{ENV_REGEX_NS}/Robot/.*"  # 匹配机器人所有 link
     ├─ history_length=3  # 记录最后 3 个时间步的数据
     └─ track_air_time=True  # 追踪脚步离地时间
```

#### 3.1.2 Manager 初始化

```
ObservationManager
  ├─ policy_obs: 生成策略网络的观测
  │  ├─ base_ang_vel: 基座角速度
  │  ├─ projected_gravity: 重力投影
  │  ├─ velocity_commands: 速度命令
  │  ├─ joint_pos: 关节位置
  │  ├─ joint_vel: 关节速度
  │  ├─ actions: 上一步动作
  │  └─ height_scan: 【传感器】高度扫描数据
  │
  └─ critic_obs: 评估网络的观测（包含特权观测）
     └─ ... 类似 policy_obs，但无噪声、无截断
     
RewardManager
  ├─ track_lin_vel_xy_exp: 追踪线速度奖励
  ├─ track_ang_vel_z_exp: 追踪角速度奖励
  ├─ alive: 活着就有奖励
  ├─ feet_slide: 【传感器】脚部滑移惩罚（使用 contact_forces）
  ├─ feet_air_time: 【传感器】脚步离地时间奖励（使用 contact_forces）
  ├─ feet_clearance: 脚部抬起高度奖励
  ├─ feet_gait: 【传感器】步态周期奖励（使用 contact_forces）
  ├─ undesired_contacts: 【传感器】不希望的接触惩罚（使用 contact_forces）
  └─ ... 其他能量、关节限制等惩罚
  
TerminationManager
  ├─ time_out: 超时终止
  ├─ base_contact: 【传感器】基座异常接触（使用 contact_forces）
  ├─ base_height: 基座高度过低
  └─ bad_orientation: 机器人翻覆
  
CommandManager
  └─ base_velocity: 生成基座速度命令（目标速度）
```

---

### 3.2 可选：视频录制包装

```
文件: scripts/rsl_rl/train.py（第 160-167 行）
```

**调用链：**
```python
if args_cli.video:
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=...,
        step_trigger=...,  # 每 2000 步记录一次
        video_length=200,  # 每段视频 200 步
    )
```

不执行该步骤（默认 --video 未指定）。

---

### 3.3 rsl-rl 环境包装

```
文件: scripts/rsl_rl/train.py（第 169 行）
```

**调用链：**
```python
env = RslRlVecEnvWrapper(
    env,
    clip_actions=agent_cfg.clip_actions  # 是否裁剪动作
)
```

**作用：**
- 将 Isaac Lab 环境的输出转换为 rsl-rl 期望的格式
- 处理批量环境并行执行
- 管理动作、观测的张量形状和数据类型

---

## 第四阶段：构建 Runner 和模型

### 4.1 创建 Runner

```
文件: scripts/rsl_rl/train.py（第 171-180 行）
```

**调用链：**
```python
if agent_cfg.class_name == "OnPolicyRunner":  # 本次执行走这条分支
    runner = OnPolicyRunner(
        env,                          # 包装后的环境
        agent_cfg.to_dict(),         # agent 配置转字典
        log_dir=log_dir,             # 日志目录
        device=agent_cfg.device      # 设备（cuda:0）
    )
    # OnPolicyRunner 会在此时创建：
    # ├─ Actor 网络（策略网络）: [obs_dim] → [action_dim]
    # │  └─ 隐藏层: [512, 256, 128]
    # ├─ Critic 网络（价值网络）: [obs_dim] → [1]
    # │  └─ 隐藏层: [512, 256, 128]
    # └─ 优化器、学习率调度器等
```

---

### 4.2 记录 Git 状态

```
文件: scripts/rsl_rl/train.py（第 181-182 行）
```

**调用链：**
```python
runner.add_git_repo_to_log(__file__)
# 保存当前 git 状态到日志目录（用于复现）
```

---

### 4.3 可选：加载预训练模型

```
文件: scripts/rsl_rl/train.py（第 183-187 行）
```

**调用链：**
```python
if agent_cfg.resume:  # 如果指定了 --resume
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    runner.load(resume_path)  # 加载之前保存的检查点
```

不执行该步骤（默认未指定 resume）。

---

### 4.4 保存配置文件

```
文件: scripts/rsl_rl/train.py（第 189-192 行）
```

**调用链：**
```python
dump_yaml(
    os.path.join(log_dir, "params", "env.yaml"),
    env_cfg
)
# 保存: logs/rsl_rl/g1_rough/2025-11-25_14-30-45/params/env.yaml

dump_yaml(
    os.path.join(log_dir, "params", "agent.yaml"),
    agent_cfg
)
# 保存: logs/rsl_rl/g1_rough/2025-11-25_14-30-45/params/agent.yaml
```

---

## 第五阶段：训练循环（最重要！）

### 5.1 启动训练

```
文件: scripts/rsl_rl/train.py（第 195 行）
```

**调用链：**
```python
runner.learn(
    num_learning_iterations=agent_cfg.max_iterations,  # 10000 次迭代
    init_at_random_ep_len=True  # 初始化随机长度
)
```

---

### 5.2 训练循环内每次迭代（简化版）

OnPolicyRunner 内部做的事（rsl-rl 库）：

```
for iteration in range(num_learning_iterations):  # 0 ~ 9999
    
    # 【阶段 A】收集经验：num_steps_per_env=24 步 × num_envs=4096 环境
    for step in range(num_steps_per_env):
        
        # ∘ 获取当前观测（来自 ObservationManager）
        obs = env.obs  # shape: [4096, obs_dim]
        
        # ∘ 策略网络前向传播（Actor）
        actions, log_probs = actor(obs)  # [4096, action_dim]
        
        # ∘ 执行动作到环境（Isaac Lab env.step()）
        env.step(actions)
            ↓ 在 ManagerBasedRLEnv.step() 内部：
            
            【环节 A1】物理模拟
            ├─ 运行物理引擎 decimation 次（通常 4 次）
            ├─ 每次用设置的 PD 控制器更新关节目标位置
            └─ 模拟地面接触、重力等
            
            【环节 A2】传感器数据更新（按 update_period）
            ├─ height_scanner 更新 (update_period = decimation × sim.dt = 0.02 秒)
            │  └─ 发射射线到地面，计算相对高度
            ├─ contact_forces 更新 (update_period = sim.dt = 0.005 秒)
            │  └─ 记录每个 body 的接触力、接触时间、离地时间
            └─ 其他传感器（IMU、关节力等）
            
            【环节 A3】事件触发（Randomization）
            ├─ startup: 只在重置时运行（摩擦系数随机化等）
            ├─ reset: 在环节重置时运行（外力推送、质量变化等）
            └─ interval: 每隔一定时间运行（push_robot 等）
            
            【环节 A4】观测生成（ObservationManager 调用各个 ObsTerm）
            ├─ base_ang_vel()          # 基座角速度
            ├─ projected_gravity()     # 投影重力
            ├─ generated_commands()    # 速度命令
            ├─ joint_pos_rel()         # 相对关节位置
            ├─ joint_vel_rel()         # 相对关节速度
            ├─ last_action()           # 上一步动作
            └─ height_scan()           # 【关键】调用 mdp.height_scan
                 └─ 从 height_scanner sensor 读取 ray_hits_w
                 └─ 计算: sensor_height - hit_z - offset
                 └─ 返回 shape [4096, 128] 的高度值
            
            next_obs = [base_ang_vel, gravity, commands, joint_pos, ...]
            # shape: [4096, obs_dim], obs_dim=212（取决于观测配置）
            
            【环节 A5】奖励计算（RewardManager 调用各个 RewTerm）
            ├─ track_lin_vel_xy_exp()         # 追踪线速度
            ├─ track_ang_vel_z_exp()          # 追踪角速度
            ├─ alive()                        # 活着奖励
            ├─ feet_slide()                   # 【传感器】读 contact_forces.data
            │                                   └─ net_forces_w_history
            ├─ feet_air_time_positive_biped() # 【传感器】读 contact_forces.data
            │                                   └─ current_air_time
            ├─ feet_clearance()               # 脚部抬起高度
            ├─ feet_gait()                    # 【传感器】步态对齐奖励
            │                                   └─ contact_forces.data.current_contact_time
            ├─ undesired_contacts()           # 【传感器】异常接触惩罚
            │                                   └─ contact_forces.data.net_forces_w_history
            └─ ... 其他奖励项
            
            rewards = sum(all_reward_terms)  # shape: [4096]
            
            【环节 A6】终止条件判断（TerminationManager）
            ├─ time_out()                     # 时间超过 episode_length
            ├─ base_contact()                 # 【传感器】基座接触
            │                                   └─ contact_forces.data
            ├─ base_height()                  # 高度过低
            └─ bad_orientation()              # 翻覆
            
            dones = [time_out, base_contact, base_height, bad_orientation]
            # shape: [4096]
            
            【环节 A7】环境重置（对那些 done=True 的环境）
            ├─ events["reset_base"](reset)          # 重置基座位置/速度
            ├─ events["reset_robot_joints"](reset)  # 重置关节位置
            └─ 其他重置事件
        
        # ∘ 评估网络前向传播（Critic）
        values = critic(obs)  # shape: [4096, 1]
        
        # ∘ 记录数据
        buffer.append({
            obs, actions, rewards, dones, values, log_probs, next_obs, ...
        })
    
    # 【阶段 B】样本收集完成，计算 GAE（广义优势估计）和 TD 目标
    advantages, td_targets = compute_gae(rewards, values, dones)
    
    # 【阶段 C】执行 num_learning_epochs=5 次更新
    for epoch in range(num_learning_epochs):
        
        # 将数据分为 num_mini_batches=4 个批次
        for mini_batch in split_into_batches(buffer, 4):
            
            # ∘ Actor 损失 + 熵奖励
            actor_loss = compute_actor_loss(mini_batch, actor, old_log_probs)
            
            # ∘ Critic 损失
            critic_loss = compute_critic_loss(mini_batch, critic)
            
            # ∘ 反向传播与梯度更新
            total_loss = actor_loss + critic_loss
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm=1.0)
            optimizer.step()
    
    # 【阶段 D】学习率调度与日志记录
    scheduler.step()  # 调整学习率（adaptive schedule）
    
    if iteration % save_interval == 0:
        save_checkpoint(f"model_{iteration}.pt", runner)
    
    if iteration % log_interval == 0:
        print(f"Iteration {iteration}: avg_reward={...}, avg_loss={...}")
```

---

### 5.3 训练循环中的传感器数据流

**高度扫描传感器（height_scanner）的数据流：**
```
物理步 → RayCaster 发射射线 → 射线命中地面 → 获得 ray_hits_w
                                    ↓
                        height_scan_ch() 函数处理
                                    ↓
                    sensor_z - hit_z - offset = 相对高度
                                    ↓
                        返回 [4096, H, W, 1] 张量
                                    ↓
                        观测向量中包含 height_scan
                                    ↓
                    Actor/Critic 网络使用这个高度信息
                                    ↓
                    策略学会如何跨越障碍（通过高度图）
```

**接触力传感器（contact_forces）的数据流：**
```
物理步 → 每个 body 受到接触力 → ContactSensor 记录力的大小和方向
                            ↓
                 记录到 data.net_forces_w_history
                 记录 data.current_contact_time
                 记录 data.current_air_time
                            ↓
        在各个 Reward Term 中使用：
        ├─ feet_slide(): 脚部滑移 = ∥v_feet∥ × is_in_contact
        ├─ feet_air_time(): 腾空时间奖励
        ├─ feet_gait(): 步态周期对齐
        └─ undesired_contacts(): 检测非脚部接触
                            ↓
        在 Termination 中使用：
        └─ base_contact(): 基座接触地面 = 终止
                            ↓
        训练期间，网络学会:
        ├─ 如何控制脚部接触/离地时序
        ├─ 如何减少滑移
        └─ 如何避免非脚部接触
```

---

## 第六阶段：训练完成

### 6.1 关闭环境

```
文件: scripts/rsl_rl/train.py（第 198 行）
```

**调用链：**
```python
env.close()
# 清理资源
```

---

### 6.2 关闭模拟器

```
文件: scripts/rsl_rl/train.py（第 206 行）
```

**调用链：**
```python
simulation_app.close()
# 关闭 Isaac Sim 应用
```

---

## 完整函数调用图（逻辑视图）

```
main()
├─ cli_args.update_rsl_rl_cfg()
├─ gym.make()
│  ├─ 查找 gym.register 注册表
│  └─ ManagerBasedRLEnv.__init__()
│     ├─ Scene 创建（机器人、地形、传感器）
│     ├─ ObservationManager 初始化
│     │  └─ 注册各个 ObsTerm 回调（mdp.height_scan, mdp.base_ang_vel, ...）
│     ├─ RewardManager 初始化
│     │  └─ 注册各个 RewTerm 回调（mdp.feet_air_time, mdp.feet_slide, ...）
│     ├─ TerminationManager 初始化
│     │  └─ 注册各个 DoneTerm 回调（mdp.illegal_contact, ...）
│     ├─ CommandManager 初始化
│     └─ env.reset()  # 初始化环境
│
├─ RslRlVecEnvWrapper()  # 包装环境
│
├─ OnPolicyRunner()  # 创建训练器
│  ├─ Actor 网络初始化
│  ├─ Critic 网络初始化
│  └─ 优化器初始化
│
└─ runner.learn()  # 训练循环
   └─ 对于每次迭代 (0 ~ 9999):
      ├─ 第 1 ~ 24 步：收集经验
      │  └─ env.step() × num_steps_per_env × num_envs
      │     ├─ 物理模拟
      │     ├─ 传感器更新（height_scanner, contact_forces）
      │     ├─ 事件触发（randomization）
      │     ├─ 观测生成（ObservationManager 调用各个 ObsTerm）
      │     ├─ 奖励计算（RewardManager 调用各个 RewTerm）
      │     ├─ 终止判断（TerminationManager）
      │     └─ 重置处理
      ├─ 优势函数计算（GAE）
      └─ 策略网络更新（5 个 epoch × 4 个 mini-batch）
         └─ actor.forward(), critic.forward()
         └─ loss.backward(), optimizer.step()
```

---

## 关键的 MDP 函数列表

### ObservationManager 中的函数调用
```
legged_lab/tasks/locomotion/velocity/mdp/observations.py
├─ height_scan()         # 普通高度扫描
└─ height_scan_ch()      # 2D 网格高度扫描（用于 Scandots）

legged_lab/tasks/locomotion/velocity/mdp/observations.py (Isaac Lab 内置)
├─ base_lin_vel()        # 基座线速度
├─ base_ang_vel()        # 基座角速度
├─ projected_gravity()   # 重力投影
├─ generated_commands()  # 生成的速度命令
├─ joint_pos_rel()       # 相对关节位置
├─ joint_vel_rel()       # 相对关节速度
└─ last_action()         # 上一步动作
```

### RewardManager 中的函数调用
```
legged_lab/tasks/locomotion/velocity/mdp/rewards.py
├─ track_lin_vel_xy_yaw_frame_exp()      # 线速度追踪
├─ track_ang_vel_z_world_exp()           # 角速度追踪
├─ is_alive()                             # 活着奖励
├─ feet_air_time_positive_biped()        # 腾空时间（使用 contact_forces）
├─ feet_slide()                           # 滑移惩罚（使用 contact_forces）
├─ feet_gait()                            # 步态对齐（使用 contact_forces）
├─ foot_clearance_reward()                # 脚部抬起奖励
├─ joint_deviation_l1()                   # 关节偏差（手臂等）
├─ undesired_contacts()                   # 异常接触惩罚（使用 contact_forces）
├─ lin_vel_z_l2()                         # Z 方向速度惩罚
├─ ang_vel_xy_l2()                        # XY 方向角速度惩罚
├─ flat_orientation_l2()                  # 平坦姿态惩罚
├─ base_height_l2()                       # 基座高度惩罚（使用 height_scanner）
├─ joint_vel_l2()                         # 关节速度惩罚
├─ joint_acc_l2()                         # 关节加速度惩罚
├─ action_rate_l2()                       # 动作变化率惩罚
├─ joint_pos_limits()                     # 关节限制惩罚
└─ joint_energy()                         # 能量消耗惩罚
```

### TerminationManager 中的函数调用
```
legged_lab/tasks/locomotion/velocity/mdp/terminations.py
├─ time_out()                 # 时间超时
├─ illegal_contact()          # 【传感器】非法接触（使用 contact_forces）
├─ root_height_below_minimum() # 基座高度过低
└─ bad_orientation()          # 翻覆判定
```

---

## 总结：关键阶段执行时机

| 阶段 | 执行时机 | 调用函数 |
|------|---------|---------|
| 参数解析 | 脚本开始 | `argparse.parse_known_args()` |
| 模拟器启动 | 参数解析后 | `AppLauncher()` |
| 配置加载 | main() 装饰器 | `@hydra_task_config()` |
| 环境创建 | main() 内 | `gym.make()` → `ManagerBasedRLEnv.__init__()` |
| 传感器初始化 | 环境创建时 | `height_scanner`, `contact_forces` 初始化 |
| 环境重置 | 环境创建完成 | `env.reset()` |
| 网络构建 | Runner 创建 | `OnPolicyRunner()` → Actor/Critic 初始化 |
| **每个训练步** | runner.learn() 内 | `env.step()` → 观测/奖励/终止计算 |
| **每个物理步** | env.step() 内 | 物理模拟、传感器更新、事件触发 |
| **每次迭代（多步后）** | 收集完数据后 | 策略网络更新、学习率调度 |
| 清理 | 训练结束 | `env.close()`, `simulation_app.close()` |

