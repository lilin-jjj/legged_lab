# 训练执行流程 - 快速查阅版

## 一句话总结
```
命令行参数 → 启动模拟器 → 加载配置 → 创建环境(传感器初始化) → 
构建神经网络 → 循环执行(物理步进 + 传感器更新 + 观测计算 + 奖励计算 + 网络更新)
```

---

## 执行流程图（简洁版）

```
python train.py --task=LeggedLab-Isaac-Velocity-Rough-G1-v0 --headless
                                  ↓
                    ┌─────────────────────────┐
                    │  1️⃣  参数解析          │
                    │  argparse.parse_args()  │
                    └─────────────┬───────────┘
                                  ↓
                    ┌─────────────────────────┐
                    │  2️⃣  启动模拟器         │
                    │  AppLauncher(args_cli)  │
                    │  (Isaac Sim 后台启动)   │
                    └─────────────┬───────────┘
                                  ↓
                    ┌─────────────────────────┐
                    │  3️⃣  加载配置          │
                    │  @hydra_task_config()   │
                    │  env_cfg = G1RoughEnvCfg│
                    │  agent_cfg = PPORunnerCfg│
                    └─────────────┬───────────┘
                                  ↓
                    ┌─────────────────────────────┐
        ┌───────────│  4️⃣  创建环境            │
        │           │  env = gym.make(...)       │
        │           │  ManagerBasedRLEnv.__init__│
        │           └─────────────┬───────────────┘
        │                         ↓
        │           【场景创建】【传感器初始化】
        │           ├─ 机器人 (G1, 27 DOF)
        │           ├─ 地形 (粗糙地形生成)
        │           ├─ 传感器 1: height_scanner ← ⭐ RayCaster
        │           ├─ 传感器 2: contact_forces ← ⭐ ContactSensor
        │           ├─ 观测管理器 (ObservationManager)
        │           │  ├─ 注册 height_scan() 回调
        │           │  └─ 注册其他观测回调
        │           ├─ 奖励管理器 (RewardManager)
        │           │  ├─ 注册 feet_slide() ← 使用 contact_forces
        │           │  ├─ 注册 feet_air_time() ← 使用 contact_forces
        │           │  └─ 注册其他奖励回调
        │           ├─ 终止管理器 (TerminationManager)
        │           │  ├─ 注册 illegal_contact() ← 使用 contact_forces
        │           │  └─ 注册其他终止回调
        │           └─ 初始化重置 (env.reset())
        │                         ↓
        │           ┌─────────────────────────────┐
        └──────────→│  5️⃣  包装环境             │
                    │  RslRlVecEnvWrapper(env)    │
                    └─────────────┬───────────────┘
                                  ↓
                    ┌─────────────────────────────┐
                    │  6️⃣  创建 Runner          │
                    │  OnPolicyRunner(...)        │
                    │  ├─ Actor 网络: [obs] → [action]
                    │  ├─ Critic 网络: [obs] → [value]
                    │  └─ 优化器初始化
                    └─────────────┬───────────────┘
                                  ↓
                    ┌─────────────────────────────┐
                    │  7️⃣  训练循环            │
                    │  runner.learn(10000 iters)  │
                    └─────────────┬───────────────┘
                                  ↓
        ┌─────────────────────────────────────────────────┐
        │      FOR iteration in range(10000):             │
        │                                                 │
        │  ┌──── 阶段 A: 收集经验 (24 步 × 4096 环境) ───┐│
        │  │                                             ││
        │  │  FOR step in range(24):                     ││
        │  │    FOR env in parallel_envs:                ││
        │  │                                             ││
        │  │      obs = env.obs  ← 从上一步保存的观测   ││
        │  │      action = actor(obs)  ← 策略网络推理   ││
        │  │      env.step(action)  ← 【关键！执行以下】││
        │  │      │                                      ││
        │  │      ├─ 【物理模拟】                        ││
        │  │      │  run_physics_simulation(decimation) ││
        │  │      │  应用关节 PD 控制器到目标位置        ││
        │  │      │  计算重力、碰撞、约束                ││
        │  │      │                                      ││
        │  │      ├─ 【传感器更新】⭐⭐⭐              ││
        │  │      │  height_scanner.update()            ││
        │  │      │  ├─ 发射射线到地面                  ││
        │  │      │  ├─ 读取射线命中点 (ray_hits_w)    ││
        │  │      │  └─ 存储到 data.ray_hits_w          ││
        │  │      │                                      ││
        │  │      │  contact_forces.update()            ││
        │  │      │  ├─ 计算每个 body 的接触力          ││
        │  │      │  ├─ 存储到 data.net_forces_w_history││
        │  │      │  ├─ 更新 data.current_air_time      ││
        │  │      │  └─ 更新 data.current_contact_time  ││
        │  │      │                                      ││
        │  │      ├─ 【事件触发】Randomization          ││
        │  │      │  (摩擦系数、质量、外力等随机化)     ││
        │  │      │                                      ││
        │  │      ├─ 【观测生成】ObservationManager ⭐  ││
        │  │      │  next_obs = [                        ││
        │  │      │    base_ang_vel(),                   ││
        │  │      │    projected_gravity(),             ││
        │  │      │    generated_commands(),            ││
        │  │      │    joint_pos_rel(),                 ││
        │  │      │    joint_vel_rel(),                 ││
        │  │      │    last_action(),                   ││
        │  │      │    height_scan(),  ← 【使用传感器】││
        │  │      │      └─ 从 height_scanner 读数据    ││
        │  │      │      └─ 转换为相对高度值            ││
        │  │      │  ]                                   ││
        │  │      │  shape: [4096, obs_dim]              ││
        │  │      │                                      ││
        │  │      ├─ 【奖励计算】RewardManager ⭐      ││
        │  │      │  rewards = sum([                     ││
        │  │      │    track_lin_vel_xy_exp(),          ││
        │  │      │    track_ang_vel_z_exp(),           ││
        │  │      │    alive(),                          ││
        │  │      │    feet_slide(),  ← 【使用传感器】  ││
        │  │      │      └─ contact_forces.data.net...  ││
        │  │      │    feet_air_time(),  ← 【使用传感器】││
        │  │      │      └─ contact_forces.data.air...  ││
        │  │      │    feet_gait(),   ← 【使用传感器】  ││
        │  │      │      └─ contact_forces.data.contact...││
        │  │      │    undesired_contacts(), ← 【使用传感器】││
        │  │      │      └─ contact_forces.data.net...  ││
        │  │      │    base_height_l2(), ← 【使用传感器】││
        │  │      │      └─ height_scanner 数据          ││
        │  │      │    ... 其他奖励项                    ││
        │  │      │  ])                                  ││
        │  │      │  shape: [4096]                       ││
        │  │      │                                      ││
        │  │      ├─ 【终止判断】TerminationManager ⭐  ││
        │  │      │  dones = [                           ││
        │  │      │    time_out(),                       ││
        │  │      │    illegal_contact(), ← 【使用传感器】││
        │  │      │      └─ contact_forces.data          ││
        │  │      │    root_height_below_minimum(),      ││
        │  │      │    bad_orientation(),                ││
        │  │      │  ]                                   ││
        │  │      │  shape: [4096]                       ││
        │  │      │                                      ││
        │  │      ├─ 【重置处理】                        ││
        │  │      │  for env_id where dones[env_id]:     ││
        │  │      │    reset_base()                      ││
        │  │      │    reset_robot_joints()             ││
        │  │      │                                      ││
        │  │      └─ 存储: obs, action, reward, done,    ││
        │  │          next_obs, value, log_prob         ││
        │  │                                             ││
        │  └─ (重复 24 步)                              ││
        │                                                 │
        │  ┌──── 阶段 B: 计算优势函数 ─────────────────┐│
        │  │  advantages, returns = compute_gae(       ││
        │  │    rewards, values, dones,                 ││
        │  │    gamma=0.99, lam=0.95                   ││
        │  │  )                                         ││
        │  └────────────────────────────────────────────┘│
        │                                                 │
        │  ┌──── 阶段 C: 策略网络更新 (5 epochs) ──────┐│
        │  │                                             ││
        │  │  FOR epoch in range(5):                     ││
        │  │    FOR mini_batch in split_data(4 batches): ││
        │  │                                             ││
        │  │      # 【步骤 1】重新计算对数概率           ││
        │  │      log_probs, entropy = actor(            ││
        │  │        mini_batch["obs"]                    ││
        │  │      )                                      ││
        │  │                                             ││
        │  │      # 【步骤 2】Critic 预测价值           ││
        │  │      pred_values = critic(                  ││
        │  │        mini_batch["obs"]                    ││
        │  │      )                                      ││
        │  │                                             ││
        │  │      # 【步骤 3】计算损失函数              ││
        │  │      actor_loss = compute_ppo_loss(         ││
        │  │        log_probs,                           ││
        │  │        mini_batch["log_probs"],             ││
        │  │        advantages,                          ││
        │  │        clip_param=0.2                       ││
        │  │      ) - 0.01 * entropy  # 熵奖励           ││
        │  │                                             ││
        │  │      value_loss = MSE(pred_values, returns) ││
        │  │                                             ││
        │  │      total_loss = actor_loss + value_loss   ││
        │  │                                             ││
        │  │      # 【步骤 4】反向传播与更新           ││
        │  │      optimizer.zero_grad()                  ││
        │  │      total_loss.backward()                  ││
        │  │      nn.utils.clip_grad_norm_(              ││
        │  │        actor.parameters(),                  ││
        │  │        max_grad_norm=1.0                    ││
        │  │      )                                      ││
        │  │      optimizer.step()                       ││
        │  │                                             ││
        │  └─────────────────────────────────────────────┘│
        │                                                 │
        │  ┌──── 阶段 D: 学习率调度与保存 ──────────────┐│
        │  │  if iteration % save_interval == 0:        ││
        │  │    save_checkpoint(f"model_{iteration}.pt") ││
        │  │  scheduler.step()  # 调整学习率            ││
        │  │  print(stats)  # 打印日志                   ││
        │  └─────────────────────────────────────────────┘│
        │                                                 │
        └─────────────────────────────────────────────────┘
                                  ↓
                    ┌─────────────────────────────┐
                    │  8️⃣  训练完成              │
                    │  env.close()                │
                    │  simulation_app.close()     │
                    └─────────────────────────────┘
```

---

## 按调用顺序列出的关键函数（从入口到循环）

### 阶段 0: 启动前
| 顺序 | 函数/步骤 | 位置 | 说明 |
|------|---------|------|------|
| 1 | `argparse.ArgumentParser()` | train.py:14-36 | 定义命令行参数 |
| 2 | `parser.parse_known_args()` | train.py:36 | 解析参数 |
| 3 | `cli_args.add_rsl_rl_args(parser)` | train.py:35 | 添加 rsl-rl 参数 |
| 4 | `AppLauncher.add_app_launcher_args(parser)` | train.py:36 | 添加启动器参数 |

### 阶段 1: 模拟器启动
| 5 | `AppLauncher(args_cli)` | train.py:44 | 启动 Isaac Sim |
| 6 | `app_launcher.app` | train.py:45 | 获取模拟器对象 |
| 7 | `metadata.version("rsl-rl-lib")` | train.py:55 | 检查 rsl-rl 版本 |

### 阶段 2: 配置加载与 main 函数调用
| 8 | `@hydra_task_config(task, agent)` | train.py:107 | 装饰器加载配置 |
| 9 | `main(env_cfg, agent_cfg)` | train.py:108+ | main 函数执行 |

### 阶段 3: 配置初始化
| 10 | `cli_args.update_rsl_rl_cfg()` | train.py:110 | 用 CLI 参数覆盖配置 |
| 11 | `env_cfg.scene.num_envs = ...` | train.py:111 | 设置环境数量 |
| 12 | `env_cfg.seed = ...` | train.py:115 | 设置随机种子 |
| 13 | `datetime.now().strftime()` | train.py:129 | 生成日志时间戳 |

### 阶段 4: 环境创建（⭐ 关键）
| 14 | `gym.make(task, cfg=env_cfg)` | train.py:150 | 创建环境 |
| → | **进入 ManagerBasedRLEnv.__init__()** |  | *来自 isaaclab 库* |
| 14a | `Scene()` 初始化 | isaaclab | 创建场景、物理引擎 |
| 14b | `height_scanner` 初始化 | isaaclab | **初始化 RayCaster 传感器** |
| 14c | `contact_forces` 初始化 | isaaclab | **初始化 ContactSensor 传感器** |
| 14d | `ObservationManager()` | isaaclab | 注册观测回调 |
| 14e | `RewardManager()` | isaaclab | 注册奖励回调 |
| 14f | `TerminationManager()` | isaaclab | 注册终止回调 |
| 14g | `env.reset()` | isaaclab | 环境初始化重置 |

### 阶段 5: 环境包装与 Runner 创建
| 15 | `RslRlVecEnvWrapper(env)` | train.py:169 | 包装环境 |
| 16 | `OnPolicyRunner(env, cfg)` | train.py:172 | 创建 PPO runner |
| 16a | Actor 网络初始化 | rsl_rl | [obs_dim] → [action_dim] |
| 16b | Critic 网络初始化 | rsl_rl | [obs_dim] → [1] |
| 16c | 优化器初始化 | rsl_rl | Adam 优化器 |

### 阶段 6: 训练循环核心（⭐ 最频繁）
| 17 | `runner.learn(10000)` | train.py:195 | 训练主循环 |
| → | **FOR iteration in range(10000):** |  |  |
| 18 | `env.step(action)` | rsl_rl | **【关键】执行 24 × 4096 次** |
| → | **进入 env.step()** |  | *来自 isaaclab* |
| 18a | `physics_substep()` | isaaclab | 物理模拟（4 次） |
| 18b | `height_scanner.update()` | isaaclab | **射线扫描传感器更新** ⭐ |
| 18c | `contact_forces.update()` | isaaclab | **接触力传感器更新** ⭐ |
| 18d | `events.trigger()` | isaaclab | 随机化事件 |
| 18e | `obs_mgr.compute()` | isaaclab | **调用观测回调** ⭐ |
| 18e-i | `mdp.height_scan()` | velocity/mdp/observations.py | **读取 height_scanner 数据** |
| 18f | `rew_mgr.compute()` | isaaclab | **调用奖励回调** ⭐ |
| 18f-i | `mdp.feet_slide()` | velocity/mdp/rewards.py | **读取 contact_forces 数据** |
| 18f-ii | `mdp.feet_air_time()` | velocity/mdp/rewards.py | **读取 contact_forces 数据** |
| 18f-iii | `mdp.feet_gait()` | velocity/mdp/rewards.py | **读取 contact_forces 数据** |
| 18f-iv | `mdp.undesired_contacts()` | velocity/mdp/rewards.py | **读取 contact_forces 数据** |
| 18g | `term_mgr.compute()` | isaaclab | **调用终止回调** ⭐ |
| 18g-i | `mdp.illegal_contact()` | velocity/mdp/terminations.py | **读取 contact_forces 数据** |
| 18h | `reset()` | isaaclab | 重置 done=True 的环境 |
| 19 | `actor.forward(obs)` | rsl_rl | 策略网络推理 |
| 20 | `critic.forward(obs)` | rsl_rl | 价值网络推理 |
| 21 | `compute_gae()` | rsl_rl | 计算优势函数 |
| 22 | `actor.forward(obs)` (重新计算) | rsl_rl | 策略网络重新推理（多 epoch） |
| 23 | `critic.forward(obs)` | rsl_rl | 价值网络重新推理 |
| 24 | `loss.backward()` | pytorch | 反向传播 |
| 25 | `optimizer.step()` | pytorch | 梯度更新 |
| 26 | `scheduler.step()` | pytorch | 学习率调度 |

### 阶段 7: 清理
| 27 | `env.close()` | train.py:198 | 关闭环境 |
| 28 | `simulation_app.close()` | train.py:206 | 关闭模拟器 |

---

## 传感器数据在高频循环中的用途

### height_scanner（高度扫描传感器）
```
env.step() 内：
  ├─ height_scanner.update()  ← 射线扫描更新 (update_period = 0.02s)
  │  └─ data.ray_hits_w: [4096, 128, 3] 射线命中点坐标
  │
  └─ obs_mgr.compute()
     └─ mdp.height_scan()  ← 观测计算回调
        └─ height = sensor_z - ray_z - offset
        └─ 返回 [4096, 128] 高度值 → 输入策略网络

reward_mgr.compute()
├─ mdp.base_height_l2()  ← 奖励回调
   └─ 使用 height_scanner 数据
```

### contact_forces（接触力传感器）
```
env.step() 内：
  ├─ contact_forces.update()  ← 接触力更新 (update_period = 0.005s)
  │  ├─ data.net_forces_w_history: [4096, 3, num_bodies, 3]
  │  ├─ data.current_air_time: [4096, num_bodies]
  │  └─ data.current_contact_time: [4096, num_bodies]
  │
  ├─ reward_mgr.compute()
  │  ├─ mdp.feet_slide()  ← 读 net_forces_w_history
  │  ├─ mdp.feet_air_time()  ← 读 current_air_time
  │  ├─ mdp.feet_gait()  ← 读 current_contact_time
  │  └─ mdp.undesired_contacts()  ← 读 net_forces_w_history
  │
  └─ term_mgr.compute()
     └─ mdp.illegal_contact()  ← 读 net_forces_w_history
```

---

## 常见问题

**Q: 为什么传感器要在 env.step() 内更新？**
A: 因为传感器直接依赖物理模拟结果。每次物理步进后，传感器会读取最新的物理数据（射线命中点、接触力等），这样观测和奖励才能准确反映当前物理状态。

**Q: 为什么 height_scanner 的 update_period 是 0.02s 而 contact_forces 是 0.005s？**
A: 高度扫描不需要那么频繁（机器人运动相对缓慢），而接触力需要高频更新来准确捕捉脚步离接触的瞬间。

**Q: 为什么每个 iteration 要跑 24 步而不是 1 步？**
A: 这叫 "收集轨迹"。PPO 算法需要收集足够多的样本（24 步 × 4096 环境 = 98304 个转移）来计算优势函数和更新策略，这样才能保证数据的多样性和梯度的稳定性。

**Q: Actor 和 Critic 为什么要各有一个网络？**
A: Actor 学习 "应该采取什么动作"，Critic 学习 "当前状态有多好"。这样 Critic 能帮助 Actor 更好地评估动作的优势（advantage），加快收敛。

