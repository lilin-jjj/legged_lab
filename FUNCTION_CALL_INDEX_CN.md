# 函数调用快速索引表

## 按执行顺序的完整函数列表

### 【前置阶段】参数解析与启动

```
1. argparse.ArgumentParser()
   位置: train.py:14-36
   功能: 定义命令行参数解析器
   输入: 无
   输出: parser 对象

2. cli_args.add_rsl_rl_args(parser)
   位置: scripts/rsl_rl/cli_args.py
   功能: 向 parser 添加 rsl-rl 相关参数
   输入: parser
   输出: parser (修改后)

3. AppLauncher.add_app_launcher_args(parser)
   位置: isaaclab.app
   功能: 向 parser 添加 Isaac Sim 启动参数
   输入: parser
   输出: parser (修改后)

4. parser.parse_known_args()
   位置: train.py:36
   功能: 解析命令行参数
   输入: sys.argv (默认)
   输出: args_cli, hydra_args
   关键参数: task="LeggedLab-Isaac-Velocity-Rough-G1-v0", headless=True

5. importlib.metadata.version()
   位置: train.py:55
   功能: 检查 rsl-rl 库版本
   输入: "rsl-rl-lib"
   输出: version 字符串
```

---

### 【阶段 1】Isaac Sim 启动

```
6. AppLauncher(args_cli)
   位置: train.py:44
   功能: 启动/连接 Isaac Sim 应用
   输入: args_cli (命令行参数)
   输出: app_launcher 对象
   初始化内容:
   ├─ Omniverse Nucleus 连接
   ├─ 物理引擎初始化
   └─ 渲染引擎初始化（headless 模式下无 GUI）

7. app_launcher.app
   位置: train.py:45
   功能: 获取 Isaac Sim 应用对象
   输入: app_launcher
   输出: simulation_app 对象
```

---

### 【阶段 2】配置加载与 main 函数

```
8. @hydra_task_config(args_cli.task, args_cli.agent)
   位置: train.py:107
   功能: Hydra 装饰器，根据 task id 加载配置
   输入: task_id, agent_id
   输出: 将 env_cfg, agent_cfg 注入到 main 函数
   
   具体过程:
   ├─ 根据 task="LeggedLab-Isaac-Velocity-Rough-G1-v0" 搜索 gym.register
   ├─ 找到 env_cfg_entry_point="...G1RoughEnvCfg"
   ├─ 加载 G1RoughEnvCfg 配置类
   ├─ 实例化 env_cfg 对象
   ├─ 找到 rsl_rl_cfg_entry_point="...G1RoughPPORunnerCfg"
   ├─ 加载 G1RoughPPORunnerCfg 配置类
   └─ 实例化 agent_cfg 对象

9. main(env_cfg, agent_cfg)
   位置: train.py:108-206
   功能: 训练主函数
   输入: env_cfg (ManagerBasedRLEnvCfg), agent_cfg (RslRlBaseRunnerCfg)
   输出: None (训练结束后返回)
```

---

### 【阶段 3】配置参数调整

```
10. cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    位置: scripts/rsl_rl/cli_args.py
    功能: 用命令行参数覆盖配置
    输入: agent_cfg, args_cli
    输出: agent_cfg (修改后)
    覆盖项:
    ├─ max_iterations (如果指定了 --max_iterations)
    ├─ device (如果指定了 --device)
    └─ 其他 rsl-rl 参数

11. env_cfg.scene.num_envs = args_cli.num_envs or env_cfg.scene.num_envs
    位置: train.py:111
    功能: 设置并行环境数量
    输入: args_cli.num_envs (或默认 4096)
    输出: env_cfg.scene.num_envs = 4096 (本例)

12. env_cfg.seed = agent_cfg.seed
    位置: train.py:115
    功能: 设置随机种子
    输入: agent_cfg.seed (通常为随机或指定值)
    输出: env_cfg.seed

13. datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    位置: train.py:129
    功能: 生成日志时间戳
    输入: 无
    输出: log_dir 时间戳字符串 (例如 "2025-11-25_14-30-45")
```

---

### 【阶段 4】环境创建（⭐⭐⭐ 关键）

```
14. gym.make(
      args_cli.task,
      cfg=env_cfg,
      render_mode="rgb_array" if args_cli.video else None
    )
    位置: train.py:150
    功能: 创建强化学习环境
    输入: task_id, env_cfg (G1RoughEnvCfg), render_mode
    输出: env (ManagerBasedRLEnv 实例)
    
    内部调用链:
    ├─ gym.registry 查找任务注册
    │  └─ entry_point="isaaclab.envs:ManagerBasedRLEnv"
    │
    └─ ManagerBasedRLEnv(cfg=env_cfg) 初始化
       │
       ├─ 【14.1】Scene 初始化 (isaaclab.scene)
       │  ├─ terrain: TerrainImporter (粗糙地形生成)
       │  │  └─ ROUGH_TERRAINS_CFG: 生成随机坡度、障碍、凹凸
       │  │
       │  ├─ robot: ArticulationCfg (G1_27DOF_CFG)
       │  │  ├─ 加载 G1 URDF 模型
       │  │  ├─ 27 个自由度（关节）
       │  │  ├─ 质量、摩擦系数等物理参数
       │  │  └─ 创建对应的 Articulation 对象
       │  │
       │  ├─【⭐ 传感器 1】height_scanner: RayCasterCfg
       │  │  ├─ prim_path="{ENV_REGEX_NS}/Robot/waist_yaw_link"
       │  │  ├─ offset=(0, 0, 20): 相对传感器位置（向上 20m）
       │  │  ├─ ray_alignment='yaw': 射线沿 yaw 轴对齐
       │  │  ├─ pattern_cfg=GridPatternCfg(resolution=0.1, size=[1.6, 1.0])
       │  │  │  └─ 生成网格射线（约 160×100 条射线）
       │  │  ├─ mesh_prim_paths=["/World/ground"]
       │  │  │  └─ 只扫描地面
       │  │  │
       │  │  ├─ 创建 RayCaster 对象
       │  │  └─ 初始化 data 字段:
       │  │     ├─ data.pos_w: [num_envs, 3] 传感器位置
       │  │     ├─ data.ray_hits_w: [num_envs, num_rays, 3] 射线命中点
       │  │     └─ ...其他字段
       │  │
       │  └─【⭐ 传感器 2】contact_forces: ContactSensorCfg
       │     ├─ prim_path="{ENV_REGEX_NS}/Robot/.*"
       │     │  └─ 匹配机器人所有 prim
       │     ├─ history_length=3: 记录最后 3 个时间步
       │     ├─ track_air_time=True: 追踪离地时间
       │     │
       │     ├─ 创建 ContactSensor 对象
       │     └─ 初始化 data 字段:
       │        ├─ data.net_forces_w_history: [num_envs, 3, num_bodies, 3]
       │        │  └─ 接触力向量历史
       │        ├─ data.current_contact_time: [num_envs, num_bodies]
       │        │  └─ 当前接触时间
       │        ├─ data.current_air_time: [num_envs, num_bodies]
       │        │  └─ 当前离地时间
       │        └─ data.last_air_time: [num_envs, num_bodies]
       │           └─ 上一次离地的累计时间
       │
       ├─ 【14.2】ObservationManager 初始化
       │  │
       │  ├─ policy obs group 注册:
       │  │  ├─ ObsTerm(mdp.base_ang_vel)
       │  │  ├─ ObsTerm(mdp.projected_gravity)
       │  │  ├─ ObsTerm(mdp.generated_commands)
       │  │  ├─ ObsTerm(mdp.joint_pos_rel)
       │  │  ├─ ObsTerm(mdp.joint_vel_rel)
       │  │  ├─ ObsTerm(mdp.last_action)
       │  │  └─【⭐】ObsTerm(mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")})
       │  │     └─ 告诉这个 ObsTerm 使用哪个传感器的数据
       │  │
       │  └─ critic obs group 注册:
       │     ├─ base_lin_vel, base_ang_vel, ... (类似 policy)
       │     └─ 但无噪声、无截断（critic 需要准确的特权观测）
       │
       ├─ 【14.3】RewardManager 初始化
       │  │
       │  ├─ RewTerm(mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0)
       │  │  └─ 追踪线速度奖励
       │  ├─ RewTerm(mdp.track_ang_vel_z_world_exp, weight=0.5)
       │  │  └─ 追踪角速度奖励
       │  ├─ RewTerm(mdp.is_alive, weight=0.15)
       │  │  └─ 活着奖励
       │  │
       │  ├─【⭐】RewTerm(mdp.feet_slide, weight=-0.2, params={
       │  │    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
       │  │  })
       │  │  └─ 脚部滑移惩罚（使用 contact_forces 传感器数据）
       │  │
       │  ├─【⭐】RewTerm(mdp.feet_air_time_positive_biped, weight=0.0, params={
       │  │    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
       │  │  })
       │  │  └─ 腾空时间奖励（使用 contact_forces 传感器数据）
       │  │
       │  ├─【⭐】RewTerm(mdp.feet_clearance, weight=1.0, params={
       │  │    "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*")
       │  │  })
       │  │  └─ 脚部抬起奖励
       │  │
       │  ├─【⭐】RewTerm(mdp.feet_gait, weight=0.5, params={
       │  │    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")
       │  │  })
       │  │  └─ 步态对齐奖励（使用 contact_forces 传感器数据）
       │  │
       │  ├─ RewTerm(mdp.joint_deviation_hip, weight=-1.0)
       │  │  └─ 关节偏差惩罚
       │  ├─ RewTerm(mdp.joint_deviation_arms, weight=-0.1)
       │  │  └─ 手臂关节偏差惩罚
       │  ├─ RewTerm(mdp.joint_deviation_waist, weight=-1.0)
       │  │  └─ 腰部关节偏差惩罚
       │  │
       │  ├─【⭐】RewTerm(mdp.base_height_l2, weight=-10.0, params={
       │  │    "target_height": 0.78,
       │  │    "sensor_cfg": SceneEntityCfg("height_scanner")
       │  │  })
       │  │  └─ 基座高度惩罚（使用 height_scanner 传感器）
       │  │
       │  ├─ RewTerm(mdp.lin_vel_z_l2, weight=0.0)
       │  ├─ RewTerm(mdp.ang_vel_xy_l2, weight=-0.05)
       │  ├─ RewTerm(mdp.flat_orientation_l2, weight=-1.0)
       │  ├─ RewTerm(mdp.joint_vel_l2, weight=-0.001)
       │  ├─ RewTerm(mdp.joint_acc_l2, weight=-1.25e-7)
       │  ├─ RewTerm(mdp.action_rate_l2, weight=-0.005)
       │  ├─ RewTerm(mdp.joint_pos_limits, weight=-5.0)
       │  ├─ RewTerm(mdp.joint_energy, weight=-2e-5)
       │  │
       │  └─【⭐】RewTerm(mdp.undesired_contacts, weight=-1, params={
       │     "threshold": 1,
       │     "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"])
       │  })
       │  └─ 不希望的接触惩罚（使用 contact_forces，排除脚踝）
       │
       ├─ 【14.4】TerminationManager 初始化
       │  │
       │  ├─ DoneTerm(mdp.time_out, time_out=True)
       │  │  └─ 时间超时终止
       │  │
       │  ├─【⭐】DoneTerm(mdp.illegal_contact, params={
       │  │    "sensor_cfg": SceneEntityCfg("contact_forces", body_names="waist_yaw_link"),
       │  │    "threshold": 1.0
       │  │  })
       │  │  └─ 基座非法接触终止（使用 contact_forces 传感器）
       │  │
       │  ├─ DoneTerm(mdp.root_height_below_minimum, params={"minimum_height": 0.2})
       │  │  └─ 基座高度过低终止
       │  │
       │  └─ DoneTerm(mdp.bad_orientation, params={"limit_angle": math.radians(45.0)})
       │     └─ 翻覆终止
       │
       ├─ 【14.5】CommandManager 初始化
       │  └─ base_velocity: UniformVelocityCommandCfg
       │     ├─ resampling_time_range=(10.0, 10.0)
       │     │  └─ 每 10 秒重新采样一次目标速度
       │     ├─ ranges: {lin_vel_x, lin_vel_y, ang_vel_z, heading}
       │     │  └─ 目标速度的范围
       │     └─ 产生目标速度命令供学习
       │
       ├─ 【14.6】EventManager 初始化 (Randomization)
       │  └─ 注册各种随机化事件:
       │     ├─ physics_material (startup): 摩擦系数随机化
       │     ├─ add_base_mass (startup): 质量随机化
       │     ├─ base_external_force_torque (reset): 外力推送
       │     ├─ reset_base (reset): 基座位置/速度随机化
       │     ├─ reset_robot_joints (reset): 关节位置随机化
       │     └─ push_robot (interval): 定期推送机器人
       │
       └─ 【14.7】env.reset()
          ├─ 触发 startup 事件
          ├─ 重置所有环境到初始状态
          ├─ 返回初始观测和信息字典
          └─ 返回值: obs (shape: [4096, obs_dim])
```

---

### 【阶段 5】环境包装与 Runner 创建

```
15. RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    位置: train.py:169
    功能: 将 Isaac Lab 环境适配为 rsl-rl 兼容格式
    输入: env (ManagerBasedRLEnv)
    输出: wrapped_env (rsl-rl 期望的格式)
    包装内容:
    ├─ obs tensor 形状转换
    ├─ reward tensor 形状转换
    ├─ action 动作裁剪（如果 clip_actions=True）
    └─ info 字典处理

16. OnPolicyRunner(
      env,
      agent_cfg.to_dict(),
      log_dir=log_dir,
      device=agent_cfg.device
    )
    位置: train.py:172 (来自 rsl_rl 库)
    功能: 创建 PPO 训练器
    输入: env, config dict, log_dir, device
    输出: runner (OnPolicyRunner 实例)
    初始化内容:
    │
    ├─ 【16.1】Actor 网络初始化
    │  ├─ 输入维度: obs_dim (例如 212)
    │  ├─ 隐藏层: [512, 256, 128]
    │  ├─ 输出维度: action_dim (27, 对应 G1 的 27 个关节)
    │  ├─ 激活函数: ELU
    │  ├─ 初始化策略
    │  └─ 参数初始化为小随机值
    │
    ├─ 【16.2】Critic 网络初始化
    │  ├─ 输入维度: obs_dim (例如 212)
    │  ├─ 隐藏层: [512, 256, 128]
    │  ├─ 输出维度: 1 (价值估计)
    │  ├─ 激活函数: ELU
    │  └─ 参数初始化为小随机值
    │
    ├─ 【16.3】优化器初始化
    │  ├─ Adam 优化器
    │  ├─ learning_rate=1.0e-3
    │  └─ 管理 Actor 和 Critic 的参数
    │
    ├─ 【16.4】学习率调度器初始化
    │  ├─ schedule="adaptive"
    │  ├─ 根据 KL 散度动态调整学习率
    │  └─ desired_kl=0.01
    │
    └─ 【16.5】算法参数保存
       ├─ clip_param=0.2 (PPO 剪切范围)
       ├─ entropy_coef=0.01 (熵奖励系数)
       ├─ value_loss_coef=1.0 (价值损失权重)
       ├─ gamma=0.99 (折扣因子)
       ├─ lam=0.95 (GAE 衰减因子)
       └─ max_grad_norm=1.0 (梯度裁剪)

17. runner.add_git_repo_to_log(__file__)
    位置: train.py:181 (来自 rsl_rl 库)
    功能: 保存当前 git 状态到日志目录
    输入: 源文件路径
    输出: 日志目录中的 .git 快照

18. dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    位置: train.py:189
    功能: 将环境配置保存为 YAML 文件
    输入: 文件路径, env_cfg 对象
    输出: env.yaml (包含所有环境参数)

19. dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    位置: train.py:190
    功能: 将 agent 配置保存为 YAML 文件
    输入: 文件路径, agent_cfg 对象
    输出: agent.yaml (包含所有学习参数)
```

---

### 【阶段 6】训练循环（高频调用）

```
20. runner.learn(
      num_learning_iterations=agent_cfg.max_iterations,  # 10000
      init_at_random_ep_len=True
    )
    位置: train.py:195 (来自 rsl_rl 库)
    功能: 执行 PPO 训练的主循环
    输入: 迭代次数, 初始化标志
    输出: None (训练完成)
    
    核心循环结构 (伪代码):
    ───────────────────────────────────────
    
    FOR iteration = 0 to 9999:
      
      # ──────────────────────────────────
      # 【步骤 A】收集经验
      # ──────────────────────────────────
      
      FOR step = 0 to 23:  # num_steps_per_env = 24
        
        # 【A.1】获取当前观测
        obs = env.obs
        # shape: [4096, obs_dim]
        # 其中 obs_dim 取决于观测配置，本例约 212
        
        # 【A.2】策略网络推理
        actions, log_probs = actor.forward(obs)
        # actions shape: [4096, 27]  (27 个关节)
        # log_probs shape: [4096]  (对数概率)
        
        # 【A.3】环境步进
        obs, rewards, dones, info = env.step(actions)
        # ⭐⭐⭐ 这一步内部发生大量事情（见下文）
        
        # 【A.4】评估网络推理
        values = critic.forward(obs)
        # values shape: [4096, 1]  (状态价值)
        
        # 【A.5】存储数据到回放缓冲
        buffer.push({
          "obs": obs,
          "actions": actions,
          "rewards": rewards,
          "dones": dones,
          "values": values,
          "log_probs": log_probs,
        })
      
      # ──────────────────────────────────
      # 【步骤 B】计算优势函数
      # ──────────────────────────────────
      
      advantages, returns = compute_gae(
        rewards=buffer["rewards"],         # [4096×24]
        values=buffer["values"],           # [4096×24, 1]
        dones=buffer["dones"],             # [4096×24]
        gamma=0.99,
        lam=0.95,
      )
      # advantages shape: [4096×24]
      # returns shape: [4096×24]
      
      # ──────────────────────────────────
      # 【步骤 C】策略更新 (5 epochs × 4 mini-batches)
      # ──────────────────────────────────
      
      FOR epoch = 0 to 4:  # num_learning_epochs = 5
        
        # 将数据随机打乱并分割成 4 个 mini-batch
        FOR mini_batch in split_data(buffer, num_mini_batches=4):
          
          # 【C.1】重新计算对数概率（基于新策略）
          new_log_probs, entropy = actor.forward(
            mini_batch["obs"],
            return_entropy=True,
          )
          # new_log_probs shape: [sample_size]
          # entropy shape: [sample_size]
          
          # 【C.2】重新计算价值估计（基于新 critic）
          new_values = critic.forward(mini_batch["obs"])
          # new_values shape: [sample_size, 1]
          
          # 【C.3】计算 PPO 损失
          # 策略损失 (Actor Loss)
          ratio = torch.exp(new_log_probs - mini_batch["log_probs"])
          surr1 = ratio * mini_batch["advantages"]
          surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * mini_batch["advantages"]
          actor_loss = -torch.min(surr1, surr2).mean()
          
          # 价值损失 (Critic Loss)
          value_loss = F.mse_loss(new_values, mini_batch["returns"])
          
          # 熵奖励（鼓励探索）
          entropy_loss = -entropy_coef * entropy.mean()
          
          # 总损失
          total_loss = actor_loss + value_loss + entropy_loss
          
          # 【C.4】反向传播与梯度更新
          optimizer.zero_grad()
          total_loss.backward()
          
          # 梯度裁剪（防止梯度爆炸）
          torch.nn.utils.clip_grad_norm_(
            actor.parameters(),
            max_norm=1.0
          )
          
          # 优化器步进
          optimizer.step()
      
      # ──────────────────────────────────
      # 【步骤 D】学习率调度与日志记录
      # ──────────────────────────────────
      
      # 学习率调度
      scheduler.step(current_kl_divergence)
      
      # 定期保存模型
      if iteration % save_interval == 0:
        save_checkpoint(f"model_{iteration}.pt", actor, critic, optimizer)
      
      # 打印训练进度
      if iteration % log_interval == 0:
        print(f"[Iteration {iteration}]")
        print(f"  Avg Reward: {avg_reward:.3f}")
        print(f"  Avg Episode Length: {avg_ep_len:.0f}")
        print(f"  Policy Loss: {policy_loss:.4f}")
        print(f"  Value Loss: {value_loss:.4f}")
    
    ───────────────────────────────────────
```

---

### 【关键】env.step(action) 内部执行流程

```
21. env.step(actions)  ← 这是最频繁调用的函数！
    位置: isaaclab (外部库)
    功能: 执行一个环境时间步
    输入: actions (shape: [4096, 27])
    输出: obs, rewards, dones, info
    
    内部执行步骤：
    ════════════════════════════════════════════════════════
    
    FOR each of 4096 parallel environments:
      
      # ┌─────────────────────────────────────────────────
      # │ 【步骤 21.1】物理模拟 (执行 4 次)
      # └─────────────────────────────────────────────────
      
      FOR phys_step = 0 to 3:  # decimation = 4
        
        # 使用 PD 控制器计算关节力矩
        joint_torques = controller.compute_torques(
          target_positions=actions[env_id],  # 目标位置
          current_positions=joint_pos,
          current_velocities=joint_vel,
          kp=stiffness,
          kd=damping,
        )
        
        # 应用力矩到物理引擎
        articulation.set_joint_efforts(joint_torques)
        
        # 模拟一个物理时间步 (sim.dt = 0.005 秒)
        physics_engine.step()
        
        # 更新关节位置/速度
        joint_pos, joint_vel, joint_acc = articulation.data.get_state()
      
      # ┌─────────────────────────────────────────────────
      # │ 【步骤 21.2】传感器数据更新 ⭐⭐⭐
      # └─────────────────────────────────────────────────
      
      # ────────────────────────────────────
      # 【⭐】传感器 1: height_scanner (RayCaster)
      # ────────────────────────────────────
      # 检查: update_period = decimation × sim.dt = 0.02 秒
      #       (约每 4 个物理步更新一次)
      
      if should_update_height_scanner:
        
        # 【21.2.1】获取传感器位置（基座位置 + offset）
        sensor_pos = base_pos + offset_pos
        # offset=(0, 0, 20): 向上偏移 20m
        
        # 【21.2.2】发射射线到地面
        FOR each ray in the grid pattern:
          
          ray_origin = sensor_pos
          ray_direction = normalize(offset + ray_grid_point)
          
          # 物理引擎射线投射
          hit_info = physics_engine.raycast(
            origin=ray_origin,
            direction=ray_direction,
            max_distance=25.0,  # 最大射线长度
          )
          
          if hit_info.hit:
            ray_hits_w[env_id, ray_idx] = hit_info.position  # [x, y, z]
          else:
            ray_hits_w[env_id, ray_idx] = [NaN, NaN, NaN]  # 未命中
        
        # 【21.2.3】存储到传感器数据结构
        height_scanner.data.pos_w[env_id] = sensor_pos  # [3]
        height_scanner.data.ray_hits_w[env_id] = ray_hits_w  # [128, 3]
      
      # ────────────────────────────────────
      # 【⭐】传感器 2: contact_forces (ContactSensor)
      # ────────────────────────────────────
      # 检查: update_period = sim.dt = 0.005 秒
      #       (每个物理步都更新)
      
      if should_update_contact_forces:
        
        # 【21.2.4】遍历所有被监视的 body
        FOR each body in tracked_bodies:
          
          # 从物理引擎获取接触信息
          contact_forces = physics_engine.get_contact_forces(body)
          # shape: [num_contacts, 3]  (世界坐标系)
          
          # 网接触力（所有接触点的力向量和）
          net_force = sum(contact_forces)  # [3]
          
          # 存储到历史缓冲区
          # 当前数据向后移一位，新数据放在第一位
          contact_forces.data.net_forces_w_history[env_id, 1:] = \
            contact_forces.data.net_forces_w_history[env_id, :-1]
          contact_forces.data.net_forces_w_history[env_id, 0] = net_force
          
          # ─────────────────────────────────
          # 追踪接触/离地时间
          # ─────────────────────────────────
          
          is_in_contact = len(contact_forces) > 0
          
          if is_in_contact:
            # 身体在接触
            contact_forces.data.current_contact_time[env_id, body_id] += sim.dt
            contact_forces.data.current_air_time[env_id, body_id] = 0.0
          else:
            # 身体不在接触（腾空）
            contact_forces.data.current_air_time[env_id, body_id] += sim.dt
            contact_forces.data.current_contact_time[env_id, body_id] = 0.0
          
          # 记录最后一次离地时间（用于 feet_air_time 奖励）
          if was_in_contact_last_step and not is_in_contact:
            contact_forces.data.last_air_time[env_id, body_id] = \
              contact_forces.data.current_air_time[env_id, body_id]
      
      # ┌─────────────────────────────────────────────────
      # │ 【步骤 21.3】事件触发 (Randomization)
      # └─────────────────────────────────────────────────
      
      # 检查是否需要触发各类事件
      
      # 模式 1: startup (仅在 env.reset() 时触发，但此处不重复触发)
      # 模式 2: reset (仅当 env done 时触发)
      # 模式 3: interval (定期触发)
      
      # 【21.3.1】interval 事件：push_robot
      if (env_step_count % push_interval) == 0:
        
        # 随机生成外力
        force = random.uniform(-0.5, 0.5, size=2)  # x, y 方向
        
        # 应用到基座
        base_link.add_force(force)
        # 这会在下一个物理步产生效果
      
      # 模式 2 的事件会在 done 时的 reset() 中处理
      
      # ┌─────────────────────────────────────────────────
      # │ 【步骤 21.4】观测生成 ⭐⭐⭐
      # │ 调用 ObservationManager
      # └─────────────────────────────────────────────────
      
      obs_dict = obs_manager.compute()
      
      # 【21.4.1】policy 观测组
      policy_obs = torch.cat([
        
        # ① base_ang_vel() - 基座角速度
        mdp.base_ang_vel(env),
        # → base.ang_vel_w  shape: [3]
        
        # ② projected_gravity() - 重力投影
        mdp.projected_gravity(env),
        # → 基座坐标系下的重力方向  shape: [3]
        
        # ③ generated_commands() - 速度命令
        mdp.generated_commands(env, command_name="base_velocity"),
        # → [lin_vel_x, lin_vel_y, ang_vel_z]  shape: [3]
        
        # ④ joint_pos_rel() - 相对关节位置
        mdp.joint_pos_rel(env),
        # → (current - default) / scale  shape: [27]
        
        # ⑤ joint_vel_rel() - 相对关节速度
        mdp.joint_vel_rel(env),
        # → current / scale  shape: [27]
        
        # ⑥ last_action() - 上一步动作
        mdp.last_action(env),
        # → 上一步的执行动作  shape: [27]
        
        # 【⭐】⑦ height_scan() - 高度扫描
        mdp.height_scan(env, sensor_cfg=SceneEntityCfg("height_scanner")),
        # 内部执行:
        # ├─ sensor = env.scene.sensors["height_scanner"]
        # ├─ sensor_z = sensor.data.pos_w[:, 2]
        # ├─ ray_z = sensor.data.ray_hits_w[..., 2]
        # ├─ height = sensor_z - ray_z - 0.5  (offset=0.5)
        # └─ return height  shape: [128] 或 [H, W, 1]（取决于 RayCaster 类型）
        
      ], dim=-1)
      # → policy_obs shape: [3 + 3 + 3 + 27 + 27 + 27 + 128 = 218]
      # (实际可能有噪声、截断等，所以数字可能不同)
      
      # 【21.4.2】critic 观测组（类似，但无噪声）
      critic_obs = torch.cat([
        mdp.base_lin_vel(env),
        mdp.base_ang_vel(env),
        ...
        mdp.height_scan(env, ...),  # 无噪声版本
      ], dim=-1)
      
      # 存储到环境状态
      obs_dict["policy"] = policy_obs
      obs_dict["critic"] = critic_obs
      
      # ┌─────────────────────────────────────────────────
      # │ 【步骤 21.5】奖励计算 ⭐⭐⭐
      # │ 调用 RewardManager
      # └─────────────────────────────────────────────────
      
      reward = 0.0
      
      # 【21.5.1】基本奖励项
      reward += 1.0 * mdp.track_lin_vel_xy_yaw_frame_exp(
        env,
        std=0.5,
        command_name="base_velocity",
      )
      # → 追踪目标线速度的奖励  返回标量
      
      reward += 0.5 * mdp.track_ang_vel_z_world_exp(
        env,
        command_name="base_velocity",
        std=0.5,
      )
      # → 追踪目标角速度的奖励  返回标量
      
      reward += 0.15 * mdp.is_alive(env)
      # → 活着就有 0.15 的奖励  返回 1.0 或 0.0
      
      # 【21.5.2】使用 height_scanner 的奖励
      reward -= 10.0 * mdp.base_height_l2(
        env,
        target_height=0.78,  # 目标高度 78cm
        sensor_cfg=SceneEntityCfg("height_scanner"),
      )
      # 内部:
      # ├─ sensor = env.scene.sensors["height_scanner"]
      # ├─ ray_hits_z = sensor.data.ray_hits_w[..., 2]
      # ├─ height = ray_hits_z.max() - offset  (获取最高接触点)
      # │   实际上是获取当前基座距地面的高度
      # └─ return ||height - target_height||^2  返回标量（惩罚）
      
      # 【⭐】【21.5.3】使用 contact_forces 的奖励
      
      # ① feet_slide() - 脚部滑移惩罚
      reward -= 0.2 * mdp.feet_slide(
        env,
        sensor_cfg=SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        asset_cfg=SceneEntityCfg("robot"),
      )
      # 内部:
      # ├─ contact_sensor = env.scene.sensors["contact_forces"]
      # ├─ 获取 feet 的 body_ids
      # ├─ net_forces = contact_sensor.data.net_forces_w_history[:, sensor_cfg.body_ids, :]
      # │  shape: [4096, history_length, num_feet, 3]
      # ├─ is_in_contact = norm(net_forces) > 1.0
      # ├─ feet_vel = robot.data.body_lin_vel_w[:, body_ids, :2]  (xy 方向)
      # └─ return sum(||feet_vel|| * is_in_contact)  返回标量（惩罚滑移）
      
      # ② feet_air_time_positive_biped() - 腾空时间奖励
      reward += 0.0 * mdp.feet_air_time_positive_biped(
        env,
        command_name="base_velocity",
        threshold=0.4,
        sensor_cfg=SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
      )
      # 内部:
      # ├─ contact_sensor = env.scene.sensors["contact_forces"]
      # ├─ air_time = contact_sensor.data.current_air_time[:, body_ids]
      # ├─ contact_time = contact_sensor.data.current_contact_time[:, body_ids]
      # ├─ 判断是否单脚支撑: sum(contact_time > 0) == 1
      # └─ 对于单脚支撑，返回 clamp(min(air_time, contact_time), max=0.4)
      
      # ③ feet_gait() - 步态对齐奖励
      reward += 0.5 * mdp.feet_gait(
        env,
        period=0.8,
        offset=[0.0, 0.5],
        threshold=0.55,
        command_name="base_velocity",
        sensor_cfg=SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
      )
      # 内部:
      # ├─ contact_sensor = env.scene.sensors["contact_forces"]
      # ├─ is_contact = contact_sensor.data.current_contact_time[:, body_ids] > 0
      # ├─ 生成周期为 0.8 秒的相位轨迹，两脚偏移 0.5 周期
      # ├─ 判断两脚的接触状态是否与期望相匹配
      # └─ 返回匹配度（奖励对齐的步态）
      
      # ④ undesired_contacts() - 不希望接触惩罚
      reward -= 1.0 * mdp.undesired_contacts(
        env,
        threshold=1,
        sensor_cfg=SceneEntityCfg(
          "contact_forces",
          body_names=["(?!.*ankle.*).*"]  # 排除 ankle，即非脚部
        ),
      )
      # 内部:
      # ├─ contact_sensor = env.scene.sensors["contact_forces"]
      # ├─ 过滤出非脚部 body 的接触力
      # ├─ has_undesired_contact = sum(||net_forces||) > threshold
      # └─ 返回惩罚（鼓励脚部接触，避免身体其他部分接触）
      
      # 【21.5.4】其他奖励项
      reward -= 1.0 * mdp.lin_vel_z_l2(env)  # Z 方向速度
      reward -= 0.05 * mdp.ang_vel_xy_l2(env)  # XY 角速度
      reward -= 0.001 * mdp.joint_vel_l2(env)  # 关节速度
      reward -= 1.25e-7 * mdp.joint_acc_l2(env)  # 关节加速度
      reward -= 0.005 * mdp.action_rate_l2(env)  # 动作变化率
      reward -= 5.0 * mdp.joint_pos_limits(env)  # 关节限制违反
      reward -= 2e-5 * mdp.joint_energy(env)  # 能量消耗
      reward -= 1.0 * mdp.joint_deviation_hip(env)  # 髋关节偏差
      reward -= 0.1 * mdp.joint_deviation_arms(env)  # 手臂关节偏差
      reward -= 1.0 * mdp.joint_deviation_waist(env)  # 腰部偏差
      reward -= 1.0 * mdp.foot_clearance_reward(env)  # 脚部抬起
      
      # 最终奖励（可能经过归一化）
      env.reward[env_id] = reward
      
      # ┌─────────────────────────────────────────────────
      # │ 【步骤 21.6】终止条件判断 ⭐⭐⭐
      # │ 调用 TerminationManager
      # └─────────────────────────────────────────────────
      
      done = False
      
      # ① time_out() - 超时终止
      if env_step_count[env_id] >= max_episode_length:
        done = True
        # 标记原因: "time_limit"
      
      # 【⭐】② illegal_contact() - 非法接触终止
      illegal_contact = mdp.illegal_contact(
        env,
        sensor_cfg=SceneEntityCfg("contact_forces", body_names="waist_yaw_link"),
        threshold=1.0,
      )
      # 内部:
      # ├─ contact_sensor = env.scene.sensors["contact_forces"]
      # ├─ 获取 waist (基座) 的接触力
      # ├─ net_force = contact_sensor.data.net_forces_w_history[:, body_id, :]
      # └─ illegal = norm(net_force) > threshold
      # → 返回 True 则基座接触地面，即失败
      if illegal_contact:
        done = True
      
      # ③ root_height_below_minimum() - 高度过低
      if base_pos[env_id, 2] < 0.2:  # z < 0.2m
        done = True
      
      # ④ bad_orientation() - 翻覆
      roll, pitch, yaw = quaternion_to_euler(base_quat)
      if abs(roll) > 45° or abs(pitch) > 45°:
        done = True
      
      env.done[env_id] = done
      
      # ┌─────────────────────────────────────────────────
      # │ 【步骤 21.7】环境重置处理
      # └─────────────────────────────────────────────────
      
      if done:
        
        # 触发 reset 类型的事件
        for event in events["reset"]:
          event.trigger(env_id)
          # 例如: reset_base() 会重置基座位置、速度
          
          # reset_base() 内部:
          # ├─ base_pos = random.uniform(-0.5, 0.5, size=2)
          # ├─ base_vel = [0, 0, 0]
          # ├─ base_quat = [0, 0, 0, 1]  (无旋转)
          # └─ 设置到物理引擎
          
          # reset_robot_joints() 内部:
          # ├─ 重置所有关节到默认位置
          # └─ 重置关节速度为 0
        
        # 重置环境计数器
        env_step_count[env_id] = 0
        episode_length[env_id] = 0
        
        # 重新计算观测（用于下一个 step）
        obs_dict["policy"][env_id] = compute_observations(env_id)
        obs_dict["critic"][env_id] = compute_observations(env_id)
    
    # 返回所有 4096 个环境的数据
    return obs_dict["policy"], rewards, dones, info
```

---

### 【阶段 7】清理与结束

```
22. env.close()
    位置: train.py:198
    功能: 关闭环境，清理资源
    输入: 无
    输出: 无

23. simulation_app.close()
    位置: train.py:206
    功能: 关闭 Isaac Sim 模拟器
    输入: 无
    输出: 无
```

---

## 传感器数据的完整生命周期

### height_scanner RayCaster

```
【创建阶段】env.reset()
  ├─ 初始化 RayCaster 对象
  ├─ 设置射线网格 (128 条射线)
  └─ 分配 data 缓冲区

【更新阶段】env.step() - 每 20ms
  ├─ 获取传感器位置
  ├─ 发射射线
  └─ 存储到 data.ray_hits_w [4096, 128, 3]

【使用阶段】ObservationManager.compute()
  ├─ 读取 data.pos_w 和 data.ray_hits_w
  ├─ 计算相对高度: height = sensor_z - ray_z - 0.5
  └─ 返回 [4096, 128] 张量 → 送入网络

【使用阶段】RewardManager.compute()
  └─ base_height_l2() 读取最高射线点判断基座高度
```

### contact_forces ContactSensor

```
【创建阶段】env.reset()
  ├─ 初始化 ContactSensor 对象
  ├─ 追踪机器人所有 body
  └─ 分配 data 缓冲区

【更新阶段】env.step() - 每 5ms
  ├─ 从物理引擎读取接触力
  ├─ 更新 data.net_forces_w_history [4096, 3, num_bodies, 3]
  ├─ 更新 data.current_contact_time [4096, num_bodies]
  ├─ 更新 data.current_air_time [4096, num_bodies]
  └─ 更新 data.last_air_time [4096, num_bodies]

【使用阶段】RewardManager.compute() - 多个地方
  ├─ feet_slide(): 用 net_forces_w_history
  ├─ feet_air_time(): 用 current_air_time
  ├─ feet_gait(): 用 current_contact_time
  └─ undesired_contacts(): 用 net_forces_w_history

【使用阶段】TerminationManager.compute()
  └─ illegal_contact(): 用 net_forces_w_history 判断基座接触
```

---

## 总结：函数调用频率

| 函数 | 调用频率 | 调用位置 | 目的 |
|------|---------|---------|------|
| main() | 1 次 | 脚本启动 | 训练主函数 |
| gym.make() | 1 次 | main() | 创建环境 |
| env.step() | 24 × 4096 × 10000 = 983 亿次 | runner.learn() | 执行物理模拟 |
| height_scanner.update() | ~25 亿次 | env.step() | 传感器更新（每 20ms） |
| contact_forces.update() | ~98 亿次 | env.step() | 传感器更新（每 5ms） |
| mdp.height_scan() | 24 × 4096 × 10000 = 983 亿次 | ObservationManager | 观测计算 |
| mdp.feet_slide() | 983 亿次 | RewardManager | 奖励计算 |
| mdp.feet_air_time() | 983 亿次 | RewardManager | 奖励计算 |
| actor.forward() | 24 × 4096 × 10000 + 5 × 4 × ... | runner.learn() | 策略推理 |
| critic.forward() | 类似 actor | runner.learn() | 价值推理 |
| loss.backward() | 5 epoch × 4 mini_batch × 10000 = 200 万次 | runner.learn() | 梯度计算 |

