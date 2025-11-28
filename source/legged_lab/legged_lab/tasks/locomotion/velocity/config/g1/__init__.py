# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# 环境注册机制说明
##
# 什么是 gym.register？
# =====================
# gym.register() 是用来"注册"环境的函数。把一个"环境 ID 字符串"关联到对应的"环境实现代码"，
# 这样后续可以通过 gym.make("环境ID") 来快速创建环境，不需要直接导入和实例化。
#
# 类比现实：
#   - 学校里有学生信息管理系统（Gym 的注册表）
#   - 每个学生有"学号"（id）和"对应的学生记录"（entry_point + kwargs）
#   - 老师想找一个学生时，只需说"学号是多少"，系统自动返回对应的学生信息
#   - 这里 gym.make("LeggedLab-Isaac-Velocity-Rough-G1-v0") 就像"查询学号"
#
# 本文件注册的环境们：
# ====================

# 1. 训练用（Rough地形，带随机化）
gym.register(
    id="LeggedLab-Isaac-Velocity-Rough-G1-v0",  # 环境的"身份证号码"
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # 指向实现该环境的 Python 类
    disable_env_checker=True,  # 禁用 Gym 的环保检查器（本项目有自己的检查）
    kwargs={  # 传给这个类的参数（配置信息）
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughEnvCfg",
        # ↑ 告诉 ManagerBasedRLEnv 使用哪个配置类。这里是 rough_env_cfg.py 中的 G1RoughEnvCfg
        #   作用：定义机器人模型、传感器（height_scanner、contact_forces）、奖励函数、终止条件等
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        # ↑ 告诉训练脚本使用哪个 agent（强化学习算法）的配置。这里是 PPO 算法的配置
    },
)

# 2. 评估用（Rough地形，与训练相同配置，但 num_envs 更少）
# "Play" 是指"评估已训练的模型"的意思。使用相同的环境设置，但参数更轻量
gym.register(
    id="LeggedLab-Isaac-Velocity-Rough-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughEnvCfg_PLAY",  # Play 版本的配置
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)

# 3. 训练用（Flat地形，无随机化，简单环境）
gym.register(
    id="LeggedLab-Isaac-Velocity-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg",  # 平坦地形配置
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

# 4. 评估用（Flat地形）
gym.register(
    id="LeggedLab-Isaac-Velocity-Flat-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

# 5. 训练用（Rough地形，使用特殊的高度扫描传感器 Scandots）
# Scandots：是一种高度扫描传感器的特殊配置（通常用于输出 2D 高度图而非 1D 扫描）
gym.register(
    id="LeggedLab-Isaac-Velocity-Scandots-Rough-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.scandots_rough_env_cfg:G1ScandotsRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ScandotsRoughPPORunnerCfg",
    },
)

# 6. 评估用（Rough地形，Scandots）
gym.register(
    id="LeggedLab-Isaac-Velocity-Scandots-Rough-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.scandots_rough_env_cfg:G1ScandotsRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ScandotsRoughPPORunnerCfg",
    },
)

# 7. 训练用（Rough地形，使用深度相机 Depth）
gym.register(
    id="LeggedLab-Isaac-Velocity-Rough-G1-Depth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_depth_env_cfg:G1RoughDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)

# 8. 评估用（Rough地形，使用深度相机 Depth）
gym.register(
    id="LeggedLab-Isaac-Velocity-Rough-G1-Depth-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_depth_env_cfg:G1RoughDepthEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)