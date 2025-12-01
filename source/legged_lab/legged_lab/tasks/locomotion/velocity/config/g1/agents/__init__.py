# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import rsl_rl_ppo_cfg, rsl_rl_depth_ppo_cfg

##
# Register Gym environments.
##

# G1 Depth Camera with PPO
gym.register(
    id="LeggedLab-Isaac-Velocity-Rough-G1-Depth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "legged_lab.tasks.locomotion.velocity.config.g1.rough_depth_env_cfg:G1RoughDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{rsl_rl_depth_ppo_cfg.__name__}.G1DepthPPORunnerCfg",
    },
)
