#!/usr/bin/env python3
"""
G1深度相机改进训练脚本
集成深度编码器，使用类似extreme-parkour的训练方式
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# 添加命令行参数
import cli_args

parser = argparse.ArgumentParser(description="Train G1 with depth encoder (improved)")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument("--depth_training_interval", type=int, default=10, help="Interval for depth encoder training.")
parser.add_argument("--depth_steps_per_env", type=int, default=24, help="Steps per env for depth training.")

# 添加RSL-RL参数
cli_args.add_rsl_rl_args(parser)
# 添加AppLauncher参数
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 如果录制视频，启用相机
if args_cli.video:
    args_cli.enable_cameras = True

# 清理sys.argv供Hydra使用
sys.argv = [sys.argv[0]] + hydra_args

# 启动Omniverse应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import platform
from packaging import version

# 检查RSL-RL版本
RSL_RL_VERSION = "3.1.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

import gymnasium as gym
import torch
from datetime import datetime
from collections import deque

import omni
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# 导入扩展以设置环境任务
import legged_lab.tasks
from legged_lab.rsl_rl.depth_training_wrapper import DepthTrainingWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """使用深度编码器训练G1"""
    
    # 覆盖配置
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # 设置环境种子
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 多GPU训练配置
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # 指定日志目录
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # 设置IO描述符导出标志
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        omni.log.warn("IO descriptors are only supported for manager based RL environments.")

    env_cfg.log_dir = log_dir

    # 创建Isaac环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 转换为单智能体实例
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 保存恢复路径
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # 包装用于视频录制
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 包装环境用于RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 创建RSL-RL runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # 创建深度训练包装器
    depth_wrapper = None
    if hasattr(agent_cfg, 'depth_encoder') and agent_cfg.depth_encoder.use_depth_encoder:
        print("[INFO] Creating depth encoder wrapper...")
        depth_wrapper = DepthTrainingWrapper(
            env=env.unwrapped,
            actor_critic=runner.alg.actor_critic,
            depth_encoder_cfg=agent_cfg.depth_encoder,
            device=agent_cfg.device,
        )
        print(f"[INFO] Depth encoder created with output_dim={agent_cfg.depth_encoder.output_dim}")
    
    # 写入git状态到日志
    runner.add_git_repo_to_log(__file__)
    
    # 加载检查点
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)
        if depth_wrapper is not None:
            # 尝试加载深度编码器权重
            try:
                checkpoint = torch.load(resume_path)
                depth_wrapper.load_state_dict(checkpoint)
                print("[INFO] Depth encoder weights loaded successfully.")
            except:
                print("[WARN] Could not load depth encoder weights from checkpoint.")

    # 保存配置到日志目录
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # 运行训练
    print("[INFO] Starting training...")
    train_with_depth_encoder(
        runner=runner,
        depth_wrapper=depth_wrapper,
        num_iterations=agent_cfg.max_iterations,
        depth_training_interval=args_cli.depth_training_interval,
        depth_steps_per_env=args_cli.depth_steps_per_env,
        log_dir=log_dir,
        save_interval=agent_cfg.save_interval,
    )

    # 关闭环境
    env.close()


def train_with_depth_encoder(
    runner,
    depth_wrapper,
    num_iterations,
    depth_training_interval=10,
    depth_steps_per_env=24,
    log_dir=None,
    save_interval=100,
):
    """
    使用深度编码器的训练循环
    
    Args:
        runner: RSL-RL OnPolicyRunner
        depth_wrapper: 深度训练包装器
        num_iterations: 训练迭代次数
        depth_training_interval: 深度编码器训练间隔
        depth_steps_per_env: 每个环境的深度训练步数
        log_dir: 日志目录
        save_interval: 保存间隔
    """
    
    if depth_wrapper is None:
        # 如果没有深度编码器，使用标准训练
        runner.learn(num_learning_iterations=num_iterations, init_at_random_ep_len=True)
        return
    
    # 训练循环
    for it in range(num_iterations):
        # 标准PPO训练
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=(it == 0))
        
        # 每隔一定迭代进行深度编码器训练
        if it % depth_training_interval == 0 and it > 0:
            print(f"\n[INFO] Iteration {it}: Training depth encoder...")
            train_depth_encoder_step(runner, depth_wrapper, depth_steps_per_env)
        
        # 保存检查点
        if it % save_interval == 0 and it > 0:
            save_path = os.path.join(log_dir, f'model_{it}.pt')
            runner.save(save_path)
            
            # 保存深度编码器权重
            if depth_wrapper is not None:
                depth_state = depth_wrapper.get_state_dict()
                torch.save(depth_state, os.path.join(log_dir, f'depth_encoder_{it}.pt'))
                print(f"[INFO] Saved checkpoint at iteration {it}")
    
    # 保存最终模型
    final_path = os.path.join(log_dir, f'model_{num_iterations}.pt')
    runner.save(final_path)
    if depth_wrapper is not None:
        depth_state = depth_wrapper.get_state_dict()
        torch.save(depth_state, os.path.join(log_dir, f'depth_encoder_{num_iterations}.pt'))
    print(f"[INFO] Training completed. Final model saved.")


def train_depth_encoder_step(runner, depth_wrapper, num_steps):
    """
    执行一步深度编码器训练
    
    类似extreme-parkour的learn_vision方法
    """
    env = runner.env
    device = runner.device
    
    depth_wrapper.depth_encoder.train()
    depth_wrapper.depth_actor.train()
    
    # 收集数据
    depth_latent_buffer = []
    actions_teacher_buffer = []
    actions_student_buffer = []
    
    obs = env.get_observations()
    
    for step in range(num_steps):
        with torch.no_grad():
            # 获取深度图像
            # 假设深度图像在观测的特定位置
            # 需要根据实际环境配置调整
            try:
                depth_image = env.unwrapped.scene.sensors["depth_camera"].data.output["distance_to_image_plane"]
                depth_image = depth_image.view(depth_image.shape[0], -1)  # 展平
            except:
                print("[WARN] Could not get depth image from environment")
                return
            
            # 处理深度观测，获取深度特征
            depth_latent = depth_wrapper.process_depth_observation(obs, depth_image)
            depth_latent_buffer.append(depth_latent)
            
            # 获取teacher动作（使用完整观测，包括展平的深度图像）
            actions_teacher = runner.alg.actor_critic.act_inference(obs)
            actions_teacher_buffer.append(actions_teacher)
            
            # 构建student观测：用深度特征替换展平的深度图像
            # 假设观测结构：[本体感知, 展平深度图像, ...]
            # 需要将展平的深度图像部分替换为深度特征
            obs_student = obs.clone()
            
            # 获取本体感知维度
            n_proprio = depth_wrapper.n_proprio
            # 深度图像维度 (64*64=4096)
            depth_img_dim = 64 * 64
            
            # 构建新的观测：[本体感知, 深度特征, 其他观测]
            # 注意：这里需要根据实际的观测结构调整
            if obs.shape[1] > n_proprio + depth_img_dim:
                # 有其他观测项
                obs_student = torch.cat([
                    obs[:, :n_proprio],           # 本体感知
                    depth_latent,                  # 深度特征 (32维) 替换展平深度图像
                    obs[:, n_proprio + depth_img_dim:]  # 其他观测
                ], dim=1)
            else:
                # 只有本体感知和深度图像
                obs_student = torch.cat([
                    obs[:, :n_proprio],           # 本体感知
                    depth_latent,                  # 深度特征 (32维)
                ], dim=1)
            
            # 使用深度actor生成student动作
            actions_student = depth_wrapper.depth_actor.act_inference(obs_student)
            actions_student_buffer.append(actions_student)
        
        # 执行动作
        obs, _, _, _, _ = env.step(actions_student.detach())
    
    # 更新深度编码器和深度actor
    if len(depth_latent_buffer) > 0:
        actions_teacher = torch.cat(actions_teacher_buffer, dim=0)
        actions_student = torch.cat(actions_student_buffer, dim=0)
        
        # 更新深度actor
        depth_actor_loss = depth_wrapper.update_depth_actor(actions_student, actions_teacher)
        
        # 分离隐藏状态
        depth_wrapper.detach_hidden_states()
        
        print(f"  Depth actor loss: {depth_actor_loss:.4f}")


if __name__ == "__main__":
    main()
    simulation_app.close()
