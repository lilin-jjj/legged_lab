#!/usr/bin/env python3
"""
G1深度相机推理脚本
使用训练好的深度编码器进行推理
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="Play G1 with depth encoder")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--depth_checkpoint", type=str, default=None, help="Path to depth encoder checkpoint.")
parser.add_argument("--use_depth_encoder", action="store_true", default=False, help="Use depth encoder for inference.")

# 添加AppLauncher参数
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启用相机
args_cli.enable_cameras = True

# 启动Omniverse应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks
import legged_lab.tasks
from legged_lab.rsl_rl.depth_training_wrapper import DepthTrainingWrapper
from legged_lab.rsl_rl import DepthEncoderCfg


def main():
    """使用深度编码器进行推理"""
    
    # 创建环境
    env = gym.make(args_cli.task, num_envs=args_cli.num_envs, render_mode="rgb_array")
    
    # 转换为单智能体实例
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # 包装环境
    env = RslRlVecEnvWrapper(env)
    
    # 加载策略
    if args_cli.checkpoint is None:
        raise ValueError("Please provide a checkpoint path using --checkpoint")
    
    print(f"[INFO] Loading checkpoint from: {args_cli.checkpoint}")
    checkpoint = torch.load(args_cli.checkpoint)
    
    # 创建策略网络
    # 这里需要从checkpoint中恢复网络结构
    # 简化处理：假设使用标准的actor-critic
    from rsl_rl.modules import ActorCritic
    
    # 从checkpoint获取网络配置
    # 实际应用中需要保存和加载完整的配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建actor-critic（需要根据实际配置调整）
    num_obs = env.num_obs
    num_actions = env.num_actions
    
    actor_critic = ActorCritic(
        num_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation='elu',
    ).to(device)
    
    actor_critic.load_state_dict(checkpoint['model_state_dict'])
    actor_critic.eval()
    
    # 创建深度编码器（如果需要）
    depth_wrapper = None
    if args_cli.use_depth_encoder:
        print("[INFO] Creating depth encoder for inference...")
        
        # 创建深度编码器配置
        depth_cfg = DepthEncoderCfg(
            use_depth_encoder=True,
            scandots_output_dim=32,
            hidden_state_dim=512,
            output_dim=32,
            learning_rate=1.0e-4,
            use_recurrent=True,
            num_frames=1,
        )
        
        depth_wrapper = DepthTrainingWrapper(
            env=env.unwrapped,
            actor_critic=actor_critic,
            depth_encoder_cfg=depth_cfg,
            device=device,
        )
        
        # 加载深度编码器权重
        if args_cli.depth_checkpoint is not None:
            print(f"[INFO] Loading depth encoder from: {args_cli.depth_checkpoint}")
            depth_state = torch.load(args_cli.depth_checkpoint)
            depth_wrapper.load_state_dict(depth_state)
        elif 'depth_encoder_state_dict' in checkpoint:
            print("[INFO] Loading depth encoder from main checkpoint")
            depth_wrapper.load_state_dict(checkpoint)
        else:
            print("[WARN] No depth encoder weights found, using random initialization")
        
        depth_wrapper.depth_encoder.eval()
        depth_wrapper.depth_actor.eval()
    
    # 运行推理
    print("[INFO] Starting inference...")
    obs = env.get_observations()
    
    num_episodes = 0
    episode_rewards = []
    current_reward = torch.zeros(env.num_envs, device=device)
    
    while num_episodes < 10:  # 运行10个episode
        with torch.no_grad():
            if depth_wrapper is not None and args_cli.use_depth_encoder:
                # 使用深度编码器
                try:
                    depth_image = env.unwrapped.scene.sensors["depth_camera"].data.output["distance_to_image_plane"]
                    depth_image = depth_image.view(depth_image.shape[0], -1)
                    
                    # 处理深度观测
                    depth_latent = depth_wrapper.process_depth_observation(obs, depth_image)
                    
                    # 使用深度actor生成动作
                    actions = depth_wrapper.depth_actor.forward(obs)
                except Exception as e:
                    print(f"[WARN] Error using depth encoder: {e}")
                    # 回退到标准actor
                    actions = actor_critic.act_inference(obs)
            else:
                # 使用标准actor
                actions = actor_critic.act_inference(obs)
        
        # 执行动作
        obs, rewards, dones, infos = env.step(actions)
        current_reward += rewards
        
        # 检查episode结束
        done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_indices) > 0:
            for idx in done_indices:
                episode_rewards.append(current_reward[idx].item())
                current_reward[idx] = 0
                num_episodes += 1
                print(f"Episode {num_episodes} reward: {episode_rewards[-1]:.2f}")
    
    # 打印统计信息
    print("\n[INFO] Inference completed")
    print(f"Average reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
    print(f"Max reward: {max(episode_rewards):.2f}")
    print(f"Min reward: {min(episode_rewards):.2f}")
    
    # 关闭环境
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
