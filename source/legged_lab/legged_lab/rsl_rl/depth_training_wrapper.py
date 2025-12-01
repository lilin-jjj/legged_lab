"""
深度编码器训练包装器
用于在RSL-RL训练循环中集成深度编码器
"""

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional

from .modules.depth_backbone import DepthOnlyFCBackbone, RecurrentDepthBackbone


def create_depth_actor_from_teacher(teacher_actor, obs_dim_reduction):
    """
    从teacher actor创建depth actor，调整输入维度
    
    Args:
        teacher_actor: 主actor网络
        obs_dim_reduction: 观测维度减少量 (深度图像维度 - 深度特征维度)
    
    Returns:
        depth_actor: 适配深度特征的actor网络
    """
    import torch.nn as nn
    
    # 深拷贝teacher actor
    depth_actor = deepcopy(teacher_actor)
    
    # 找到第一个Linear层并调整输入维度
    # 假设actor的第一层是Linear层
    for name, module in depth_actor.named_modules():
        if isinstance(module, nn.Linear):
            # 找到第一个Linear层
            old_in_features = module.in_features
            new_in_features = old_in_features - obs_dim_reduction
            out_features = module.out_features
            
            # 创建新的Linear层
            new_linear = nn.Linear(new_in_features, out_features, bias=module.bias is not None)
            
            # 复制权重（截取前new_in_features维）
            with torch.no_grad():
                new_linear.weight.data = module.weight.data[:, :new_in_features].clone()
                if module.bias is not None:
                    new_linear.bias.data = module.bias.data.clone()
            
            # 替换第一个Linear层
            # 需要找到父模块并替换
            parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = dict(depth_actor.named_modules())[parent_name]
            else:
                parent = depth_actor
            
            setattr(parent, child_name, new_linear)
            
            print(f"[INFO] Adjusted first Linear layer: {old_in_features} -> {new_in_features}")
            break  # 只调整第一层
    
    return depth_actor


class DepthTrainingWrapper:
    """
    深度编码器训练包装器
    
    负责：
    1. 创建深度编码器和深度actor
    2. 在训练过程中处理深度图像
    3. 执行深度编码器的蒸馏训练
    """
    
    def __init__(
        self,
        env,
        actor_critic,
        depth_encoder_cfg,
        device,
    ):
        """
        Args:
            env: 环境实例
            actor_critic: Actor-Critic网络
            depth_encoder_cfg: 深度编码器配置
            device: 设备
        """
        self.env = env
        self.actor_critic = actor_critic
        self.cfg = depth_encoder_cfg
        self.device = device
        
        # 获取本体感知维度
        # 从环境配置中获取
        if hasattr(env.unwrapped.cfg, 'observations'):
            # 计算policy观测中的本体感知维度
            # 假设policy观测的第一部分是本体感知信息
            # 需要根据实际配置调整
            self.n_proprio = self._get_proprio_dim(env)
        else:
            self.n_proprio = 48  # G1默认值
        
        # 创建深度编码器
        if self.cfg.use_depth_encoder:
            self._create_depth_encoder()
        else:
            self.depth_encoder = None
            self.depth_actor = None
            self.depth_encoder_optimizer = None
            self.depth_actor_optimizer = None
    
    def _get_proprio_dim(self, env):
        """从环境配置中获取本体感知维度"""
        try:
            # 尝试从观测配置中获取
            obs_cfg = env.unwrapped.cfg.observations.policy
            
            # 计算本体感知维度（不包括深度图像）
            n_proprio = 0
            for term_name in dir(obs_cfg):
                if term_name.startswith('_'):
                    continue
                term = getattr(obs_cfg, term_name)
                if term is None:
                    continue
                if term_name == 'depth_image':
                    continue  # 跳过深度图像
                # 这里需要根据实际的观测项计算维度
                # 简化处理：使用固定值
            
            # G1双足机器人的本体感知维度
            # 包括：base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + 
            #      commands(4) + joint_pos(10) + joint_vel(10) + actions(10) = 43
            # 实际可能有所不同，需要根据配置调整
            return 43
        except:
            return 48  # 默认值
    
    def _create_depth_encoder(self):
        """创建深度编码器和深度actor"""
        # 创建基础深度backbone
        depth_backbone = DepthOnlyFCBackbone(
            prop_dim=self.n_proprio,
            scandots_output_dim=self.cfg.scandots_output_dim,
            hidden_state_dim=self.cfg.hidden_state_dim,
            output_activation=None,
            num_frames=self.cfg.num_frames,
        ).to(self.device)
        
        # 创建循环深度编码器（如果启用）
        if self.cfg.use_recurrent:
            # 创建一个简化的环境配置对象
            class EnvCfgWrapper:
                def __init__(self, n_proprio):
                    self.n_proprio = n_proprio
            
            env_cfg_wrapper = EnvCfgWrapper(self.n_proprio)
            
            self.depth_encoder = RecurrentDepthBackbone(
                depth_backbone,
                env_cfg_wrapper,
                output_dim=self.cfg.output_dim,
            ).to(self.device)
        else:
            self.depth_encoder = depth_backbone
        
        # 创建深度actor
        # 主actor输入：本体感知 + 展平深度图像(4096维) + 其他
        # 深度actor输入：本体感知 + 深度特征(32维) + 其他
        # 需要调整输入维度
        
        # 计算维度差异
        depth_img_dim = 64 * 64  # 4096
        depth_feature_dim = self.cfg.scandots_output_dim  # 32
        self.obs_dim_diff = depth_img_dim - depth_feature_dim  # 4064
        
        # 使用辅助函数创建depth actor，自动调整第一层的输入维度
        self.depth_actor = create_depth_actor_from_teacher(
            self.actor_critic.actor,
            self.obs_dim_diff
        ).to(self.device)
        
        print(f"[INFO] Depth actor created with adjusted input dimension (reduction: {self.obs_dim_diff})")
        
        # 创建优化器
        self.depth_encoder_optimizer = torch.optim.Adam(
            self.depth_encoder.parameters(),
            lr=self.cfg.learning_rate
        )
        
        self.depth_actor_optimizer = torch.optim.Adam(
            list(self.depth_actor.parameters()) + list(self.depth_encoder.parameters()),
            lr=self.cfg.learning_rate
        )
    
    def process_depth_observation(self, obs, depth_image):
        """
        处理深度观测
        
        Args:
            obs: 完整观测 (batch_size, obs_dim)
            depth_image: 深度图像 (batch_size, H, W) 或 (batch_size, H*W)
        
        Returns:
            depth_latent: 深度特征 (batch_size, output_dim)
        """
        if self.depth_encoder is None:
            return None
        
        # 提取本体感知信息（前n_proprio维）
        obs_proprio = obs[:, :self.n_proprio].clone()
        
        # 如果深度图像是展平的，需要reshape
        if len(depth_image.shape) == 2:
            # 假设是64x64的图像
            H = W = 64
            depth_image = depth_image.view(-1, H, W)
        
        # 通过深度编码器处理
        depth_latent = self.depth_encoder(depth_image, obs_proprio)
        
        return depth_latent
    
    def update_depth_encoder(self, depth_latent, target_latent, max_grad_norm=1.0):
        """
        更新深度编码器（蒸馏训练）
        
        Args:
            depth_latent: 深度编码器输出 (batch_size, output_dim)
            target_latent: 目标特征（从teacher获取） (batch_size, output_dim)
            max_grad_norm: 最大梯度范数
        
        Returns:
            loss: 损失值
        """
        if self.depth_encoder is None:
            return 0.0
        
        # 计算L2损失
        loss = (target_latent.detach() - depth_latent).norm(p=2, dim=1).mean()
        
        # 反向传播
        self.depth_encoder_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.depth_encoder.parameters(), max_grad_norm)
        self.depth_encoder_optimizer.step()
        
        return loss.item()
    
    def update_depth_actor(self, actions_student, actions_teacher, max_grad_norm=1.0):
        """
        更新深度actor
        
        Args:
            actions_student: 学生actor输出 (batch_size, action_dim)
            actions_teacher: 教师actor输出 (batch_size, action_dim)
            max_grad_norm: 最大梯度范数
        
        Returns:
            loss: 损失值
        """
        if self.depth_actor is None:
            return 0.0
        
        # 计算L2损失
        loss = (actions_teacher.detach() - actions_student).norm(p=2, dim=1).mean()
        
        # 反向传播
        self.depth_actor_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.depth_actor.parameters()) + list(self.depth_encoder.parameters()),
            max_grad_norm
        )
        self.depth_actor_optimizer.step()
        
        return loss.item()
    
    def detach_hidden_states(self):
        """分离深度编码器的隐藏状态"""
        if self.depth_encoder is not None and hasattr(self.depth_encoder, 'detach_hidden_states'):
            self.depth_encoder.detach_hidden_states()
    
    def reset_hidden_states(self, batch_size=None):
        """重置深度编码器的隐藏状态"""
        if self.depth_encoder is not None and hasattr(self.depth_encoder, 'reset_hidden_states'):
            self.depth_encoder.reset_hidden_states(batch_size, self.device)
    
    def get_state_dict(self):
        """获取状态字典用于保存"""
        if self.depth_encoder is None:
            return {}
        
        return {
            'depth_encoder_state_dict': self.depth_encoder.state_dict(),
            'depth_actor_state_dict': self.depth_actor.state_dict(),
            'depth_encoder_optimizer_state_dict': self.depth_encoder_optimizer.state_dict(),
            'depth_actor_optimizer_state_dict': self.depth_actor_optimizer.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        """加载状态字典"""
        if self.depth_encoder is None:
            return
        
        if 'depth_encoder_state_dict' in state_dict:
            self.depth_encoder.load_state_dict(state_dict['depth_encoder_state_dict'])
        if 'depth_actor_state_dict' in state_dict:
            self.depth_actor.load_state_dict(state_dict['depth_actor_state_dict'])
        if 'depth_encoder_optimizer_state_dict' in state_dict:
            self.depth_encoder_optimizer.load_state_dict(state_dict['depth_encoder_optimizer_state_dict'])
        if 'depth_actor_optimizer_state_dict' in state_dict:
            self.depth_actor_optimizer.load_state_dict(state_dict['depth_actor_optimizer_state_dict'])
