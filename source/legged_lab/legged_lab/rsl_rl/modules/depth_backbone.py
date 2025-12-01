"""
深度图像编码器模块
基于extreme-parkour的实现，适配双足机器人G1
移除了偏航角(yaw)输出，因为双足机器人不需要
"""

import torch
import torch.nn as nn


class DepthOnlyFCBackbone(nn.Module):
    """
    深度图像卷积编码器
    输入: 深度图像 (batch_size, height, width)
    输出: 压缩的深度特征向量 (batch_size, scandots_output_dim)
    
    适用于64x64的深度图像输入
    """
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        
        # 深度图像压缩网络
        # 输入: [batch_size, 1, 64, 64]
        self.image_compression = nn.Sequential(
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # -> [batch_size, 32, 60, 60]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # -> [batch_size, 32, 30, 30]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            # -> [batch_size, 64, 28, 28]
            activation,
            nn.Flatten(),
            # -> [batch_size, 64 * 28 * 28] = [batch_size, 50176]
            nn.Linear(64 * 28 * 28, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
            # -> [batch_size, scandots_output_dim]
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        """
        前向传播
        Args:
            images: 深度图像 (batch_size, height, width)
        Returns:
            latent: 压缩的深度特征 (batch_size, scandots_output_dim)
        """
        # 添加通道维度: (batch_size, height, width) -> (batch_size, 1, height, width)
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)
        return latent


class RecurrentDepthBackbone(nn.Module):
    """
    循环深度编码器
    将深度特征与本体感知信息融合，并通过GRU处理时序信息
    
    与extreme-parkour的区别：
    1. 移除了偏航角(yaw)输出，双足机器人不需要
    2. 输出维度调整为纯深度特征
    """
    def __init__(self, base_backbone, env_cfg, output_dim=32) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        
        self.base_backbone = base_backbone
        self.output_dim = output_dim
        
        # 获取本体感知维度
        if env_cfg is None:
            n_proprio = 48  # G1双足机器人的默认本体感知维度
        else:
            # 从环境配置中获取
            if hasattr(env_cfg, 'env') and hasattr(env_cfg.env, 'n_proprio'):
                n_proprio = env_cfg.env.n_proprio
            elif hasattr(env_cfg, 'n_proprio'):
                n_proprio = env_cfg.n_proprio
            else:
                n_proprio = 48
        
        # 融合MLP: 将深度特征(32维) + 本体感知信息 -> 128 -> 32维
        self.combination_mlp = nn.Sequential(
            nn.Linear(32 + n_proprio, 128),
            activation,
            nn.Linear(128, 32)
        )
        
        # GRU循环网络处理时序信息
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        
        # 输出MLP: 512 -> output_dim (默认32维，不包含偏航角)
        self.output_mlp = nn.Sequential(
            nn.Linear(512, output_dim),
            last_activation
        )
        
        self.hidden_states = None

    def forward(self, depth_image, proprioception):
        """
        前向传播
        Args:
            depth_image: 深度图像 (batch_size, height, width)
            proprioception: 本体感知信息 (batch_size, n_proprio)
        Returns:
            depth_latent: 融合后的深度特征 (batch_size, output_dim)
        """
        # 1. 通过基础编码器处理深度图像
        depth_image = self.base_backbone(depth_image)
        
        # 2. 拼接深度特征和本体感知信息并进行融合
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        
        # 3. 通过GRU处理时序信息
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        
        # 4. 生成最终输出（纯深度特征，不包含偏航角）
        depth_latent = self.output_mlp(depth_latent.squeeze(1))
        
        return depth_latent

    def detach_hidden_states(self):
        """分离隐藏状态，避免梯度传播"""
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.detach().clone()

    def reset_hidden_states(self, batch_size=None, device=None):
        """重置隐藏状态"""
        if batch_size is not None and device is not None:
            self.hidden_states = torch.zeros(1, batch_size, 512, device=device)
        else:
            self.hidden_states = None


class StackDepthEncoder(nn.Module):
    """
    堆叠深度编码器
    处理多帧深度图像序列
    
    注：双足机器人可能不需要这个，但保留以备将来使用
    """
    def __init__(self, base_backbone, env_cfg, buffer_len=10) -> None:
        super().__init__()
        activation = nn.ELU()
        self.base_backbone = base_backbone
        self.buffer_len = buffer_len
        
        # 获取本体感知维度
        if hasattr(env_cfg, 'env') and hasattr(env_cfg.env, 'n_proprio'):
            n_proprio = env_cfg.env.n_proprio
        elif hasattr(env_cfg, 'n_proprio'):
            n_proprio = env_cfg.n_proprio
        else:
            n_proprio = 48
        
        # 融合MLP
        self.combination_mlp = nn.Sequential(
            nn.Linear(32 + n_proprio, 128),
            activation,
            nn.Linear(128, 32)
        )

        # 1D卷积处理时序信息
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=buffer_len, out_channels=16, kernel_size=4, stride=2),
            activation,
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2),
            activation
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(16 * 14, 32), 
            activation
        )
        
    def forward(self, depth_image, proprioception):
        """
        前向传播
        Args:
            depth_image: 深度图像序列 (batch_size, buffer_len, height, width)
            proprioception: 本体感知信息 (batch_size, n_proprio)
        Returns:
            depth_latent: 融合后的深度特征 (batch_size, 32)
        """
        # 处理每一帧深度图像
        batch_size, num_frames = depth_image.shape[0], depth_image.shape[1]
        depth_latent = self.base_backbone(depth_image.flatten(0, 1))
        depth_latent = depth_latent.reshape(batch_size, num_frames, -1)
        
        # 通过1D卷积处理时序信息
        depth_latent = self.conv1d(depth_latent)
        depth_latent = self.mlp(depth_latent.flatten(1, 2))
        
        return depth_latent
