from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlOnPolicyRunnerCfg
from .amp_cfg import RslRlAmpCfg

#########################
# Policy configurations #
#########################

@configclass
class RslRlPpoActorCriticConv2dCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with convolutional layers."""

    class_name: str = "ActorCriticConv2d"
    """The policy class name. Default is ActorCriticConv2d."""

    conv_layers_params: list[dict] = [
        {"out_channels": 4, "kernel_size": 3, "stride": 2},
        {"out_channels": 8, "kernel_size": 3, "stride": 2},
        {"out_channels": 16, "kernel_size": 3, "stride": 2},
    ]
    """List of convolutional layer parameters for the convolutional network."""

    conv_linear_output_size: int = 16
    """Output size of the linear layer after the convolutional features are flattened."""


@configclass
class DepthEncoderCfg:
    """深度编码器配置"""
    
    use_depth_encoder: bool = True
    """是否使用深度编码器"""
    
    scandots_output_dim: int = 32
    """深度特征输出维度"""
    
    hidden_state_dim: int = 512
    """GRU隐藏状态维度"""
    
    output_dim: int = 32
    """最终输出维度（不包含偏航角）"""
    
    learning_rate: float = 1.0e-4
    """深度编码器学习率"""
    
    use_recurrent: bool = True
    """是否使用循环网络（GRU）"""
    
    num_frames: int = 1
    """输入帧数"""

############################
# Algorithm configurations #
############################


@configclass
class RslRlPpoAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the AMP algorithm."""
    
    class_name: str = "PPOAmp"
    """The algorithm class name. Default is PPOAmp."""

    amp_cfg: RslRlAmpCfg = RslRlAmpCfg()
    """Configuration for the AMP (Adversarial Motion Priors) in the training."""


#########################
# Runner configurations #
#########################

    