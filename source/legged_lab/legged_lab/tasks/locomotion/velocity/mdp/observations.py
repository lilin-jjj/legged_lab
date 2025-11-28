from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def height_scan(env: "ManagerBasedEnv", sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    scan = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset

    scan = torch.nan_to_num(scan, nan=0.0, posinf=10.0, neginf=-10.0)
    scan = torch.clamp(scan, -10.0, 10.0)

    return scan


def height_scan_ch(env: "ManagerBasedEnv", sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    
    add a channel dimension to the output tensor, so that it can be used as a 2D image
    
    ref: isaaclab.envs.mdp.observations.height_scan
    """
    # extract the used quantities (to enable type-hinting)
    # RayCaster/Array 提供的数据字段说明：
    #  - sensor.data.pos_w: 传感器在世界坐标系下的位置，shape=(num_envs, 3)
    #  - sensor.data.ray_hits_w: 射线命中点的世界坐标，shape=(num_envs, num_rays, 3)
    #  - 对于 Array 形式，ray_hits_w 按 pattern_cfg 中指定的顺序排列。
    # 本函数将每个射线的击中点 z 分量转换为相对高度：sensor_height - hit_z - offset，
    # 并根据 ordering/shape 重构为 [num_envs, H, W, 1] 的格式供网络使用（H, W 对应格点尺寸）。
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    ordering = sensor.cfg.pattern_cfg.ordering
    """Specifies the ordering of points in the generated grid. Defaults to ``"xy"``.

    Consider a grid pattern with points at :math:`(x, y)` where :math:`x` and :math:`y` are the grid indices.
    The ordering of the points can be specified as "xy" or "yx". This determines the inner and outer loop order
    when iterating over the grid points.

    * If "xy" is selected, the points are ordered with inner loop over "x" and outer loop over "y".
    * If "yx" is selected, the points are ordered with inner loop over "y" and outer loop over "x".

    For example, the grid pattern points with :math:`X = (0, 1, 2)` and :math:`Y = (3, 4)`:

    * "xy" ordering: :math:`[(0, 3), (1, 3), (2, 3), (1, 4), (2, 4), (2, 4)]`
    * "yx" ordering: :math:`[(0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4)]`
    """
    
    shape = sensor.cfg.shape    # define in RayCasterArrayCfg

    # height scan: height = sensor_height - hit_point_z - offset
    # 注意：如果某些射线未命中（例如超出范围），ray_hits_w 的值可能是填充值，需要在上层检查或在网络中适当裁剪。
    scan = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset

    # 数值安全处理：去除 NaN / Inf，防止后续 reward 或网络输入出现 inf 导致训练崩溃
    scan = torch.nan_to_num(scan, nan=0.0, posinf=10.0, neginf=-10.0)
    scan = torch.clamp(scan, -10.0, 10.0)
    
    # TODO: check
    if ordering == "yx":
        scan = scan.reshape( -1 ,shape[0], shape[1])
    elif ordering == "xy":
        scan = scan.reshape(-1, shape[1], shape[0]).transpose(1, 2)
    else:
        raise ValueError(f"Invalid ordering: {ordering}. Expected 'xy' or 'yx'.")
    
    # 最终返回形状为 [num_envs, H, W, 1] 的张量（H, W 由 RayCasterArrayCfg.shape 决定），
    # 值为相对高度（米），并且已经按照 ordering 进行了排列。
    return scan.unsqueeze(-1)  # add a channel dimension to the output tensor


def depth_image_flat(env: "ManagerBasedEnv", sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Flatten depth image from a depth camera sensor into a vector per environment.

    This helper matches the TienKung-Lab style usage where the depth image
    (num_envs, H, W, 1) is reshaped to (num_envs, H * W) and concatenated
    to the actor / critic observation vectors.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    depth_image = sensor.data.output["distance_to_image_plane"]
    # (num_envs, H, W, C) -> (num_envs, H * W * C)
    return depth_image.view(depth_image.shape[0], -1)
