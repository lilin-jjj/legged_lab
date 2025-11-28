# Depth-camera variant of the G1 rough environment.

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.camera import TiledCameraCfg
from isaaclab.sim.spawners.sensors import PinholeCameraCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.velocity.mdp as mdp
from legged_lab.tasks.locomotion.velocity.velocity_env_cfg import (
    ObservationsCfg,
    MySceneCfg,
)
from .rough_env_cfg import G1RoughEnvCfg


@configclass
class DepthSceneCfg(MySceneCfg):
    """Scene configuration that adds a depth camera mounted on the robot base."""

    depth_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        # use default offset / intrinsics, can be tuned later
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        data_types=["distance_to_image_plane"],
        width=64,  # 128
        height=64,  # 72
        debug_vis=False,
    )


@configclass
class DepthObservationsCfg(ObservationsCfg):
    """Extend default observations with a flattened depth image term."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        depth_image = ObsTerm(
            func=mdp.depth_image_flat,
            params={"sensor_cfg": SceneEntityCfg("depth_camera")},
            clip=(0.0, 20.0),
        )

        def __post_init__(self):
            super().__post_init__()
            # disable inherited height_scan term; we only want depth_image as terrain observation
            self.height_scan = None

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        depth_image = ObsTerm(
            func=mdp.depth_image_flat,
            params={"sensor_cfg": SceneEntityCfg("depth_camera")},
            clip=(0.0, 20.0),
        )

        def __post_init__(self):
            super().__post_init__()
            # disable inherited height_scan term for critic as well
            self.height_scan = None

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1RoughDepthEnvCfg(G1RoughEnvCfg):
    """G1 rough terrain environment with additional depth camera observations."""

    def __post_init__(self):
        super().__post_init__()
        # attach depth camera configuration onto existing scene so that
        # robot and terrain settings from the parent G1 config are preserved.
        depth_scene = DepthSceneCfg(num_envs=self.scene.num_envs, env_spacing=self.scene.env_spacing)
        self.scene.depth_camera = depth_scene.depth_camera
        # replace observations with the depth-augmented variant
        self.observations = DepthObservationsCfg()


@configclass
class G1RoughDepthEnvCfg_PLAY(G1RoughDepthEnvCfg):
    """Play configuration for G1 rough terrain environment with depth camera."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
