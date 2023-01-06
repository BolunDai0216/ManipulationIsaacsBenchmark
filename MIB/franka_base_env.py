import math
from pdb import set_trace

import numpy as np
from isaacgym import gymapi
from isaacgym.torch_utils import to_torch


class FrankaBaseEnv:
    def __init__(self, cfg):
        self.gym = gymapi.acquire_gym()
        self.cfg = cfg

        # set simulation parameters
        self._set_sim_params()

        # create simulation
        self.sim = self.gym.create_sim(
            self.cfg.device.compute_device_id,
            self.cfg.device.render_device_id,
            gymapi.SIM_PHYSX,
            self.sim_params,
        )
        if self.sim is None:
            raise Exception("Failed to create sim")

        # create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise Exception("Failed to create viewer")

        # load franka asset to simulation
        self._load_franka_asset()

        # set franka joint configurations
        self._set_franka_joint_configs()

        # set simulation environment parameters
        self._set_env_params()

        # create lists for saving envs
        self.envs = []
        self.hand_idxs = []
        self.init_pos_list = []
        self.init_rot_list = []

        # load ground plane
        self._load_ground_plane()

        # create the environments
        self._create_envs()

        # set camera position
        self._set_camera_position()

        self.gym.prepare_sim(self.sim)

    def reset(self):
        pass

    def step(self, action):
        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action))

        # update viewer
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)

    def _set_sim_params(self):
        self.sim_params = gymapi.SimParams()
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        self.sim_params.dt = 1.0 / 60.0
        self.sim_params.substeps = 2
        self.sim_params.use_gpu_pipeline = True
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 8
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.rest_offset = 0.0
        self.sim_params.physx.contact_offset = 0.001
        self.sim_params.physx.friction_offset_threshold = 0.001
        self.sim_params.physx.friction_correlation_distance = 0.0005
        self.sim_params.physx.num_threads = 0
        self.sim_params.physx.use_gpu = True

    def _load_franka_asset(self):
        self.frana_asset_options = gymapi.AssetOptions()
        self.frana_asset_options.armature = 0.01
        self.frana_asset_options.fix_base_link = True
        self.frana_asset_options.disable_gravity = True
        self.frana_asset_options.flip_visual_attachments = True
        self.franka_asset = self.gym.load_asset(
            self.sim,
            self.cfg.assets.root_folder,
            self.cfg.assets.franka_urdf,
            self.frana_asset_options,
        )

    def _set_franka_joint_configs(self):
        self.franka_dof_props = self.gym.get_asset_dof_properties(self.franka_asset)
        self.franka_num_dofs = self.gym.get_asset_dof_count(self.franka_asset)
        self.franka_num_bodies = self.gym.get_asset_rigid_body_count(self.franka_asset)

        # set joint to position control with Kp and Kd gains
        for i in range(self.franka_num_dofs):
            self.franka_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            self.franka_dof_props["stiffness"][i] = self.cfg.franka.joint_stiffness[i]
            self.franka_dof_props["damping"][i] = self.cfg.franka.joint_damping[i]

        # set the default joint positions
        self.default_dof_pos = np.zeros(self.franka_num_dofs, dtype=np.float32)
        self.default_dof_pos[:7] = (
            0.3 * (self.franka_dof_props["lower"] + self.franka_dof_props["upper"])[:7]
        )
        self.default_dof_pos[7:] = self.franka_dof_props["upper"][7:]
        self.default_dof_pos_tensor = to_torch(
            self.default_dof_pos, device=self.cfg.device.compute_device_name
        )

        # set default joint state
        self.default_dof_state = np.zeros(self.franka_num_dofs, gymapi.DofState.dtype)
        self.default_dof_state["pos"] = self.default_dof_pos

    def _set_env_params(self):
        self.num_envs = self.cfg.env.num_envs
        self.env_num_per_row = int(math.sqrt(self.num_envs))

        _spacing = self.cfg.env.spacing
        self.env_lower = gymapi.Vec3(-_spacing, -_spacing, 0.0)
        self.env_upper = gymapi.Vec3(_spacing, _spacing, _spacing)

        self.franka_pose = gymapi.Transform()
        self.franka_pose.p = gymapi.Vec3(0, 0, 0)

    def _load_ground_plane(self):
        self.plane_params = gymapi.PlaneParams()
        self.plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, self.plane_params)

    def _create_envs(self):
        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(
                self.sim, self.env_lower, self.env_upper, self.env_num_per_row
            )
            self.envs.append(env)

            # add franka
            franka_handle = self.gym.create_actor(
                env, self.franka_asset, self.franka_pose, f"franka_{i}", i, 2
            )

            # set dof properties
            self.gym.set_actor_dof_properties(env, franka_handle, self.franka_dof_props)

            # set initial dof states
            self.gym.set_actor_dof_states(
                env, franka_handle, self.default_dof_state, gymapi.STATE_ALL
            )

            # set initial position targets
            self.gym.set_actor_dof_position_targets(
                env, franka_handle, self.default_dof_pos
            )

    def _set_camera_position(self):
        """Points camera to the middle of the simulation environment"""
        cam_pos = gymapi.Vec3(4, 3, 2)
        cam_target = gymapi.Vec3(-4, -3, 0)
        middle_env = self.envs[self.num_envs // 2 + self.env_num_per_row // 2]
        self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)
