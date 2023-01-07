import math
import random
import time
from pdb import set_trace

import numpy as np
from isaacgym.torch_utils import *
from isaacgym import gymapi, gymtorch, gymutil
import torch


class FrankaEnv:
    def __init__(self):
        # acquire gym interface
        self.gym = gymapi.acquire_gym()

        # parse arguments
        self.args = gymutil.parse_arguments()

        # set device
        self.device = self.args.sim_device if self.args.use_gpu_pipeline else "cpu"

        # set simulation parameters
        self._set_sim_params()

        # create sim
        self.sim = self.gym.create_sim(
            self.args.compute_device_id,
            self.args.graphics_device_id,
            self.args.physics_engine,
            self.sim_params,
        )
        if self.sim is None:
            raise Exception("Failed to create sim")

        # load ground plane, set environment parameters, load assets
        self._load_ground_plane()
        self._set_env_params()
        self._load_asset()

        # set franka and simulation specific parameters
        self._set_franka_joint_configs()
        self._set_asset_default_pos()

        # create environments
        self._create_envs()

        # create viewer object
        self.viewer = None

        # use the tensor API which can be ran on CPU or GPU
        self.gym.prepare_sim(self.sim)
        self._pre_allocate_resource()

    def _set_sim_params(self):
        self.sim_params = gymapi.SimParams()
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        self.sim_params.dt = 1.0 / 60.0
        self.sim_params.substeps = 2
        self.sim_params.use_gpu_pipeline = self.args.use_gpu_pipeline

        # physics engine parameter
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 8
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.rest_offset = 0.0
        self.sim_params.physx.contact_offset = 0.001
        self.sim_params.physx.friction_offset_threshold = 0.001
        self.sim_params.physx.friction_correlation_distance = 0.0005
        self.sim_params.physx.num_threads = self.args.num_threads
        self.sim_params.physx.use_gpu = self.args.use_gpu

    def _load_asset(self):
        # create table asset
        self.table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        self.table_asset = self.gym.create_box(
            self.sim,
            self.table_dims.x,
            self.table_dims.y,
            self.table_dims.z,
            asset_options,
        )

        # create box asset
        self.box_size = 0.045
        asset_options = gymapi.AssetOptions()
        self.box_asset = self.gym.create_box(
            self.sim, self.box_size, self.box_size, self.box_size, asset_options
        )

        # create Franka arm asset
        self.asset_root = (
            "/home/bolun/Documents/IsaacGym_Preview_4_Package/isaacgym/assets"
        )
        franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        self.franka_asset = self.gym.load_asset(
            self.sim, self.asset_root, franka_asset_file, asset_options
        )

    def _load_ground_plane(self):
        """
        Load the ground plane and set its orientation.
        """
        self.plane_params = gymapi.PlaneParams()
        self.plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, self.plane_params)

    def _set_franka_joint_configs(self):
        # configure franka dofs
        self.franka_dof_props = self.gym.get_asset_dof_properties(self.franka_asset)
        franka_lower_limits = self.franka_dof_props["lower"]
        franka_upper_limits = self.franka_dof_props["upper"]
        franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)

        # set joints to position control
        self.franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
        self.franka_dof_props["stiffness"][:7].fill(400.0)
        self.franka_dof_props["damping"][:7].fill(40.0)

        # set grippers to position control
        self.franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        self.franka_dof_props["stiffness"][7:].fill(800.0)
        self.franka_dof_props["damping"][7:].fill(40.0)

        # default dof states and position targets
        self.franka_num_dofs = self.gym.get_asset_dof_count(self.franka_asset)
        self.default_dof_pos = np.zeros(self.franka_num_dofs, dtype=np.float32)
        self.default_dof_pos[:7] = franka_mids[:7]

        # grippers open
        self.default_dof_pos[7:] = franka_upper_limits[7:]

        self.default_dof_state = np.zeros(self.franka_num_dofs, gymapi.DofState.dtype)
        self.default_dof_state["pos"] = self.default_dof_pos

        # send to torch
        self.default_dof_pos_tensor = to_torch(self.default_dof_pos, device=self.device)

        # get link index of panda hand, which we will use as end effector
        franka_link_dict = self.gym.get_asset_rigid_body_dict(self.franka_asset)
        self.franka_hand_index = franka_link_dict["panda_hand"]

    def _set_asset_default_pos(self):
        self.franka_pose = gymapi.Transform()
        self.franka_pose.p = gymapi.Vec3(0, 0, 0)

        self.table_pose = gymapi.Transform()
        self.table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * self.table_dims.z)

        self.box_pose = gymapi.Transform()
        self.box_idxs = []

        self.hand_idxs = []
        self.init_pos_list = []
        self.init_rot_list = []

    def _create_envs(self):
        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(
                self.sim, self.env_lower, self.env_upper, self.num_per_row
            )
            self.envs.append(env)

            # add table
            _table_handle = self.gym.create_actor(
                env, self.table_asset, self.table_pose, "table", i, 0
            )

            # add box
            self.box_pose.p.x = self.table_pose.p.x + np.random.uniform(-0.2, 0.1)
            self.box_pose.p.y = self.table_pose.p.y + np.random.uniform(-0.3, 0.3)
            self.box_pose.p.z = self.table_dims.z + 0.5 * self.box_size
            self.box_pose.r = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi)
            )
            box_handle = self.gym.create_actor(
                env, self.box_asset, self.box_pose, "box", i, 0
            )
            color = gymapi.Vec3(
                np.random.uniform(0, 1),
                np.random.uniform(0, 1),
                np.random.uniform(0, 1),
            )
            self.gym.set_rigid_body_color(
                env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color
            )

            # get global index of box in rigid body state tensor
            box_idx = self.gym.get_actor_rigid_body_index(
                env, box_handle, 0, gymapi.DOMAIN_SIM
            )
            self.box_idxs.append(box_idx)

            # add franka
            franka_handle = self.gym.create_actor(
                env, self.franka_asset, self.franka_pose, "franka", i, 2
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

            # get inital hand pose
            hand_handle = self.gym.find_actor_rigid_body_handle(
                env, franka_handle, "panda_hand"
            )
            hand_pose = self.gym.get_rigid_transform(env, hand_handle)
            self.init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
            self.init_rot_list.append(
                [hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w]
            )

            # get global index of hand in rigid body state tensor
            hand_idx = self.gym.find_actor_rigid_body_index(
                env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM
            )
            self.hand_idxs.append(hand_idx)

    def _set_env_params(self):
        env_spacing = 1.0

        # configure env grid
        self.num_envs = 256
        self.num_per_row = int(math.sqrt(self.num_envs))
        self.env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        self.env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        self.envs = []

    def _pre_allocate_resource(self):
        # initial hand position and orientation tensors
        self.init_pos = (
            torch.Tensor(self.init_pos_list).view(self.num_envs, 3).to(self.device)
        )
        self.init_rot = (
            torch.Tensor(self.init_rot_list).view(self.num_envs, 4).to(self.device)
        )

        # hand orientation for grasping
        self.down_q = (
            torch.stack(self.num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])])
            .to(self.device)
            .view((self.num_envs, 4))
        )

        # box corner coords, used to determine grasping yaw
        box_half_size = 0.5 * self.box_size
        corner_coord = torch.Tensor([box_half_size, box_half_size, box_half_size])
        self.corners = torch.stack(self.num_envs * [corner_coord]).to(self.device)

        # downard axis
        self.down_dir = torch.Tensor([0, 0, -1]).to(self.device).view(1, 3)

        # get jacobian tensor
        # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries corresponding to franka hand
        self.j_eef = jacobian[:, self.franka_hand_index - 1, :, :7]

        # get rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        # get dof state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)
        self.dof_pos = dof_states[:, 0].view(self.num_envs, 9, 1)

        # Create a tensor noting whether the hand should return to the initial position
        self.hand_restart = torch.full([self.num_envs], False, dtype=torch.bool).to(
            self.device
        )

        # Set action tensors
        self.pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)

    def reset(self, reset_ids=0, render=True):
        self.render_or_not = render
        if self.render_or_not:
            self.render()

    def step(self, action):
        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        # Deploy actions
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(action)
        )

        state = {
            "rigid_body": self.rb_states,
            "Jacobian": self.j_eef,
            "dof_state": self.dof_pos,
        }

        if self.render_or_not:
            self.render()
            done = self.gym.query_viewer_has_closed(self.viewer)
        else:
            done = False

        return state, done

    def render(self):
        if self.viewer is None:
            # create viewer
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise Exception("Failed to create viewer")

            # point camera at middle env
            cam_pos = gymapi.Vec3(4, 3, 2)
            cam_target = gymapi.Vec3(-4, -3, 0)
            middle_env = self.envs[self.num_envs // 2 + self.num_per_row // 2]
            self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

        # update the viewer
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        self.gym.sync_frame_time(self.sim)

    def close(self):
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)

        self.gym.destroy_sim(self.sim)


def main():
    env = FrankaEnv()
    env.reset(render=True)

    for i in range(1000000):
        state, done = env.step(env.pos_action)

        if done:
            break

    env.close()


if __name__ == "__main__":
    main()
