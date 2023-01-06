import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi
from math import sqrt


class BallEnv:
    def __init__(self):
        # initialize gym
        self.gym = gymapi.acquire_gym()

        # parse arguments
        self.args = gymutil.parse_arguments()

        # set simulation parameters
        self._set_sim_params()

        # create simulation
        self.sim = self.gym.create_sim(
            self.args.compute_device_id,
            self.args.graphics_device_id,
            self.args.physics_engine,
            self.sim_params,
        )
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        # load ground plane
        self._load_ground_plane()

        # set random seed
        np.random.seed(17)

        # create viewer object
        self.viewer = None

    def _set_sim_params(self):
        """
        Set simulation environment parameters.
        """
        self.sim_params = gymapi.SimParams()
        self.sim_params.substeps = 1
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 4
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.num_threads = 0
        self.sim_params.physx.use_gpu = True
        self.sim_params.use_gpu_pipeline = False

    def _load_ground_plane(self):
        """
        Load the ground plane and set its orientation.
        """
        self.plane_params = gymapi.PlaneParams()
        self.plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, self.plane_params)

    def _set_env_params(self):
        """
        Set simulation environment parameters.
        """
        # set up the env grid
        self.num_envs = 36
        self.num_per_row = int(sqrt(self.num_envs))
        env_spacing = 1.25
        self.env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        self.env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        self.envs = []

    def _load_asset(self):
        # load ball asset
        self.asset_root = (
            "/home/bolun/Documents/IsaacGym_Preview_4_Package/isaacgym/assets"
        )
        self.asset_file = "urdf/ball.urdf"
        self.asset = self.gym.load_asset(
            self.sim, self.asset_root, self.asset_file, gymapi.AssetOptions()
        )

    def render(self):
        if self.viewer is None:
            # create viewer
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                print("*** Failed to create viewer")
                quit()

            self.gym.viewer_camera_look_at(
                self.viewer, None, gymapi.Vec3(20, 5, 20), gymapi.Vec3(0, 1, 0)
            )

        # update the viewer
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

    def close(self):
        self.gym.destroy_viewer(self.viewer)
