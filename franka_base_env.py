from pdb import set_trace

from isaacgym import gymapi


class FrankaBaseEnv:
    def __init__(self, cfg):
        self.gym = gymapi.acquire_gym()
        self.cfg = cfg

        # set simulation parameters
        self.set_sim_params()

        # create simulation
        self.sim = self.gym.create_sim(
            self.cfg.device.compute_device_id,
            self.cfg.device.render_device_id,
            gymapi.SIM_PHYSX,
            self.sim_params,
        )

        # load franka asset to simulation
        self.load_franka_asset()

        set_trace()

    def reset(self):
        pass

    def step(self):
        pass

    def set_sim_params(self):
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

    def load_franka_asset(self):
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
