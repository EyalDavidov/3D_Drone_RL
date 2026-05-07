import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

DRONE_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1),
        rot=(0.7071, 0.0, 0.0, -0.7071),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            "m1_joint": 200.0,
            "m2_joint": -200.0,
            "m3_joint": 200.0,
            "m4_joint": -200.0,
        },
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Configuration for the Crazyflie quadcopter (cf2x).

Key physics properties:
  - enable_gyroscopic_forces: Simulates gyroscopic precession from spinning rotors
  - solver iterations: Controls physics constraint solving accuracy
  - sleep/stabilization thresholds: Prevent the drone from sleeping when nearly stationary
  - damping=0.0: Actuators are "dummy" — we apply forces via wrench, not joint targets
  - init joint_vel: Initial rotor spin (cosmetic only, thrust comes from wrench)
"""