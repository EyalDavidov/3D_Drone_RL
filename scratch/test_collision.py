from isaaclab.app import AppLauncher
app = AppLauncher({'headless': True})
sim_app = app.app

import os
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from pxr import UsdPhysics, UsdGeom

# Initialize simulation context
sim_cfg = SimulationCfg(dt=0.01)
sim = SimulationContext(sim_cfg)

# Load room mesh
usd_path = os.path.abspath("source/first_drone/first_drone/tasks/direct/first_drone/assets/fps_shooter_game_arena_map_v4.usdz")
room_cfg = sim_utils.UsdFileCfg(usd_path=usd_path, scale=(0.01, 0.01, 0.01))
room_cfg.func(
    "/World/envs/env_0/Room",
    room_cfg,
    translation=(25.0, 25.0, -0.9937),
    orientation=(0.7071, 0.7071, 0.0, 0.0),
)

# Traverse and apply collision
stage = sim.stage
mesh_count = 0
collision_applied = 0
for prim in stage.Traverse():
    if str(prim.GetPath()).startswith("/World/envs/env_0/Room") and prim.IsA(UsdGeom.Mesh):
        mesh_count += 1
        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_coll = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_coll.CreateApproximationAttr().Set("none")
        collision_applied += 1

print(f"Traversed {mesh_count} meshes, applied collision to {collision_applied} meshes.")
sim.reset()
print("Simulation reset successfully with collisions!")
sim_app.close()
