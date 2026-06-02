import os
from pxr import Usd, UsdGeom, Gf

usd_path = r"D:\isaac\3D_Drone_RL\source\first_drone\first_drone\tasks\direct\first_drone\assets\fps_shooter_game_arena_map_v4.usdz"
stage = Usd.Stage.Open(usd_path)

bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])

q = Gf.Quaternion(0.7071, Gf.Vec3d(0.7071, 0.0, 0.0))
rot_m = Gf.Matrix4d(Gf.Rotation(q), Gf.Vec3d(0, 0, 0))
scale_m = Gf.Matrix4d().SetScale(Gf.Vec3d(0.01, 0.01, 0.01))
trans_m = Gf.Matrix4d().SetTranslate(Gf.Vec3d(25.0, 25.0, 0.0))
m = scale_m * rot_m * trans_m

for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Mesh):
        name = prim.GetName().lower()
        if "floor" in name or "ground" in name or "base" in name or "plane" in name:
            bbox = bbox_cache.ComputeLocalBound(prim)
            range_val = bbox.GetRange()
            min_pt = range_val.GetMin()
            max_pt = range_val.GetMax()
            
            prim_geom = UsdGeom.Imageable(prim)
            local_to_world = prim_geom.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            combined = local_to_world * m
            
            corners = [
                combined.Transform(Gf.Vec3d(min_pt[0], min_pt[1], min_pt[2])),
                combined.Transform(Gf.Vec3d(max_pt[0], max_pt[1], max_pt[2]))
            ]
            z_coords = [c[2] for c in corners]
            print(f"Match: {prim.GetName()} | Min Z = {min(z_coords):.4f} | Max Z = {max(z_coords):.4f}")
