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

mesh_bounds = []

for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Mesh):
        path = prim.GetPath()
        bbox = bbox_cache.ComputeLocalBound(prim)
        range_val = bbox.GetRange()
        min_pt = range_val.GetMin()
        max_pt = range_val.GetMax()
        
        # Transform bounds to world frame
        # Compute local to world matrix for this prim
        prim_geom = UsdGeom.Imageable(prim)
        local_to_world = prim_geom.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        
        # Combine with our custom Room transformation m
        combined = local_to_world * m
        
        corners = []
        for x in [min_pt[0], max_pt[0]]:
            for y in [min_pt[1], max_pt[1]]:
                for z in [min_pt[2], max_pt[2]]:
                    corners.append(Gf.Vec3d(x, y, z))
                    
        transformed_corners = [combined.Transform(c) for c in corners]
        z_coords = [c[2] for c in transformed_corners]
        
        mesh_bounds.append({
            "path": path,
            "min_z": min(z_coords),
            "max_z": max(z_coords),
            "name": prim.GetName()
        })

# Sort meshes by their min_z
mesh_bounds.sort(key=lambda x: x["min_z"])

print("Top 30 meshes sorted by Min Z:")
for item in mesh_bounds[:30]:
    print(f"Mesh: {item['name']} | Path: {item['path']} | Min Z = {item['min_z']:.4f} | Max Z = {item['max_z']:.4f}")
