import os
from pxr import Usd, UsdGeom, Gf

usd_path = r"D:\isaac\3D_Drone_RL\source\first_drone\first_drone\tasks\direct\first_drone\assets\fps_shooter_game_arena_map_v4.usdz"
stage = Usd.Stage.Open(usd_path)

bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])

q = Gf.Quaternion(0.7071, Gf.Vec3d(0.7071, 0.0, 0.0))
rot_m = Gf.Matrix4d(Gf.Rotation(q), Gf.Vec3d(0, 0, 0))
scale_m = Gf.Matrix4d().SetScale(Gf.Vec3d(0.01, 0.01, 0.01))
trans_m = Gf.Matrix4d().SetTranslate(Gf.Vec3d(25.0, 25.0, -0.9937)) # translated down
m = scale_m * rot_m * trans_m

obstacles_2d = []

for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Mesh):
        name = prim.GetName()
        bbox = bbox_cache.ComputeLocalBound(prim)
        range_val = bbox.GetRange()
        min_pt = range_val.GetMin()
        max_pt = range_val.GetMax()
        
        # Calculate world transform of this prim
        prim_geom = UsdGeom.Imageable(prim)
        local_to_world = prim_geom.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        combined = local_to_world * m
        
        # Corners of the 3D bounding box
        corners = []
        for x in [min_pt[0], max_pt[0]]:
            for y in [min_pt[1], max_pt[1]]:
                for z in [min_pt[2], max_pt[2]]:
                    corners.append(Gf.Vec3d(x, y, z))
                    
        transformed_corners = [combined.Transform(c) for c in corners]
        x_coords = [c[0] for c in transformed_corners]
        y_coords = [c[1] for c in transformed_corners]
        z_coords = [c[2] for c in transformed_corners]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        min_z, max_z = min(z_coords), max(z_coords)
        
        # Filter out the floor mesh (which has max_z < 0.1)
        if max_z > 0.5:
            obstacles_2d.append({
                "name": name,
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y
            })

print(f"Found {len(obstacles_2d)} obstacles:")
for obs in obstacles_2d:
    print(f"    # {obs['name']}")
    print(f"    [{obs['min_x']:.3f}, {obs['max_x']:.3f}, {obs['min_y']:.3f}, {obs['max_y']:.3f}],")
