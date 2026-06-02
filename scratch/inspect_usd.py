import os
from pxr import Usd, UsdGeom, Gf

usd_path = r"D:\isaac\3D_Drone_RL\source\first_drone\first_drone\tasks\direct\first_drone\assets\fps_shooter_game_arena_map_v4.usdz"
stage = Usd.Stage.Open(usd_path)

bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
root_prim = stage.GetDefaultPrim()
if not root_prim:
    root_prim = stage.GetPseudoRoot()

bbox = bbox_cache.ComputeLocalBound(root_prim)
range_val = bbox.GetRange()
min_pt = range_val.GetMin()
max_pt = range_val.GetMax()

print(f"Local Bounds:")
print(f"  Min: {min_pt}")
print(f"  Max: {max_pt}")

q = Gf.Quaternion(0.7071, Gf.Vec3d(0.7071, 0.0, 0.0))
rot_m = Gf.Matrix4d(Gf.Rotation(q), Gf.Vec3d(0, 0, 0))
scale_m = Gf.Matrix4d().SetScale(Gf.Vec3d(0.01, 0.01, 0.01))
trans_m = Gf.Matrix4d().SetTranslate(Gf.Vec3d(25.0, 25.0, 0.0))

# In USD, multiplication order is: transformed_pt = pt * local_to_world_matrix
# where local_to_world_matrix = scale * rotation * translation
# Let's construct it correctly
m = scale_m * rot_m * trans_m

corners = []
for x in [min_pt[0], max_pt[0]]:
    for y in [min_pt[1], max_pt[1]]:
        for z in [min_pt[2], max_pt[2]]:
            corners.append(Gf.Vec3d(x, y, z))

transformed_corners = [m.Transform(c) for c in corners]
z_coords = [c[2] for c in transformed_corners]
print(f"Transformed Z coords: Min Z = {min(z_coords):.6f}, Max Z = {max(z_coords):.6f}")

# Let's also print all corner coordinates to see the transformation details
for i, (orig, trans) in enumerate(zip(corners, transformed_corners)):
    print(f"Corner {i}: {orig} -> {trans}")
