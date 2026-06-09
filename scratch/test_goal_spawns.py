import torch

def is_inside_map_obstacle(x, y, map_obstacles, margin=0.5):
    inside = torch.zeros_like(x, dtype=torch.bool)
    for obs in map_obstacles:
        min_x, max_x, min_y, max_y = obs
        inside = inside | (
            (x >= min_x - margin) & (x <= max_x + margin)
            & (y >= min_y - margin) & (y <= max_y + margin)
        )
    return inside

def test_level(level, min_d, max_d, map_obstacles):
    env_count = 50000
    spawn_x = torch.zeros(env_count).uniform_(-20.0, 20.0)
    spawn_y = torch.zeros(env_count).uniform_(-20.0, 20.0)

    # Resample drone spawns
    for _ in range(10):
        in_obstacle = is_inside_map_obstacle(spawn_x, spawn_y, map_obstacles, margin=0.5)
        if not torch.any(in_obstacle):
            break
        n = torch.sum(in_obstacle).item()
        spawn_x[in_obstacle] = torch.zeros(n).uniform_(-20.0, 20.0)
        spawn_y[in_obstacle] = torch.zeros(n).uniform_(-20.0, 20.0)

    angle = torch.zeros(env_count).uniform_(0.0, 2 * 3.1415926535)
    dist = torch.zeros(env_count).uniform_(min_d, max_d)
    
    goal_x_local = spawn_x + dist * torch.cos(angle)
    goal_y_local = spawn_y + dist * torch.sin(angle)

    # Resample goals up to 15 iterations (original)
    bad_count_15 = 0
    for i in range(15):
        in_obstacle = is_inside_map_obstacle(goal_x_local, goal_y_local, map_obstacles, margin=0.15)
        out_of_bounds = (goal_x_local.abs() > 24.0) | (goal_y_local.abs() > 24.0)
        bad = in_obstacle | out_of_bounds
        if not torch.any(bad):
            break
        n = torch.sum(bad).item()
        angle_resample = torch.zeros(n).uniform_(0.0, 2 * 3.1415926535)
        dist_resample = torch.zeros(n).uniform_(min_d, max_d)
        goal_x_local[bad] = spawn_x[bad] + dist_resample * torch.cos(angle_resample)
        goal_y_local[bad] = spawn_y[bad] + dist_resample * torch.sin(angle_resample)

    in_obstacle = is_inside_map_obstacle(goal_x_local, goal_y_local, map_obstacles, margin=0.15)
    out_of_bounds = (goal_x_local.abs() > 24.0) | (goal_y_local.abs() > 24.0)
    bad_15 = in_obstacle | out_of_bounds
    pct_bad_15 = torch.sum(bad_15).item() / env_count * 100.0

    # Resample goals up to 100 iterations
    goal_x_local_100 = spawn_x + dist * torch.cos(angle)
    goal_y_local_100 = spawn_y + dist * torch.sin(angle)
    for i in range(100):
        in_obstacle = is_inside_map_obstacle(goal_x_local_100, goal_y_local_100, map_obstacles, margin=0.15)
        out_of_bounds = (goal_x_local_100.abs() > 24.0) | (goal_y_local_100.abs() > 24.0)
        bad = in_obstacle | out_of_bounds
        if not torch.any(bad):
            break
        n = torch.sum(bad).item()
        angle_resample = torch.zeros(n).uniform_(0.0, 2 * 3.1415926535)
        dist_resample = torch.zeros(n).uniform_(min_d, max_d)
        goal_x_local_100[bad] = spawn_x[bad] + dist_resample * torch.cos(angle_resample)
        goal_y_local_100[bad] = spawn_y[bad] + dist_resample * torch.sin(angle_resample)

    in_obstacle_100 = is_inside_map_obstacle(goal_x_local_100, goal_y_local_100, map_obstacles, margin=0.15)
    out_of_bounds_100 = (goal_x_local_100.abs() > 24.0) | (goal_y_local_100.abs() > 24.0)
    bad_100 = in_obstacle_100 | out_of_bounds_100
    pct_bad_100 = torch.sum(bad_100).item() / env_count * 100.0

    # Resample both drone spawn and goal spawn if failed
    # We do a joint resampling loop
    joint_bad = torch.ones(env_count, dtype=torch.bool)
    joint_spawn_x = spawn_x.clone()
    joint_spawn_y = spawn_y.clone()
    joint_goal_x = goal_x_local.clone()
    joint_goal_y = goal_y_local.clone()

    for i in range(50):
        if not torch.any(joint_bad):
            break
        n = torch.sum(joint_bad).item()
        # Resample spawns for bad envs
        joint_spawn_x[joint_bad] = torch.zeros(n).uniform_(-20.0, 20.0)
        joint_spawn_y[joint_bad] = torch.zeros(n).uniform_(-20.0, 20.0)
        
        # Make sure spawns are not in obstacles
        for _ in range(5):
            in_obs = is_inside_map_obstacle(joint_spawn_x, joint_spawn_y, map_obstacles, margin=0.5)
            if not torch.any(in_obs & joint_bad):
                break
            n_obs = torch.sum(in_obs & joint_bad).item()
            joint_spawn_x[in_obs & joint_bad] = torch.zeros(n_obs).uniform_(-20.0, 20.0)
            joint_spawn_y[in_obs & joint_bad] = torch.zeros(n_obs).uniform_(-20.0, 20.0)

        # Generate new goals
        angle_res = torch.zeros(n).uniform_(0.0, 2 * 3.1415926535)
        dist_res = torch.zeros(n).uniform_(min_d, max_d)
        joint_goal_x[joint_bad] = joint_spawn_x[joint_bad] + dist_res * torch.cos(angle_res)
        joint_goal_y[joint_bad] = joint_spawn_y[joint_bad] + dist_res * torch.sin(angle_res)

        # Check validity
        in_obstacle_j = is_inside_map_obstacle(joint_goal_x, joint_goal_y, map_obstacles, margin=0.15)
        out_of_bounds_j = (joint_goal_x.abs() > 24.0) | (joint_goal_y.abs() > 24.0)
        joint_bad = in_obstacle_j | out_of_bounds_j

    pct_bad_joint = torch.sum(joint_bad).item() / env_count * 100.0

    print(f"Level {level} ({min_d}m - {max_d}m):")
    print(f"  Invalid with 15 iterations: {pct_bad_15:.2f}%")
    print(f"  Invalid with 100 iterations: {pct_bad_100:.2f}%")
    print(f"  Invalid with joint spawn/goal resampling (50 iterations): {pct_bad_joint:.2f}%")

def main():
    map_obstacles = (
        (14.012, 19.012, -2.025, 4.975),
        (4.012, 9.012, 6.975, 12.975),
        (-15.988, -10.988, -21.025, -11.025),
        (0.012, 2.012, -8.025, -1.025),
        (8.012, 10.012, -17.025, -10.025),
        (-7.988, -5.988, 1.975, 8.975),
        (-18.988, -16.988, -4.025, 2.975),
        (-21.988, -16.988, 8.975, 15.975),
        (15.012, 17.012, 11.975, 18.975),
        (-11.318, -2.756, 13.145, 20.975),
        (-1.988, 3.012, -22.025, -16.025),
        (17.012, 19.012, -21.025, -14.025),
        (6.012, 7.012, 18.975, 19.975),
        (-18.988, -17.988, 20.975, 21.975),
        (10.012, 11.012, 1.975, 2.975),
        (18.012, 19.012, -8.025, -7.025),
        (-7.988, -6.988, -6.025, -5.025),
        (-20.988, -19.988, -14.025, -13.025),
    )
    
    levels = {
        1: (2.0, 5.0),
        2: (5.0, 10.0),
        3: (10.0, 18.0),
        4: (18.0, 28.0),
        5: (28.0, 40.0)
    }
    
    for level, (min_d, max_d) in levels.items():
        test_level(level, min_d, max_d, map_obstacles)

if __name__ == '__main__':
    main()
