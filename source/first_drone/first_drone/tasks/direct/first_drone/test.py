from isaaclab.app import AppLauncher

# פותח Isaac Sim עם מצלמות
app_launcher = AppLauncher({"headless": False, "enable_cameras": True})
simulation_app = app_launcher.app

import gymnasium as gym

def main():
    env = gym.make(
        "Camera-First-Drone-Direct-v0",
        cfg=None,
        render_mode="rgb_array",
    )

    print("Env created successfully")
    print("Pause here and inspect the Stage tree for /World/envs/env_0/Drone/body/front_cam")

    # משאיר את הסימולציה פתוחה
    while simulation_app.is_running():
        simulation_app.update()

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()