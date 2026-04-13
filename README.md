# First Drone

A custom reinforcement learning environment built on top of [Isaac Lab](https://github.com/isaac-sim/IsaacLab). 

This project trains a quadcopter to navigate through a custom room with obstacles (poles) and reach a target destination. It uses wrench-based physics (direct forces and torques applied to the drone body) and includes a body-mounted depth camera setup for vision-based learning.

## Features
* **Custom Environment:** `Camera-First-Drone-Direct-v0`
* **Physics:** Wrench-based control (1 thrust, 3 moments) for accurate flight dynamics.
* **Sensors:** TiledCamera configured for depth perception.
* **RL Algorithm:** Proximal Policy Optimization (PPO) using `rsl-rl`.

## Running the Project

To train the drone:

```bash
# Ensure you run this from the project root directory!
c:\Isaac\IsaacLab\isaaclab.bat -p scripts/rsl_rl/train.py --task Camera-First-Drone-Direct-v0 --num_envs 64 --enable_cameras
```

*(Note: Adjust the absolute path to `isaaclab.bat` if your installation is elsewhere).*

To play/evaluate a trained model:

```bash
c:\Isaac\IsaacLab\isaaclab.bat -p scripts/rsl_rl/play.py --task Camera-First-Drone-Direct-v0 --num_envs 4 --enable_cameras
```

## Setup & Installation

If you haven't already, install this project module into your Isaac Lab python environment:
```bash
c:\Isaac\IsaacLab\isaaclab.bat -p -m pip install -e source/first_drone
```