# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 400
    save_interval = 50
    experiment_name = "camera_first_drone_direct"

    # Actor: CNN processes the 100x100x1 depth image
    actor = RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[32, 64, 64],
            kernel_size=[8, 4, 3],
            stride=[4, 2, 1],
            activation="elu",
        ),
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
        stochastic=True,
        init_noise_std=1.0,
    )

    # Critic: MLP processes the 12-dim state vector
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=False,
        stochastic=False,
        init_noise_std=1.0,
    )

    # Map observation dict keys to actor/critic
    obs_groups = {
        "actor": ["policy", "imu"],  # actor receives depth image (CNN) + IMU data (MLP)
        "critic": ["critic"],        # critic receives full privileged state vector
    }

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )