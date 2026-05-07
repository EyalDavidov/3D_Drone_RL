# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlMLPModelCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PPORunnerMLPCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 400
    save_interval = 50
    experiment_name = "flight_controller_drone_direct"

    # Logging — use Weights & Biases for live training dashboards
    logger = "wandb"
    wandb_project = "first_drone"

    # Actor: MLP processes the standard observation vector (no camera)
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        stochastic=True,
        init_noise_std=1.0,
    )

    # Critic: MLP processes the 12-dim state vector
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        stochastic=False,
        init_noise_std=1.0,
    )

    # Map observation dict keys to actor/critic
    obs_groups = {
        "actor": ["policy", "imu"],  # actor receives the policy and imu vectors concatenated
        "critic": ["critic"],        # critic receives full privileged state vector
    }

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, #משקל הניקוד של ה critic
        use_clipped_value_loss=True, # 
        clip_param=0.2, #הגבלת צעד למידה בגג 20%
        entropy_coef=0.01, #כמה אקראיות נרצה שיהיה
        num_learning_epochs=5, #כמה פעמים נעבור על כל המידע
        num_mini_batches=4, #כמה חבילות מידע נחלק כל פעם
        learning_rate=1.0e-3,
        schedule="adaptive", #שינוי קצב למידה
        gamma=0.99, #מקדם דעיכה גבוה זה אומר רחוק יותר חשוב
        lam=0.95, #כמה נרצה שהרשת תהיה בטוחה
        desired_kl=0.01, #כמה נרצה שהרשת תהיה בטוחה
        max_grad_norm=1.0,
    )
