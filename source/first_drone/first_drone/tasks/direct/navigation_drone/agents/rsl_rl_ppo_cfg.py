from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class NavigationPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 50
    experiment_name = "navigation_drone_direct"
    logger = "wandb"
    wandb_project = "navigation_drone"

    # Actor-Critic network (Standard MLP matching SAC)
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # Slight entropy for exploration
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,  # Matching SAC learning rate
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
