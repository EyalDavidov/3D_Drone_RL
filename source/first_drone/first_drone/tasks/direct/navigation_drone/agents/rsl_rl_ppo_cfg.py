from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

@configclass
class NavigationPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 50
    experiment_name = "navigation_drone_direct"
    logger = "wandb"
    wandb_project = "first_drone"

    # Actor network
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="log",
        ),
    )

    # Critic network
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
    )

    # Map observation dict keys to actor/critic
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0001,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=5.0e-5,
        schedule="fixed",
        gamma=0.998,
        lam=0.95,
        max_grad_norm=1.0,
    )
