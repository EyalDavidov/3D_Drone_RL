from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

@configclass
class NavigationPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2500
    save_interval = 50
    experiment_name = "navigation_drone_direct"
    logger = "wandb"
    wandb_project = "Multilevel_Train"


    # Actor network
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.8,
            std_type="log",
        ),
    )

    # Critic network
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
    )

    resume = False
    load_run = ""
    load_checkpoint = ""

    # Map observation dict keys to actor/critic
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0005,  # Increased from 0.0001 to prevent rapid exploration decay
        num_learning_epochs=5,
        num_mini_batches=8,   # Restored from 16 to 8 to match the successful stable Red run updates
        learning_rate=1.5e-4,  # Halved from 3.0e-4 to prevent policy collapse and sudden regressions
        schedule="fixed",
        gamma=0.998,
        lam=0.95,
        max_grad_norm=1.0,    # Strictly enforces gradient clipping safeguard to prevent exploding standard deviations
    )


