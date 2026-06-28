from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

@configclass
class NavigationPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "navigation_drone_direct"
    logger = "wandb"
    wandb_project = "Multilevel_Train"


    # Actor network
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 64, 32],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
        ),
    )

    # Critic network
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 64, 32],
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
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


