from isaaclab.utils import configclass


@configclass
class SACAgentCfg:
    """Configuration for the SAC+VAE agent.

    Analogous to RslRlOnPolicyRunnerCfg — contains only hyperparameters.
    The SACRunner class reads these values to build models and run training.
    """

    # ---------- VAE ----------
    vae_lr: float = 1e-4

    # ---------- SAC ----------
    sac_actor_lr: float = 3e-4
    sac_critic_lr: float = 3e-4
    sac_alpha_lr: float = 3e-4
    sac_gamma: float = 0.99
    sac_tau: float = 0.005
    sac_batch_size: int = 256
    sac_replay_size: int = 200_000
    sac_warmup_steps: int = 1_000
    vae_training_steps: int = 0   # env steps to train VAE only before SAC starts
    sac_update_every: int = 4        # update SAC every N env steps (was 64 — much too slow)
    sac_gradient_steps: int = 4      # gradient updates per update phase
    sac_success_ratio: float = 0.25  # fraction of batch from success buffer
    sac_max_iterations: int = 500_000  # total env steps
    train_vae: bool = False  # whether the VAE should be updated during training

    # ---------- Logging ----------
    wandb_project: str = "first_drone"
    experiment_name: str = "sac_drone"
    log_interval: int = 100  # log every N env steps
    save_interval: int = 2000  # save model every N env steps
