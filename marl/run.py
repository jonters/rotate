'''Main entry point for running MARL algorithms.'''
import os
import sys

# Ensure the project root (parent of marl/) is on sys.path so that local
# packages like `envs` are found even after Hydra changes the working directory.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import hydra
from omegaconf import OmegaConf

from common.wandb_visualizations import Logger
from ippo import run_ippo


class NoOpLogger:
    """Minimal logger that does nothing, used during sweeps."""
    def log_item(self, *args, **kwargs): pass
    def commit(self, *args, **kwargs): pass
    def log_artifact(self, *args, **kwargs): pass
    def close(self, *args, **kwargs): pass


def sweep_num_envs(config):
    """Sweep through NUM_ENVS values and plot wall-clock time vs mean return."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    num_envs_list = list(config.sweep_num_envs)
    time_limit_minutes = config.get("sweep_time_limit_minutes", 10)
    time_limit_seconds = time_limit_minutes * 60

    # Preserve original config values to compute steps_per_chunk ratio
    original_total = int(config.algorithm.TOTAL_TIMESTEPS)
    original_chunks = int(config.algorithm.TRAIN_CHUNKS)
    original_steps_per_chunk = original_total // original_chunks
    rollout_length = int(config.algorithm.ROLLOUT_LENGTH)

    # Use a large total so the time limit is the binding constraint
    big_total = 10_000_000_000
    big_chunks = big_total // original_steps_per_chunk

    print(f"\n{'='*60}")
    print(f"NUM_ENVS Sweep: {num_envs_list}")
    print(f"Time limit per run: {time_limit_minutes} min")
    print(f"Steps per chunk: {original_steps_per_chunk} (same ratio as original config)")
    print(f"Total chunks budget: {big_chunks}")
    print(f"{'='*60}\n")

    results = {}

    for num_envs in num_envs_list:
        # Check that NUM_UPDATES per chunk >= 1
        num_updates = original_steps_per_chunk // rollout_length // num_envs
        if num_updates < 1:
            print(f"Skipping NUM_ENVS={num_envs}: would result in 0 updates per chunk "
                  f"(steps_per_chunk={original_steps_per_chunk}, "
                  f"rollout_length={rollout_length})")
            continue

        print(f"\n{'='*60}")
        print(f"Running NUM_ENVS = {num_envs}  "
              f"(NUM_UPDATES/chunk = {num_updates})")
        print(f"{'='*60}")

        OmegaConf.set_struct(config, False)
        config.algorithm.NUM_ENVS = num_envs
        config.algorithm.TOTAL_TIMESTEPS = big_total
        config.algorithm.TRAIN_CHUNKS = big_chunks
        OmegaConf.set_struct(config, True)

        logger = NoOpLogger()

        try:
            out = run_ippo(config, logger, time_limit_seconds=time_limit_seconds)
            results[num_envs] = out.get("curve", [])
            print(f"  NUM_ENVS={num_envs}: collected {len(results[num_envs])} data points")
        except Exception as e:
            print(f"  ERROR with NUM_ENVS={num_envs}: {e}")
            results[num_envs] = []

    # ----- Plot results -----
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    _plot_sweep(results, time_limit_minutes, savedir)

    # Also save raw data as .npz for later analysis
    npz_path = os.path.join(savedir, "num_envs_sweep_data.npz")
    np.savez(npz_path, **{
        f"num_envs_{k}": np.array(v) for k, v in results.items() if v
    })
    print(f"Raw sweep data saved to {npz_path}")

    return results


def _plot_sweep(results, time_limit_minutes, savedir):
    """Generate and save the sweep plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8))

    for num_envs in sorted(results.keys()):
        curve = results[num_envs]
        if not curve:
            continue
        times, returns = zip(*curve)
        ax.plot(times, returns, label=f"NUM_ENVS={num_envs}",
                marker="o", markersize=2, linewidth=1.5)

    ax.set_xlabel("Wall Clock Time (seconds)", fontsize=13)
    ax.set_ylabel("Mean Episode Return", fontsize=13)
    ax.set_title(f"NUM_ENVS Sweep — {time_limit_minutes} min per run", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    savepath = os.path.join(savedir, "num_envs_sweep.png")
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSweep plot saved to {savepath}")


@hydra.main(version_base=None, config_path="configs", config_name="base_config_marl")
def main(config):
    print(OmegaConf.to_yaml(config, resolve=True))

    # --- Sweep mode ---
    sweep_vals = config.get("sweep_num_envs", None)
    if sweep_vals:
        sweep_num_envs(config)
        return

    # --- Normal training ---
    wandb_logger = Logger(config)

    if config.algorithm["ALG"] == "ippo":
        run_ippo(config, wandb_logger)
    else:
        raise NotImplementedError(f"Algorithm {config['ALG']} not implemented.")
        
    wandb_logger.close()

if __name__ == "__main__":
    main()