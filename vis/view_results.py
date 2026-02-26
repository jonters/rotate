#!/usr/bin/env python
"""View training results locally.

Usage:
    python view_results.py <run_dir>
    python view_results.py results/hanabi/ippo/default_label/2026-02-19_00-23-59
    
    python vis/view_results.py results/hanabi/ippo/default_label/2026-02-19_00-23-59

    python vis/view_results.py results/hanabi/ippo/default_label/2026-02-18_15-04-43


Or from the run directory:
    python view_results.py .
"""
import sys
import os
import argparse
from datetime import datetime


def get_training_time(run_dir):
    """Try to estimate training time from directory name timestamp and checkpoint modification time."""
    checkpoint_path = os.path.join(run_dir, "saved_train_run")
    
    # Parse start time from directory name (format: YYYY-MM-DD_HH-MM-SS)
    dir_name = os.path.basename(run_dir)
    try:
        start_dt = datetime.strptime(dir_name, "%Y-%m-%d_%H-%M-%S")
        start_time = start_dt.timestamp()
        
        # Get end time from checkpoint modification time
        end_time = os.path.getmtime(checkpoint_path)
        
        # Also check for _METADATA file which is written at the end
        metadata_path = os.path.join(checkpoint_path, "_METADATA")
        if os.path.exists(metadata_path):
            end_time = max(end_time, os.path.getmtime(metadata_path))
        
        duration_seconds = end_time - start_time
        if duration_seconds > 0:
            return duration_seconds
    except (ValueError, OSError):
        pass
    
    return None


def main():
    parser = argparse.ArgumentParser(description="View training results from a run directory")
    parser.add_argument("run_dir", nargs="?", default=".", 
                        help="Path to the run directory (default: current directory)")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating plot")
    args = parser.parse_args()
    
    run_dir = os.path.abspath(args.run_dir)
    checkpoint_path = os.path.join(run_dir, "saved_train_run")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: No saved_train_run found in {run_dir}")
        print("Make sure you're pointing to a valid run directory.")
        sys.exit(1)
    
    # Add project root to path (go up from results/env/algo/label/timestamp)
    # Try to find the project root by looking for common markers
    project_root = run_dir
    for _ in range(10):  # Max 10 levels up
        if os.path.exists(os.path.join(project_root, "common", "save_load_utils.py")):
            break
        parent = os.path.dirname(project_root)
        if parent == project_root:
            break
        project_root = parent
    
    sys.path.insert(0, project_root)
    
    from common.save_load_utils import load_train_run
    import numpy as np

    print(f"Loading results from: {checkpoint_path}")
    out = load_train_run(checkpoint_path)

    metrics = out["metrics"]
    checkpoints = out.get("checkpoints", None)

    print("\n" + "=" * 50)
    print("TRAINING RESULTS")
    print("=" * 50)
    print(f"Run directory: {run_dir}")

    # Training time
    training_time = get_training_time(run_dir)
    if training_time is not None:
        minutes = training_time / 60
        if minutes > 60:
            print(f"Training time: {minutes/60:.1f} hours ({training_time:.0f}s)")
        else:
            print(f"Training time: {minutes:.1f} minutes ({training_time:.0f}s)")

    print(f"\nAvailable metrics: {list(metrics.keys())}")

    # Episode returns
    returns = metrics.get("returned_episode_returns", None)
    if returns is not None:
        print(f"\n--- Episode Returns ---")
        print(f"Shape: {returns.shape}")
        # Shape is typically (num_seeds, num_updates, rollout_len, num_actors)
        mean_per_update = np.nanmean(returns, axis=(0, 2, 3))
        print(f"Num updates: {len(mean_per_update)}")
        print(f"Start:  {mean_per_update[0]:.2f}")
        print(f"End:    {mean_per_update[-1]:.2f}")
        print(f"Max:    {np.max(mean_per_update):.2f} (at update {np.argmax(mean_per_update)})")

    # Episode lengths
    lengths = metrics.get("returned_episode_lengths", None)
    if lengths is not None:
        print(f"\n--- Episode Lengths ---")
        mean_lengths = np.nanmean(lengths, axis=(0, 2, 3))
        print(f"Start:  {mean_lengths[0]:.2f}")
        print(f"End:    {mean_lengths[-1]:.2f}")

    # Checkpoints info
    if checkpoints is not None:
        print(f"\n--- Checkpoints ---")
        def get_first_leaf(tree):
            if isinstance(tree, dict):
                return get_first_leaf(next(iter(tree.values())))
            return tree
        sample_leaf = get_first_leaf(checkpoints)
        if hasattr(sample_leaf, 'shape'):
            print(f"Number of checkpoints: {sample_leaf.shape[0]}")

    print("\n" + "=" * 50)

    # Optional: Plot training curves
    if not args.no_plot and returns is not None:
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            # Create figure with subplots
            num_plots = 1
            if lengths is not None:
                num_plots = 2
            
            fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
            if num_plots == 1:
                axes = [axes]
            
            # Plot 1: Episode Returns
            mean_returns = np.nanmean(returns, axis=(0, 2, 3))
            axes[0].plot(mean_returns, label='Mean Return')
            axes[0].set_xlabel("Update Step")
            axes[0].set_ylabel("Episode Return")
            axes[0].set_title("Episode Returns (sum of rewards per episode)")
            axes[0].grid(True)
            
            # Plot 2: Episode Lengths (if available)
            if lengths is not None and num_plots > 1:
                mean_lengths = np.nanmean(lengths, axis=(0, 2, 3))
                axes[1].plot(mean_lengths, color='orange', label='Mean Length')
                axes[1].set_xlabel("Update Step")
                axes[1].set_ylabel("Episode Length")
                axes[1].set_title("Episode Lengths")
                axes[1].grid(True)
            
            plt.tight_layout()
            plot_path = os.path.join(run_dir, "training_curves.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"\nTraining curves saved to: {plot_path}")
            plt.close()
            
            # Also save individual return plot for backward compatibility
            plt.figure(figsize=(10, 6))
            plt.plot(mean_returns)
            plt.xlabel("Update Step")
            plt.ylabel("Mean Episode Return")
            plt.title("Training Curve")
            plt.grid(True)
            plot_path = os.path.join(run_dir, "training_curve.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            print("\nNote: Install matplotlib to generate training curve plots")
            print("  pip install matplotlib")

if __name__ == "__main__":
    main()
