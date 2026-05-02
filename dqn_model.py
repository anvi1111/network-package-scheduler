"""
main.py — Entry Point for RL Packet Scheduling Project
========================================================

Orchestrates the full pipeline:
    1. Train a DQN agent on PacketSchedulerEnv
    2. Evaluate RL agent vs baseline schedulers
    3. Generate comparison plots and training curves

Usage:
    python main.py                          # Run full pipeline
    python main.py --train-only             # Train only
    python main.py --eval-only              # Evaluate only (requires trained model)
    python main.py --demo                   # Quick demo without training
    python main.py --timesteps 200000       # Custom training length
    python main.py --episodes 20            # More eval episodes
    python main.py --seed 123              # Set random seed
"""

import argparse
import sys
import os
import logging

import numpy as np

from env import PacketSchedulerEnv
from baselines import RoundRobinScheduler
from utils import (
    DEFAULT_CONFIG,
    setup_logging,
    ensure_dirs,
    save_config,
    get_timestamp,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RL-based Network Packet Scheduling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        Full pipeline (train + evaluate + plot)
  python main.py --train-only           Train DQN agent only
  python main.py --eval-only            Evaluate existing model only
  python main.py --demo                 Quick demo with Round Robin
  python main.py --timesteps 200000     Train for 200k timesteps
  python main.py --arrival-rates 0.3 0.5 0.7 0.9 1.0
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--train-only", action="store_true",
        help="Only train the DQN agent (skip evaluation)",
    )
    mode_group.add_argument(
        "--eval-only", action="store_true",
        help="Only evaluate (requires a trained model)",
    )
    mode_group.add_argument(
        "--demo", action="store_true",
        help="Run a quick demo without training",
    )

    # Training parameters
    parser.add_argument(
        "--timesteps", type=int, default=100_000,
        help="Total training timesteps (default: 100000)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.95,
        help="Discount factor (default: 0.95)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Training batch size (default: 64)",
    )
    parser.add_argument(
        "--buffer-size", type=int, default=50_000,
        help="Replay buffer size (default: 50000)",
    )

    # Environment parameters
    parser.add_argument(
        "--arrival-rate", type=float, default=0.6,
        help="Training arrival rate (default: 0.6)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=500,
        help="Max steps per episode (default: 500)",
    )
    parser.add_argument(
        "--max-queue-len", type=int, default=20,
        help="Max queue capacity (default: 20)",
    )
    parser.add_argument(
        "--trace-file", type=str, default=None,
        help="Path to CSV file with packet arrival trace data (default: None)",
    )

    # Evaluation parameters
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of evaluation episodes per config (default: 10)",
    )
    parser.add_argument(
        "--arrival-rates", type=float, nargs="+",
        default=[0.3, 0.45, 0.6, 0.75, 0.9],
        help="Arrival rates to test (default: 0.3 0.45 0.6 0.75 0.9)",
    )

    # General
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--model-path", type=str, default="models/dqn_scheduler",
        help="Path for saving/loading the DQN model",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Directory for results and logs",
    )
    parser.add_argument(
        "--figures-dir", type=str, default="figures",
        help="Directory for saved figures",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    """Build configuration dict from parsed arguments."""
    config = DEFAULT_CONFIG.copy()
    config.update({
        "total_timesteps": args.timesteps,
        "learning_rate": args.lr,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "arrival_rate": args.arrival_rate,
        "max_steps": args.max_steps,
        "max_queue_len": args.max_queue_len,
        "trace_file": args.trace_file,
        "n_eval_episodes": args.episodes,
        "arrival_rates": args.arrival_rates,
        "model_save_path": args.model_path,
        "results_dir": args.results_dir,
        "figures_dir": args.figures_dir,
        "log_file": os.path.join(args.results_dir, "experiment.log"),
    })
    return config


def quick_demo(config: dict, n_steps: int = 200) -> None:
    """Run a quick demo with Round Robin scheduler — no training needed."""
    print("\n" + "=" * 55)
    print("  Quick Demo: Round Robin on PacketSchedulerEnv")
    if config.get("trace_file"):
        print(f"  Using custom trace file: {config['trace_file']}")
    print("=" * 55)

    env = PacketSchedulerEnv(max_steps=n_steps, trace_file=config.get("trace_file"))
    rr = RoundRobinScheduler(env)
    obs, _ = env.reset()

    for i in range(n_steps):
        action = rr.act(obs)
        obs, reward, terminated, _, info = env.step(action)
        if i % 50 == 0:
            env.render()
        if terminated:
            break

    print(f"\n  Final avg latency : {info['avg_latency']:.2f}")
    print("=" * 55)


def main() -> None:
    """Main entry point — orchestrates train/eval/plot pipeline."""
    args = parse_args()
    config = build_config(args)

    # Setup
    logger = setup_logging(config["log_file"])
    ensure_dirs(config)

    print("\n" + "=" * 60)
    print("  RL-based Network Packet Scheduling")
    print("  " + "-" * 40)
    print(f"  Seed: {args.seed} | Timesteps: {args.timesteps:,}")
    print(f"  Eval episodes: {args.episodes} | Rates: {args.arrival_rates}")
    print("=" * 60)

    # Save config for reproducibility
    config_path = os.path.join(config["results_dir"], f"config_{get_timestamp()}.json")
    save_config(config, config_path)
    logger.info(f"Config saved to {config_path}")

    # ── Mode: Demo ──
    if args.demo:
        quick_demo(config)
        return

    # ── Mode: Train ──
    training_metrics = None
    model = None

    if not args.eval_only:
        from train import train_dqn

        logger.info("Starting DQN training...")
        model, metrics_callback = train_dqn(config=config, seed=args.seed)
        training_metrics = metrics_callback.get_metrics()

        logger.info(
            f"Training complete. Episodes: {len(training_metrics['rewards'])}, "
            f"Final avg reward (last 50): "
            f"{np.mean(training_metrics['rewards'][-50:]):.2f}"
        )

        # Plot training curves
        from visualize import plot_training_curves
        plot_training_curves(training_metrics, save_dir=config["figures_dir"])

        if args.train_only:
            print("\n  Training complete! Run with --eval-only to evaluate.")
            return

    # ── Mode: Evaluate ──
    from evaluate import run_comparison
    from visualize import plot_all_comparisons

    logger.info("Starting evaluation...")
    all_results = run_comparison(
        config=config,
        model=model,
        model_path=config["model_save_path"] if model is None else None,
        seed=args.seed,
    )

    # Generate comparison plots
    plot_all_comparisons(
        all_results,
        config["arrival_rates"],
        figures_dir=config["figures_dir"],
    )

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("  " + "-" * 40)
    print(f"  Results:  {config['results_dir']}/")
    print(f"  Figures:  {config['figures_dir']}/")
    print(f"  Model:    {config['model_save_path']}.zip")
    print(f"  Log:      {config['log_file']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
