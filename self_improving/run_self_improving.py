#!/usr/bin/env python3
"""
CLI entry point for the self-improving skills agent pipeline.

Usage:
    python self_improving/run_self_improving.py --config self_improving/configs/self_improving.yaml
    python self_improving/run_self_improving.py --config self_improving/configs/self_improving.yaml --epochs 5
    python self_improving/run_self_improving.py --config self_improving/configs/self_improving.yaml --dry-run
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(verbose: bool = False, log_file: str = None):
    """Configure logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="a"))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Self-improving Skills Agent Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Full run with default config
  python self_improving/run_self_improving.py --config self_improving/configs/self_improving.yaml

  # Custom epochs and output
  python self_improving/run_self_improving.py --config ... --epochs 5 --output-dir ./outputs/si_v2/

  # Dry run (setup only, no execution)
  python self_improving/run_self_improving.py --config ... --dry-run

  # Resume from epoch 2
  python self_improving/run_self_improving.py --config ... --resume --start-epoch 2
""",
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="Epoch to start from (for resuming)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Setup only, don't run epochs")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Log to file in addition to stdout")

    # Override specific configs
    parser.add_argument("--student-model", type=str, default=None,
                        help="Override student model key (e.g., 'gpt', 'claude')")
    parser.add_argument("--teacher-model", type=str, default=None,
                        help="Override PF helper key")
    parser.add_argument("--seed-samples", type=int, default=None,
                        help="Override seed samples per dataset")
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="Override max candidates per epoch")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(verbose=args.verbose, log_file=args.log_file)
    logger = logging.getLogger("self_improving")

    # Load config
    from self_improving.configs import load_config_from_yaml
    config = load_config_from_yaml(args.config)

    # Apply CLI overrides
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.student_model is not None:
        config.student_model = args.student_model
    if args.teacher_model is not None:
        config.teacher_model = args.teacher_model
    if args.seed_samples is not None:
        config.validation.seed_samples_per_dataset = args.seed_samples
    if args.max_candidates is not None:
        config.proposal.max_candidates_per_epoch = args.max_candidates

    logger.info("=" * 60)
    logger.info("Self-Improving Skills Agent Pipeline")
    logger.info("=" * 60)
    logger.info("Experiment: %s", config.experiment_name)
    logger.info("Output: %s", config.output_dir)
    logger.info("Epochs: %d (starting from %d)", config.num_epochs, args.start_epoch)
    logger.info("Student: %s, PF helper: %s", config.student_model, config.teacher_model)

    # Create and setup pipeline
    from self_improving.pipeline import SelfImprovingPipeline
    pipeline = SelfImprovingPipeline(config)
    pipeline.setup()

    if args.dry_run:
        logger.info("Dry run complete. Setup successful.")
        logger.info("Seed: %d samples, Val: %d samples, Library: %d skills",
                     len(pipeline.val_manager.get_seed_flat()),
                     len(pipeline.val_manager.get_validation_flat()),
                     len(pipeline.library_manager.skill_ids))
        return

    # Run pipeline
    summary = pipeline.run(start_epoch=args.start_epoch)

    # Print summary
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Final library: %d skills", summary["final_library"]["total_skills"])
    for epoch_res in summary["epoch_results"]:
        ep = epoch_res["epoch"]
        seed = epoch_res["phases"].get("seed_execution", {})
        lib = epoch_res["phases"].get("library_update", {})
        logger.info(
            "  Epoch %d: %.1f%% success, %d skills accepted, library=%d",
            ep,
            seed.get("success_rate", 0) * 100,
            lib.get("accepted", 0),
            lib.get("library_size", 0),
        )


if __name__ == "__main__":
    main()
