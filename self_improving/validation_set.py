"""
Validation set builder and manager.

Stores data under ``self_improving/data/`` (in the module directory).

CRITICAL: The test set and validation set come from the SAME datasets.
The test set uses samples [0, test_count) from each dataset (as recorded in
the inference output episodes). The validation set uses the REMAINING samples
[test_count, end) from each dataset — guaranteed zero overlap.

Datasets where test_count == total (Bamboogle, GAIA) have no validation
samples and are excluded.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .configs import ValidationSetConfig

logger = logging.getLogger(__name__)

# Resolve self_improving/ module directory
_MODULE_DIR = Path(__file__).resolve().parent


class ValidationSetManager:
    """Manages seed and validation datasets for the self-improving pipeline.

    Both seed (for self-improving training signal) and validation (for
    measuring generalization) are drawn from the TAIL of each dataset
    — i.e. the portion NOT used by the inference test set.
    """

    def __init__(self, config: ValidationSetConfig, rng_seed: int = 42):
        self.config = config
        # Data lives under self_improving/{data_subdir}/ (configurable
        # so MATH domain can use data_math/ instead of data/).
        subdir = getattr(config, "data_subdir", None) or "data"
        self.data_dir = _MODULE_DIR / subdir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(rng_seed)

        self._seed_data: Dict[str, List[Dict[str, Any]]] = {}
        self._val_data: Dict[str, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load seed + validation splits from the tail of each dataset.

        For each dataset:
          1. Read all samples from source JSONL
          2. Determine how many samples the test set uses (from inference outputs)
          3. Take only samples[test_count:] as the available pool
          4. Split the pool into seed and validation portions
        """
        source = Path(self.config.source_dir)
        if not source.is_dir():
            raise FileNotFoundError(f"Source data directory not found: {source}")

        # Determine test set sizes from inference outputs
        test_sizes = self._detect_test_sizes()

        for name in self.config.datasets:
            full_data = self._load_jsonl(source, name)
            test_n = test_sizes.get(name, 0)

            if test_n >= len(full_data):
                logger.info("Dataset %s: %d/%d used by test set — no validation available, skipping",
                            name, test_n, len(full_data))
                continue

            # Available pool = everything after the test set
            pool = full_data[test_n:]
            logger.info("Dataset %s: total=%d, test=%d, pool=%d",
                        name, len(full_data), test_n, len(pool))

            # Split pool into seed + validation.
            # Sentinel: seed_samples_per_dataset < 0 → use ENTIRE pool as seed,
            # no val split. This powers `evolve_val_samples: -1` for full-
            # training-data failure summarization in closed-loop configs.
            shuffled = list(pool)
            self.rng.shuffle(shuffled)
            if self.config.seed_samples_per_dataset < 0:
                seed_n = len(shuffled)
                val_n = 0
            else:
                seed_n = min(self.config.seed_samples_per_dataset, len(pool) // 2)
                val_n = min(self.config.val_samples_per_dataset, len(pool) - seed_n)

            self._seed_data[name] = shuffled[:seed_n]
            self._val_data[name] = shuffled[seed_n:seed_n + val_n]

            logger.info("  -> seed=%d, val=%d", len(self._seed_data[name]), len(self._val_data[name]))

        # Persist under self_improving/data/
        self._save_split("seed", self._seed_data)
        self._save_split("validation", self._val_data)

        total_seed = sum(len(v) for v in self._seed_data.values())
        total_val = sum(len(v) for v in self._val_data.values())
        logger.info("Total: %d seed + %d validation = %d samples (from %d datasets)",
                     total_seed, total_val, total_seed + total_val,
                     len(self._seed_data))

    def _detect_test_sizes(self) -> Dict[str, int]:
        """Detect how many samples each dataset uses in the test set.

        Reads from inference output episodes to get exact counts.
        """
        test_sizes = {}
        results_dir = Path(self.config.inference_results_dir)

        if results_dir.is_dir():
            for ds_dir in sorted(results_dir.iterdir()):
                if not ds_dir.is_dir():
                    continue
                # Look for baseline episodes
                ep_file = ds_dir / "baseline" / "base_clean_episodes.jsonl"
                if not ep_file.exists():
                    # Try other ablations
                    for abl in ds_dir.iterdir():
                        if abl.is_dir():
                            for f in abl.glob("*_episodes.jsonl"):
                                ep_file = f
                                break
                        if ep_file.exists():
                            break

                if ep_file.exists():
                    count = sum(1 for line in open(ep_file) if line.strip())
                    test_sizes[ds_dir.name] = count

        if test_sizes:
            logger.info("Detected test set sizes from %s: %s",
                        results_dir, {k: v for k, v in test_sizes.items()})
        else:
            # Fallback: use default test sizes from config
            logger.warning("No inference results found at %s, using default test_samples_per_dataset=%d",
                           results_dir, self.config.default_test_samples)
            for name in self.config.datasets:
                test_sizes[name] = self.config.default_test_samples

        # Per-dataset overrides take precedence — shrinks the test boundary
        # so more samples become available for seed/validation.
        overrides = getattr(self.config, "test_samples_overrides", None) or {}
        for name, override_n in overrides.items():
            if name in self.config.datasets:
                logger.info("Test-size override for %s: %s -> %d",
                            name, test_sizes.get(name), override_n)
                test_sizes[name] = override_n

        return test_sizes

    def _load_jsonl(self, source_dir: Path, dataset_name: str) -> List[Dict[str, Any]]:
        """Load a JSONL file from source directory."""
        candidates = [
            source_dir / f"{dataset_name}.jsonl",
        ]
        for path in candidates:
            if path.exists():
                return self._read_jsonl(path)

        # Fuzzy match
        for f in sorted(source_dir.glob("*.jsonl")):
            if dataset_name.replace("_rand1000", "") in f.stem:
                return self._read_jsonl(f)

        raise FileNotFoundError(
            f"Dataset {dataset_name} not found in {source_dir}. "
            f"Available: {[f.stem for f in source_dir.glob('*.jsonl')]}"
        )

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def _save_split(self, split_name: str, datasets: Dict[str, List[Dict]],
                    base: Path = None) -> None:
        base = base or self.data_dir
        split_dir = base / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for name, samples in datasets.items():
            path = split_dir / f"{name}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def seed_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        return self._seed_data

    @property
    def validation_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        return self._val_data

    def get_seed_flat(self) -> List[Dict[str, Any]]:
        """Return all seed samples as a flat list with dataset_name injected."""
        flat = []
        for name, samples in self._seed_data.items():
            for s in samples:
                entry = dict(s)
                entry["dataset_name"] = name
                flat.append(entry)
        return flat

    def get_validation_flat(self) -> List[Dict[str, Any]]:
        """Return all validation samples as a flat list with dataset_name injected."""
        flat = []
        for name, samples in self._val_data.items():
            for s in samples:
                entry = dict(s)
                entry["dataset_name"] = name
                flat.append(entry)
        return flat

    def get_seed_by_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        return self._seed_data.get(dataset_name, [])

    def get_validation_by_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        return self._val_data.get(dataset_name, [])

    # ------------------------------------------------------------------
    # Reload from saved split (for resuming)
    # ------------------------------------------------------------------

    def load_from_saved(self) -> bool:
        """Try to load from previously saved splits in self_improving/data/."""
        seed_dir = self.data_dir / "seed"
        val_dir = self.data_dir / "validation"
        if not seed_dir.exists() or not val_dir.exists():
            return False

        self._seed_data = {}
        for f in sorted(seed_dir.glob("*.jsonl")):
            self._seed_data[f.stem] = self._read_jsonl(f)

        self._val_data = {}
        for f in sorted(val_dir.glob("*.jsonl")):
            self._val_data[f.stem] = self._read_jsonl(f)

        if self._seed_data and self._val_data:
            total_seed = sum(len(v) for v in self._seed_data.values())
            total_val = sum(len(v) for v in self._val_data.values())
            logger.info("Loaded saved splits from %s: %d seed, %d val samples (%d datasets)",
                        self.data_dir, total_seed, total_val, len(self._seed_data))
            return True
        return False
