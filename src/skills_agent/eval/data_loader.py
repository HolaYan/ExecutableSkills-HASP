"""
Data loading utilities for evaluation.

Supports:
- Local JSONL/JSON files
- HuggingFace datasets
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


def load_local_data(
    data_path: str,
    num_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load data from local JSONL or JSON file.

    Args:
        data_path: Path to data file
        num_samples: Limit number of samples

    Returns:
        List of samples
    """
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if path.suffix == ".jsonl":
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                # Handle various JSON formats
                data = data.get("data", data.get("rows", data.get("samples", [data])))
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    logger.info(f"Loaded {len(data)} samples from {data_path}")

    if num_samples:
        data = data[:num_samples]

    return normalize_samples(data)


def load_hf_dataset(
    dataset_name: str,
    split: str = "test",
    subset: Optional[str] = None,
    num_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load data from HuggingFace dataset.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "inclusionAI/ASearcher-test-data")
        split: Dataset split (e.g., "test", "train")
        subset: Dataset subset/config name
        num_samples: Limit number of samples
        cache_dir: Cache directory for downloads

    Returns:
        List of samples
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    logger.info(f"Loading HuggingFace dataset: {dataset_name} (split: {split})")

    kwargs = {}
    if subset:
        kwargs["name"] = subset
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    try:
        dataset = load_dataset(dataset_name, split=split, **kwargs)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    data = [dict(row) for row in dataset]
    logger.info(f"Loaded {len(data)} samples from HuggingFace")

    if num_samples:
        data = data[:num_samples]

    return normalize_samples(data)


def normalize_samples(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize sample format.

    Ensures each sample has:
    - question: str
    - gold_answers: List[str]
    - sample_id: str
    - domain: str  (default "web_search")
    """
    normalized = []

    for i, item in enumerate(data):
        sample = {
            "question": item.get("question", ""),
            "sample_id": item.get("sample_id", item.get("id", str(i))),
        }

        # Handle various answer formats
        gold = item.get("gold_answers", item.get("answer", item.get("answers", [])))
        if isinstance(gold, str):
            gold = [gold]
        elif isinstance(gold, dict):
            # Some datasets use {"text": [...]} format
            gold = gold.get("text", [])
        sample["gold_answers"] = gold

        # Copy optional common fields
        if "documents" in item:
            sample["documents"] = item["documents"]
        if "sources" in item:
            sample["sources"] = item["sources"]
        if "benchmark" in item:
            sample["benchmark"] = item["benchmark"]

        # Code-domain fields (LCB jsonl carries public_tests/private_tests;
        # HumanEval+/MBPP+/BigCodeBench ship a single `eval_test_code` driver
        # plus `entry_point` / `variant`). Both shapes are needed by sandbox
        # PFs; pass through unconditionally — non-code samples that don't
        # define them are unaffected.
        for k in ("public_tests", "private_tests", "starter_code",
                  "metadata", "platform", "difficulty",
                  "eval_test_code", "public_test_code",
                  "entry_point", "variant"):
            if k in item:
                sample[k] = item[k]

        # Domain field (default to "web_search")
        sample["domain"] = item.get("domain", "web_search")

        normalized.append(sample)

    return normalized


def load_test_data(
    source: str,
    split: str = "test",
    num_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load test data from various sources.

    Args:
        source: Local file path or HuggingFace dataset name
        split: Dataset split for HF datasets
        num_samples: Limit number of samples
        cache_dir: Cache directory

    Returns:
        List of normalized samples
    """
    path = Path(source)

    # Check if it's a local file
    if path.exists() or path.suffix in [".json", ".jsonl"]:
        return load_local_data(str(path), num_samples)

    # Otherwise try as HuggingFace dataset
    return load_hf_dataset(
        dataset_name=source,
        split=split,
        num_samples=num_samples,
        cache_dir=cache_dir,
    )


def create_sample_data(num_samples: int = 10) -> List[Dict[str, Any]]:
    """
    Create sample test data for testing.

    Args:
        num_samples: Number of samples to create

    Returns:
        List of sample data
    """
    samples = []
    questions = [
        ("What is the capital of France?", ["Paris"]),
        ("Who wrote Romeo and Juliet?", ["William Shakespeare", "Shakespeare"]),
        ("What is the largest planet in our solar system?", ["Jupiter"]),
        ("What year did World War II end?", ["1945"]),
        ("What is the chemical symbol for gold?", ["Au"]),
        ("Who painted the Mona Lisa?", ["Leonardo da Vinci", "Da Vinci"]),
        ("What is the speed of light?", ["299,792,458 m/s", "3x10^8 m/s"]),
        ("What is the smallest prime number?", ["2"]),
        ("Who discovered penicillin?", ["Alexander Fleming", "Fleming"]),
        ("What is the atomic number of carbon?", ["6"]),
    ]

    for i in range(min(num_samples, len(questions))):
        question, answers = questions[i]
        samples.append({
            "question": question,
            "gold_answers": answers,
            "sample_id": f"sample_{i}",
            "documents": {
                f"doc_{i}": {
                    "title": f"Document about {answers[0]}",
                    "content": f"The answer to '{question}' is {answers[0]}.",
                    "url": f"https://example.com/doc_{i}",
                    "snippet": f"Information about {answers[0]}...",
                }
            }
        })

    return samples
