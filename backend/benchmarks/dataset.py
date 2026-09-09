"""
Dataset adapters for benchmark framework.

Registered datasets have known structure. No arbitrary filesystem paths.
Datasets must provide train/val/test splits with labels.
"""

import csv
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkSample:
    """A single labeled audio sample."""
    file_path: Path
    label: int            # 0 = bonafide, 1 = spoof
    attack_type: str = "" # e.g., "A01", "A02", "bonafide"
    speaker_id: str = ""  # For speaker-leakage prevention


class BenchmarkDataset(ABC):
    """Abstract base class for benchmark datasets."""

    @abstractmethod
    def get_split(self, split: str) -> list[BenchmarkSample]:
        """Return samples for a split (train/val/test)."""
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return dataset name."""
        ...

    @abstractmethod
    def available_splits(self) -> list[str]:
        """Return available split names."""
        ...


class CustomWAVDataset(BenchmarkDataset):
    """User-provided WAV directory with a labels CSV.

    Expected structure:
        dataset_dir/
            labels.csv          # file,label (0=bonafide, 1=spoof)
            train/              # optional
            val/                # optional
            test/               # optional
            *.wav               # or flat structure

    labels.csv format:
        file,label,split,speaker_id,attack_type
        audio_001.wav,0,train,,
        audio_002.wav,1,test,,A01
    """

    def __init__(self, dataset_dir: str | Path):
        self._dir = Path(dataset_dir)
        self._samples: dict[str, list[BenchmarkSample]] = {}
        self._load_labels()

    def _load_labels(self):
        labels_path = self._dir / "labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"labels.csv not found in {self._dir}. "
                "Create a CSV with columns: file,label,split"
            )

        with open(labels_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_path = self._dir / row["file"]
                if not file_path.exists():
                    logger.warning(f"Audio file not found: {file_path}")
                    continue

                split = row.get("split", "test").strip()
                sample = BenchmarkSample(
                    file_path=file_path,
                    label=int(row["label"]),
                    attack_type=row.get("attack_type", ""),
                    speaker_id=row.get("speaker_id", ""),
                )

                if split not in self._samples:
                    self._samples[split] = []
                self._samples[split].append(sample)

        total = sum(len(v) for v in self._samples.values())
        logger.info(
            f"CustomWAVDataset: loaded {total} samples from {labels_path} "
            f"(splits: {list(self._samples.keys())})"
        )

    def get_split(self, split: str) -> list[BenchmarkSample]:
        if split not in self._samples:
            raise ValueError(
                f"Split '{split}' not found. "
                f"Available: {list(self._samples.keys())}"
            )
        return self._samples[split]

    def get_name(self) -> str:
        return f"custom:{self._dir.name}"

    def available_splits(self) -> list[str]:
        return list(self._samples.keys())


# --- Dataset Registry ---

REGISTERED_DATASETS: dict[str, type] = {
    "custom_wav": CustomWAVDataset,
}


def get_dataset(name: str, path: str | Path) -> BenchmarkDataset:
    """Get a registered dataset by name."""
    if name not in REGISTERED_DATASETS:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available: {list(REGISTERED_DATASETS.keys())}"
        )
    return REGISTERED_DATASETS[name](path)
