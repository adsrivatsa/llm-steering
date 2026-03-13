from typing import Literal, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset


DatasetName = Literal["squad"]


class SquadFaithfulnessDataset(Dataset):
    """
    PyTorch Dataset wrapper around the SQuAD dataset.

    Each item is a (document, question) tuple:
      - document: the context passage
      - question: the associated question
    """

    def __init__(self, split: str = "train") -> None:
        if split not in ("train", "validation"):
            raise ValueError("split must be 'train' or 'validation'")

        self._hf_ds = load_dataset("rajpurkar/squad", split=split)

    def __len__(self) -> int:
        return len(self._hf_ds)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        example = self._hf_ds[idx]
        document = example["context"]
        question = example["question"]
        return document, question


class FaithfulnessDataset(Dataset):
    """
    Unified entry point for different faithfulness-style datasets.

    For now only supports:
      - 'squad'
    """

    def __init__(self, dataset_name: DatasetName, split: str = "train") -> None:
        self.dataset_name: DatasetName = dataset_name

        if dataset_name == "squad":
            self._dataset: Dataset = SquadFaithfulnessDataset(split=split)
        else:
            # This is protected by the Literal type, but kept for safety.
            raise ValueError(f"Unsupported dataset_name: {dataset_name!r}")

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self._dataset[idx]

