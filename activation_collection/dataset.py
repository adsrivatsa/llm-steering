from datasets import load_dataset
from torch.utils.data import Dataset


class SQuAD(Dataset):
    """
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

    def __getitem__(self, idx: int):
        return self._hf_ds[idx]
