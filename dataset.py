import json

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


class FaithEvalUnanswerable(Dataset):
    """
    Each context has had its answer-supporting sentence removed, making every
    question unanswerable from the provided context.

    Each item is a (document, question) tuple:
      - document: the modified context passage (answer removed)
      - question: the associated question
    """

    def __init__(self) -> None:
        self._hf_ds = load_dataset(
            "Salesforce/FaithEval-unanswerable-v1.0", split="test"
        )

    def __len__(self) -> int:
        return len(self._hf_ds)

    def __getitem__(self, idx: int):
        return self._hf_ds[idx]


class FaithEvalInconsistent(Dataset):
    """
    Each context contains two documents: the original and a modified version
    where one answer-bearing sentence has been replaced to support a new,
    inconsistent answer.

    Each item is a (document, question) tuple:
      - document: context containing both the original and modified document
      - question: the associated question
    """

    def __init__(self) -> None:
        self._hf_ds = load_dataset(
            "Salesforce/FaithEval-inconsistent-v1.0", split="test"
        )

    def __len__(self) -> int:
        return len(self._hf_ds)

    def __getitem__(self, idx: int):
        return self._hf_ds[idx]


class FaithEvalCounterfactual(Dataset):
    """
    Each context contains fabricated information that supports a counterfactual
    (factually wrong) multiple-choice answer.

    Each item is a CounterfactualItem TypedDict with keys:
      - id: example identifier
      - question: the multiple-choice question
      - answerKey: the label of the context-supported answer (e.g. "B")
      - choices: parsed dict with "label" (list[str]) and "text" (list[str])
      - document: the context with fabricated supporting evidence
    """

    def __init__(self) -> None:
        self._hf_ds = load_dataset(
            "Salesforce/FaithEval-counterfactual-v1.0", split="test"
        )

    def __len__(self) -> int:
        return len(self._hf_ds)

    def __getitem__(self, idx: int):
        return self._hf_ds[idx]
