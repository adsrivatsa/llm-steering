import json
from typing import Any

import requests
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import Dataset


class FaithEvalUnanswerable(Dataset):
    def __init__(self) -> None:
        self._hf_ds = load_dataset(
            "Salesforce/FaithEval-unanswerable-v1.0", split="test"
        )

    def __len__(self) -> int:
        return len(self._hf_ds)

    def __getitem__(self, idx: int):
        return self._hf_ds[idx]


class FaithEvalInconsistent(Dataset):
    def __init__(self) -> None:
        self._hf_ds = load_dataset(
            "Salesforce/FaithEval-inconsistent-v1.0", split="test"
        )

    def __len__(self) -> int:
        return len(self._hf_ds)

    def __getitem__(self, idx: int):
        return self._hf_ds[idx]


class FaithEvalCounterfactual(Dataset):
    def __init__(self) -> None:
        self._hf_ds = load_dataset(
            "Salesforce/FaithEval-counterfactual-v1.0", split="test"
        )

    def __len__(self) -> int:
        return len(self._hf_ds)

    def __getitem__(self, idx: int):
        return self._hf_ds[idx]


class CFTriviaQA(Dataset):
    def __init__(self) -> None:
        url = "https://raw.githubusercontent.com/google-research-datasets/cf_triviaqa/refs/heads/main/har_dataset.jsonl"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        self._records = [
            json.loads(line) for line in r.text.splitlines() if line.strip()
        ]

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._records[idx]


class MQuAKE(Dataset):
    def __init__(self) -> None:
        url = "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/MQuAKE-CF-3k-v2.json"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        self._records = r.json()

        url = "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/MQuAKE-T.json"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        self._records.extend(r.json())

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._records[idx]


class MCTest(Dataset):
    def __init__(self) -> None:
        mc500_test_url = "https://huggingface.co/datasets/sagnikrayc/mctest/resolve/refs%2Fconvert%2Fparquet/mc500/test/0000.parquet"
        mc160_test_url = "https://huggingface.co/datasets/sagnikrayc/mctest/resolve/refs%2Fconvert%2Fparquet/mc160/test/0000.parquet"
        mc500_test = load_dataset("parquet", data_files=[mc500_test_url], split="train")
        mc160_test = load_dataset("parquet", data_files=[mc160_test_url], split="train")
        self._hf_ds = concatenate_datasets([mc500_test, mc160_test])

    def __len__(self) -> int:
        return len(self._hf_ds)

    def __getitem__(self, idx: int):
        return self._hf_ds[idx]
