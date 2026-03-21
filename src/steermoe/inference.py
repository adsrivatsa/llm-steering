from __future__ import annotations

import json
import os
import re
from typing import Any, List, Optional

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset

from src.inference.dataset import FaithEvalCounterfactual
from src.inference.llm import ModelName, get_moe_llm


def build_faitheval_counterfactual_prompt(item: dict[str, Any]) -> str:
    document = item["context"]
    question = item["question"]
    choices = item["choices"]
    labels: List[str] = list(choices["label"])
    texts: List[str] = list(choices["text"])
    choices_str = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))

    return (
        f"Document: {document}\n\n"
        f"Question: {question}\n\n"
        f"{choices_str}\n\n"
        "Using only the document above, answer with the single letter of the correct option (A, B, C, or D)."
        "For my final answer, format it as answer=letter"
    )


def parse_mcq_output(text: str) -> str:
    """
    Parse a model output into the final answer letter.

    With thinking disabled the model reliably ends with:
        analysis{brief reasoning}assistantfinalanswer={letter}
    """
    # Primary: assistantfinal marker followed by answer=X (the consistent format)
    m = re.search(
        r"assistantfinal.*?answer\s*=\s*([A-D])", text, re.IGNORECASE | re.DOTALL
    )
    if m:
        return m.group(1).upper()
    # Fallback: any answer=X anywhere in the output
    matches = re.findall(r"answer\s*=\s*([A-D])", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    # Last resort: last standalone A/B/C/D in the text
    matches = re.findall(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", text)
    if matches:
        return matches[-1]
    return text.strip()


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def output_jsonl_path(
    checkpoint_dir: str, dataset_name: str, model_name: ModelName, pass_name: str
) -> str:
    model_safe = safe_model_name(str(model_name))
    return os.path.join(checkpoint_dir, dataset_name, f"{model_safe}_{pass_name}.jsonl")


def nonempty_lines(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def faitheval_couterfactual_inference(
    *,
    model_name: ModelName,
    checkpoint_dir: str,
    pass_name: str,
    deltas_per_layer: Optional[torch.Tensor] = None,
    n_activate: int = 0,
    n_deactivate: int = 0,
    max_new_tokens: int = 4,
    batch_size: int = 4,
    dataset_name: str = "faitheval_counterfactual",
    chat_template_kwargs: Optional[dict] = None,
) -> str:
    """
    Run inference on FaithEval-counterfactual, streaming outputs to JSONL.

    Outputs:
      - appended to a JSONL file in `checkpoint_dir` with schema:
          `checkpoint_dir/dataset_name/<model_safe>_<pass_name>.jsonl`
      - resume is handled by counting existing JSONL lines
    """
    ds = FaithEvalCounterfactual()
    outputs_jsonl_path = output_jsonl_path(
        checkpoint_dir=checkpoint_dir,
        dataset_name=dataset_name,
        model_name=model_name,
        pass_name=pass_name,
    )
    start = nonempty_lines(outputs_jsonl_path)
    if start > 0:
        remaining = max(0, len(ds) - start)
        print(
            f"Resuming {pass_name} inference from example {start} ({remaining} remaining)"
        )

    llm = get_moe_llm(model_name)

    os.makedirs(os.path.dirname(outputs_jsonl_path) or ".", exist_ok=True)
    file_mode = "a" if start > 0 else "w"

    def collate_iden(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return batch

    subset = Subset(ds, range(start, len(ds)))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_iden,
    )

    with open(outputs_jsonl_path, file_mode, encoding="utf-8") as out_f:
        for batch_items in tqdm(loader, desc="FaithEval Counterfactual inference"):
            batch_prompts = [
                build_faitheval_counterfactual_prompt(item) for item in batch_items
            ]

            batch_outputs = llm.infer(
                batch_prompts,
                deltas_per_layer=deltas_per_layer,
                n_activate=n_activate,
                n_deactivate=n_deactivate,
                max_new_tokens=max_new_tokens,
                chat_template_kwargs=chat_template_kwargs,
            )

            for item, prompt, output in zip(batch_items, batch_prompts, batch_outputs):
                record = {
                    "id": item["id"],
                    "answerKey": item["answerKey"],
                    "question": item["question"],
                    "choices": item["choices"],
                    "document": item["context"],
                    "prompt": prompt,
                    "model_output": parse_mcq_output(output),
                    "model_output_raw": output,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            exit()

    return outputs_jsonl_path
