import json
import os
import re
from typing import Any, List

from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset
from vllm import LLM, SamplingParams

from src.steermoe.inference import util
from src.steermoe.inference.dataset import FaithEvalCounterfactual


_LETTER_LABELS = ["A", "B", "C", "D", "E"]


def build_prompt(item: dict[str, Any]) -> str:
    document = item["context"]
    question = item["question"]
    choices = item["choices"]
    texts: List[str] = list(choices["text"])
    choices_str = "\n".join(
        f"{_LETTER_LABELS[i]}. {text}" for i, text in enumerate(texts)
    )

    return (
        f"Document: {document}\n\n"
        f"Question: {question}\n\n"
        f"{choices_str}\n\n"
        "You are an expert in retrieval-based question answering. "
        "Please respond with the exact answer, using only the information provided in the context."
    )


def parse_mcq_output(text: str) -> str:
    # Primary: letter at the start of the output (e.g. "A." or "A. some text")
    m = re.match(r"\s*([A-D])[^a-z]", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: any standalone A/B/C/D in the text
    matches = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", text, re.IGNORECASE)
    if matches:
        return matches[0].upper()
    return text.strip()


def score(jsonl_path: str) -> dict[str, float | int]:
    total = correct = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total += 1
            if record["model_output"].upper() == record["answerKey"].upper():
                correct += 1
    accuracy = correct / total if total > 0 else 0.0
    return {"total": total, "correct": correct, "accuracy": accuracy}


def infer(
    *,
    llm: LLM,
    checkpoint_dir: str,
    pass_name: str,
    batch_size: int = 4,
) -> str:
    model_name = llm.model_config.model
    dataset_name = "faitheval_counterfactual"

    ds = FaithEvalCounterfactual()
    outputs_jsonl_path = util.output_jsonl_path(
        checkpoint_dir=checkpoint_dir,
        dataset_name=dataset_name,
        model_name=model_name,
        pass_name=pass_name,
    )
    start = util.nonempty_lines(outputs_jsonl_path)
    if start > 0:
        remaining = max(0, len(ds) - start)
        print(
            f"Resuming {pass_name} inference from example {start} ({remaining} remaining)"
        )

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

    sampling_params = SamplingParams(
        temperature=0.0, top_p=1, top_k=1, min_p=0, max_tokens=16, seed=0
    )

    with open(outputs_jsonl_path, file_mode, encoding="utf-8") as out_f:
        for batch_examples in tqdm(loader, desc="FaithEval Counterfactual inference"):
            batch_prompts = [
                [
                    {
                        "role": "user",
                        "content": build_prompt(item),
                    }
                ]
                for item in batch_examples
            ]

            batch_outputs = llm.chat(
                batch_prompts,
                sampling_params,
                use_tqdm=False,
                chat_template_kwargs={
                    "enable_thinking": False,
                    "reasoning_effort": "low",
                },
            )

            for item, prompt, output in zip(
                batch_examples, batch_prompts, batch_outputs
            ):
                output_text = output.outputs[0].text
                record = {
                    "id": item["id"],
                    "answerKey": item["answerKey"],
                    "question": item["question"],
                    "choices": item["choices"],
                    "document": item["context"],
                    "prompt": prompt,
                    "model_output": parse_mcq_output(output_text),
                    "model_output_raw": output_text,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return score(jsonl_path=outputs_jsonl_path)
