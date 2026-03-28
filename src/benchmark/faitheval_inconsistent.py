import json
import os
import re
from typing import Any

from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset
from vllm import LLM, SamplingParams

from src.benchmark import util
from src.benchmark.dataset import FaithEvalInconsistent


def build_prompt(item: dict[str, Any]) -> str:
    document = item["context"]
    question = item["question"]

    return (
        f"Document: {document}\n\n"
        f"Question: {question}\n\n"
        "You are an expert in retrieval-based question answering. "
        "Please respond with the exact answer, using only the information provided in the context. "
        'If there is conflicting information or multiple answers in the context, the answer should be "conflict".'
    )


def parse_output(text: str) -> str:
    text = text.strip()
    match = re.search(
        r"(?:conflict|conflicting|disagreement|inconsistent|contradictory|contradiction|inconsistency|two answers|2 answers|multiple answers)\.?",
        text,
        re.IGNORECASE,
    )
    if match:
        return "conflict"
    return text


def score(jsonl_path: str) -> dict[str, float | int]:
    total = correct = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total += 1
            if record["model_output"] == "conflict":
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
    dataset_name = "faitheval_inconsistent"

    ds = FaithEvalInconsistent()
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
    max_context_tokens = 4096
    tokenizer = llm.get_tokenizer()
    skipped_for_length = 0

    def count_input_tokens(messages: list[dict[str, str]]) -> int:
        try:
            tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            if isinstance(tokenized, list):
                return len(tokenized)
            if hasattr(tokenized, "shape"):
                return int(tokenized.shape[-1])
            return len(tokenized)
        except Exception:
            prompt_text = "\n".join(m["content"] for m in messages)
            return len(tokenizer.encode(prompt_text, add_special_tokens=True))

    with open(outputs_jsonl_path, file_mode, encoding="utf-8") as out_f:
        for batch_examples in tqdm(loader, desc="FaithEval Inconsistent inference"):
            raw_batch_prompts = [
                [
                    {
                        "role": "user",
                        "content": build_prompt(item),
                    }
                ]
                for item in batch_examples
            ]
            filtered_examples: list[dict[str, Any]] = []
            batch_prompts: list[list[dict[str, str]]] = []
            for item, prompt in zip(batch_examples, raw_batch_prompts):
                input_tokens = count_input_tokens(prompt)
                if input_tokens + sampling_params.max_tokens > max_context_tokens:
                    skipped_for_length += 1
                    continue
                filtered_examples.append(item)
                batch_prompts.append(prompt)
            if not batch_prompts:
                continue

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
                filtered_examples, batch_prompts, batch_outputs
            ):
                output_text = output.outputs[0].text
                record = {
                    "id": item["qid"],
                    "answers": item["answers"],
                    "question": item["question"],
                    "document": item["context"],
                    "prompt": prompt,
                    "model_output": parse_output(output_text),
                    "model_output_raw": output_text,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if skipped_for_length > 0:
        print(
            f"Skipped {skipped_for_length} examples where input_tokens + max_tokens exceeded {max_context_tokens}."
        )

    return score(jsonl_path=outputs_jsonl_path)
