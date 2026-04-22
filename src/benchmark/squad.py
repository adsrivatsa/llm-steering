import json
import os
import re
import string
import collections
from typing import Any

from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset
from vllm import LLM, SamplingParams

from src.benchmark import util
from src.benchmark.dataset import SQuAD


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def build_prompt(item: dict[str, Any]) -> str:
    context = item["context"]
    question = item["question"]

    return (
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "You are an expert in reading comprehension and question answering. "
        "Please respond with the exact answer, using only the information provided in the Context. Keep your answer as short as possible.\nAnswer:"
    )


def score(jsonl_path: str) -> dict[str, float | int]:
    total = 0
    em = 0.0
    f1 = 0.0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total += 1
            
            ground_truths = record["answers"]
            prediction = record["model_output"]
            
            em += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
            f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
            
    em_acc = em / total if total > 0 else 0.0
    f1_acc = f1 / total if total > 0 else 0.0
    return {"total": total, "exact_match": em_acc, "f1": f1_acc}


def infer(
    *,
    llm: LLM,
    checkpoint_dir: str,
    pass_name: str,
    batch_size: int = 4,
) -> str:
    model_name = llm.model_config.model
    dataset_name = "squad"

    ds = SQuAD()
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
        temperature=0.0, top_p=1, top_k=1, min_p=0, max_tokens=32, seed=0
    )

    with open(outputs_jsonl_path, file_mode, encoding="utf-8") as out_f:
        for batch_examples in tqdm(loader, desc="SQuAD inference"):
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
                    "question": item["question"],
                    "context": item["context"],
                    "answers": item["answers"]["text"], # List of acceptable answers
                    "prompt": prompt,
                    "model_output": output_text,
                    "model_output_raw": output_text,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return score(jsonl_path=outputs_jsonl_path)
