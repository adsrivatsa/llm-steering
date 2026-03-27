import json
import os
from typing import Any

from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

from src.steermoe.inference import util
from src.steermoe.inference.dataset import MQuAKE


def format_facts(case: dict[str, Any]) -> list[str]:
    facts: list[str] = []
    for rr in case.get("requested_rewrite", []):
        prompt = (rr.get("prompt") or "").strip()
        subject = (rr.get("subject") or "").strip()
        target_new = ((rr.get("target_new") or {}).get("str") or "").strip()
        if not (prompt and subject and target_new):
            continue
        try:
            prefix = prompt.format(subject).strip()
        except Exception:
            # If prompt formatting fails, fall back to a simple "subject -> new" statement.
            prefix = subject
        facts.append(f"{prefix} {target_new}".strip())
    return facts


def build_prompt(*, facts: list[str], question: str) -> str:
    facts_block = "\n".join(f"- {f}" for f in facts) if facts else "- (none)"
    return (
        "Document:\n"
        f"{facts_block}\n\n"
        f"Question: {question}\n\n"
        "You are an expert in retrieval-based question answering. "
        "Please respond with the exact answer, using only the information provided in the context. "
    )


def score(jsonl_path: str) -> dict[str, float | int]:
    total = correct = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total += 1
            output_lower = str(record.get("model_output", "")).lower()
            if any(
                str(candidate).lower() in output_lower
                for candidate in record.get("candidate_answers", [])
            ):
                correct += 1
    accuracy = correct / total if total > 0 else 0.0
    return {"total": total, "correct": correct, "accuracy": accuracy}


class FlattenedMQuAKEQuestions(Dataset):
    def __init__(self, base_ds: MQuAKE) -> None:
        self._records: list[dict[str, Any]] = []
        for case in base_ds:
            candidate_answers = [case["new_answer"], *case.get("new_answer_alias", [])]
            facts = format_facts(case)
            for question_idx, question in enumerate(case["questions"]):
                self._records.append(
                    {
                        "case_id": int(case["case_id"]),
                        "question_idx": int(question_idx),
                        "question": question,
                        "candidate_answers": candidate_answers,
                        "requested_rewrite_facts": facts,
                    }
                )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._records[idx]


def infer(
    *,
    llm: LLM,
    checkpoint_dir: str,
    pass_name: str,
    batch_size: int = 4,
) -> dict[str, float | int]:
    model_name = llm.model_config.model
    dataset_name = "mquake"

    base_ds = MQuAKE()
    ds = FlattenedMQuAKEQuestions(base_ds)

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
        for batch_examples in tqdm(loader, desc="MQuAKE inference"):
            batch_prompts = [
                [
                    {
                        "role": "user",
                        "content": build_prompt(
                            facts=item["requested_rewrite_facts"],
                            question=item["question"],
                        ),
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
                output_text = output.outputs[0].text.strip()
                record = {
                    "case_id": item["case_id"],
                    "question_idx": item["question_idx"],
                    "question": item["question"],
                    "candidate_answers": item["candidate_answers"],
                    "requested_rewrite_facts": item["requested_rewrite_facts"],
                    "prompt": prompt,
                    "model_output": output_text,
                    "model_output_raw": output_text,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return score(jsonl_path=outputs_jsonl_path)
