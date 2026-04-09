import argparse
import os
import re
from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src import checkpoint
from src.benchmark import (
    cf_trivia_qa,
    faitheval_counterfactual,
    faitheval_inconsistent,
    faitheval_unanswerable,
    mctest,
    mquake,
)

ModelName = Literal[
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "Qwen/Qwen3-30B-A3B",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "allenai/OLMoE-1B-7B-0125-Instruct",
    "microsoft/Phi-3.5-MoE-instruct",
]


def save_heatmap(
    model_name: ModelName,
    algorithm: str,
    visualization_dir: str,
    benchmark_name: str,
    data: List[List[int]],
    act_labels: List[int],
    deact_labels: List[int],
):
    fig, ax = plt.subplots(
        figsize=(max(6, len(deact_labels) * 0.6), max(5, len(act_labels) * 0.5))
    )

    heatmap_kw: dict = {
        "data": data,
        "ax": ax,
        "xticklabels": deact_labels,
        "yticklabels": act_labels[::-1],
        "annot": True,
        "fmt": ".3f",
        "cmap": "viridis",
    }

    if np.isfinite(data).any():
        heatmap_kw["vmin"] = float(np.nanmin(data))
        heatmap_kw["vmax"] = float(np.nanmax(data))

    sns.heatmap(**heatmap_kw)
    ax.set_xlabel("n_deactivated")
    ax.set_ylabel("n_activated")
    ax.set_title(f"{benchmark_name}\n{algorithm} ({model_name})")
    fig.tight_layout()

    folder = os.path.join(visualization_dir, "heatmap", model_name)
    os.makedirs(folder, exist_ok=True)

    out_path = os.path.join(folder, f"{algorithm}_{benchmark_name}_heatmap.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(
    model_name: ModelName,
    task: str,
    algorithm: str,
    inference_dir: str,
    visualization_dir: str,
):
    model_name = checkpoint.safe_model_name(model_name)

    if task == "faithfulness":
        benchmarks = {
            "cf_trivia_qa": cf_trivia_qa.score,
            "faitheval_counterfactual": faitheval_counterfactual.score,
            "faitheval_inconsistent": faitheval_inconsistent.score,
            "faitheval_unanswerable": faitheval_unanswerable.score,
            "mctest": mctest.score,
            "mquake": mquake.score,
        }

    pattern = rf"^{algorithm}_a(\d+)_d(\d+)"

    cum_heatmap = None

    for benchmark_name, score_fn in benchmarks.items():
        n_activateds, n_deactivateds = set({}), set({})
        scores = {}

        path = os.path.join(inference_dir, model_name, benchmark_name)
        for file_name in os.listdir(path):
            match = re.search(pattern, file_name)
            if not match:
                continue
            n_activated, n_deactivated = int(match.group(1)), int(match.group(2))
            n_activateds.add(n_activated)
            n_deactivateds.add(n_deactivated)

            file_path = os.path.join(path, file_name)
            acc = score_fn(file_path)["accuracy"]

            scores[(n_activated, n_deactivated)] = acc

        act_labels = sorted(list(n_activateds))
        deact_labels = sorted(list(n_deactivateds))
        heatmap = [[0] * len(deact_labels) for _ in range(len(act_labels))]
        if not cum_heatmap:
            cum_heatmap = [[0] * len(deact_labels) for _ in range(len(act_labels))]

        for i, act in enumerate(act_labels):
            for j, deact in enumerate(deact_labels):
                heatmap[i][j] = scores[(act, deact)]
                cum_heatmap[i][j] += scores[(act, deact)]

        heatmap = np.asarray(heatmap, dtype=np.float64)[::-1, :]
        save_heatmap(
            model_name=model_name,
            algorithm=algorithm,
            visualization_dir=visualization_dir,
            benchmark_name=benchmark_name,
            data=heatmap,
            act_labels=act_labels,
            deact_labels=deact_labels,
        )

    cum_heatmap = np.array(cum_heatmap, dtype=np.float64)[::-1, :] / len(benchmarks)
    save_heatmap(
        model_name=model_name,
        algorithm=algorithm,
        visualization_dir=visualization_dir,
        benchmark_name="all",
        data=cum_heatmap,
        act_labels=act_labels,
        deact_labels=deact_labels,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algorithm", type=str, choices=["steermoe", "toksteermoe"], default="steermoe"
    )
    parser.add_argument(
        "--task", type=str, choices=["faithfulness"], default="faithfulness"
    )
    parser.add_argument(
        "--llm",
        "--model",
        dest="model_name",
        type=str,
        default="allenai/OLMoE-1B-7B-0125-Instruct",
    )
    parser.add_argument("--inference-dir", type=str, default="inference")
    parser.add_argument("--visualization-dir", type=str, default="visualization")

    args = parser.parse_args()
    algorithm: str = args.algorithm
    task: str = args.task
    model_name: ModelName = args.model_name
    inference_dir: str = args.inference_dir
    visualization_dir: str = args.visualization_dir

    main(
        algorithm=algorithm,
        task=task,
        model_name=model_name,
        inference_dir=inference_dir,
        visualization_dir=visualization_dir,
    )
