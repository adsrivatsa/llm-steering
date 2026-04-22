import argparse
import os
import torch
from transformers import AutoConfig

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False

from src.benchmark import (
    cf_trivia_qa,
    faitheval_counterfactual,
    faitheval_unanswerable,
    faitheval_inconsistent,
    mctest,
    mquake,
    squad,
)

# Crucial override to launch the new VLLM plugin mode we registered
os.environ["LLM_REGISTRATION"] = "steermoe3d"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

from typing import Literal, Tuple
from vllm import LLM
from src.vllm_plugin import register as register_vllm_models

ModelName = Literal[
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "Qwen/Qwen3-30B-A3B",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "allenai/OLMoE-1B-7B-0125-Instruct",
    "microsoft/Phi-3.5-MoE-instruct",
]

def resolve_parallelism(model_name: str) -> tuple[int, int]:
    n_gpus = torch.cuda.device_count()
    print("GPUs available:", n_gpus)
    if n_gpus <= 1:
        return 1, 1

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    kv_heads = getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", None))

    if kv_heads is not None and kv_heads % n_gpus == 0:
        return n_gpus, 1

    return 1, n_gpus

def steer3d(
    llm: LLM,
    delta: torch.Tensor,
    n_activated: int,
    n_deactivated: int,
    eps: float = 0.01,
) -> None:
    # We now dispatch the raw [L, D, E] matrix directly directly to the runtime
    # without calculating hard manual weights statically.
    delta = delta.float()
    
    def apply(self):
        model = self.model_runner.model
        model.add_steermoe3d_args(delta.clone(), n_activated, n_deactivated, eps)
        return ""

    llm.collective_rpc(apply)

def main(
    model_name: ModelName,
    task: str,
    delta_path: str,
    inference_dir: str,
    n_activated: int,
    n_deactivated: int,
):
    register_vllm_models()

    tp, pp = resolve_parallelism(model_name)
    print(f"Parallelism: tensor_parallel_size={tp}, pipeline_parallel_size={pp}")

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        gpu_memory_utilization=0.98,
        max_num_seqs=1,
        enforce_eager=True,
        enable_prefix_caching=False,
        trust_remote_code=True,
    )

    if not os.path.exists(delta_path):
        raise FileNotFoundError(f"Trained 3D DELTA tensor not found at {delta_path}!")
        
    print(f"Loading 3D DELTA from {delta_path}...")
    delta = torch.load(delta_path, map_location="cpu", weights_only=False) # [L, D, E]

    steer3d(
        llm=llm,
        delta=delta,
        n_activated=n_activated,
        n_deactivated=n_deactivated,
        eps=0.01,
    )

    pass_name = f"steermoe3d_a{n_activated}_d{n_deactivated}"

    if task == "faithfulness":
        print("Running faitheval_counterfactual...")
        score = faitheval_counterfactual.infer(llm=llm, checkpoint_dir=inference_dir, pass_name=pass_name, batch_size=4)
        print("faitheval_counterfactual:", score)

        print("Running faitheval_unanswerable...")
        score = faitheval_unanswerable.infer(llm=llm, checkpoint_dir=inference_dir, pass_name=pass_name, batch_size=4)
        print("faitheval_unanswerable:", score)

        print("Running faitheval_inconsistent...")
        score = faitheval_inconsistent.infer(llm=llm, checkpoint_dir=inference_dir, pass_name=pass_name, batch_size=4)
        print("faitheval_inconsistent:", score)

        print("Running cf_trivia_qa...")
        score = cf_trivia_qa.infer(llm=llm, checkpoint_dir=inference_dir, pass_name=pass_name, batch_size=4)
        print("cf_trivia_qa:", score)

        print("Running mquake...")
        score = mquake.infer(llm=llm, checkpoint_dir=inference_dir, pass_name=pass_name, batch_size=4)
        print("mquake:", score)

        print("Running mctest...")
        score = mctest.infer(llm=llm, checkpoint_dir=inference_dir, pass_name=pass_name, batch_size=4)
        print("mctest:", score)

    elif task == "squad":
        if _WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY"):
            wandb.init(
                project="tokenaware-steering-moe",
                entity=os.environ.get("WANDB_ENTITY", "VLAvengers"),
                group="squad_inference",
                name=f"squad_{model_name.split('/')[-1]}_A{n_activated}_D{n_deactivated}",
                config={
                    "model": model_name,
                    "experts_activated": n_activated,
                    "experts_deactivated": n_deactivated,
                    "task": "squad"
                }
            )

        print("Running squad...")
        score = squad.infer(llm=llm, checkpoint_dir=inference_dir, pass_name=pass_name, batch_size=4)
        print("squad:", score)
        
        if _WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY") and wandb.run is not None:
            wandb.log({
                "squad_exact_match": score["exact_match"],
                "squad_f1": score["f1"],
                "experts_activated": n_activated,
                "experts_deactivated": n_deactivated
            })
            wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["faithfulness", "squad"], default="faithfulness")
    parser.add_argument("--llm", "--model", dest="model_name", type=str, default="allenai/OLMoE-1B-7B-0125-Instruct")
    # Instead of pulling activations_dir sequentially, we just pull the natively trained delta.pt
    parser.add_argument("--delta-path", type=str, required=True, help="Path to your trained_delta.pt file.")
    parser.add_argument("--inference-dir", type=str, default="inference")
    parser.add_argument("--experts-activated", type=int, default=2)
    parser.add_argument("--experts-deactivated", type=int, default=16)

    args = parser.parse_args()
    
    main(
        model_name=args.model_name,
        task=args.task,
        delta_path=args.delta_path,
        inference_dir=args.inference_dir,
        n_activated=args.experts_activated,
        n_deactivated=args.experts_deactivated,
    )
