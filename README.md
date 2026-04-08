# 🚀 Risk-Scaled & Token-Aware MoE Steering

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-EE4C2C.svg)](https://pytorch.org/)
[![vLLM](https://img.shields.io/badge/vLLM-0.18.0-green.svg)](https://github.com/vllm-project/vllm)

This repository contains the implementation for **Risk-Scaled and Token-Aware Steering in Mixture-of-Experts (MoE) Language Models**. We extend the SteerMoE framework by introducing granular, dynamic inference-time interventions that improve model safety and faithfulness without requiring fine-tuning.

## ✨ Key Features

**$\Delta$-Scaled Steering:** Replaces uniform steering strength with interventions scaled proportionally to each expert's risk difference score.
**Token-Aware Granularity:** Preserves token-level routing granularity, applying steering selectively rather than via a sin.
**vLLM Integration:** Implemented seamlessly via vLLM general plugins for efficient inference.

## ⚙️ Requirements & Installation

This project requires **Python 3.12** and utilizes a custom `vLLM` build. 

**Core Dependencies:**
* `torch == 2.10.0`
* `transformers == 4.57.6`
* `vllm == 0.18.0` (Loaded via local wheel for custom plugin registration)

**Setup:**
We recommend using `uv` for environment management as defined in our `pyproject.toml`.

```bash
# Clone the repository
git clone [https://github.com/yourusername/llm-steering.git](https://github.com/yourusername/llm-steering.git)
cd llm-steering
```

```bash
# Install dependencies using uv
uv venv
source .venv/bin/activate
uv pip install -e .
```


*Note: Ensure the local `vllm-0.18.0+cu126-cp312-cp312-linux_x86_64.whl` wheel is present in the root directory prior to installation.*

## 🧪 Evaluation & Benchmarks

Our approach is evaluated across two primary axes:
1. **Faithfulness:** Tested against FaithEval, CF-TriviaQA, and MQuAKE to measure resilience against parametric memory drift.
2. **Safety:** Evaluated using TDC2023, MaliciousInstruct, and AdvBench, utilizing Llama-Guard-3-8B as a judge.

## 👥 Authors
Abhinav Srivatsa, Carl Cheng, Nima Kelidari, Bhushan Shankar Halasagi, Harsh Sharma *(CSCI 544)*
