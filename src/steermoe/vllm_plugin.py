def register():
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "GptOssForCausalLM",
        "src.steermoe.modelling.gptmoe20b:GptOssForCausalLM",
    )
