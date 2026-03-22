def register():
    import os

    mode = os.environ.get("LLM_REGISTRATION", "activation")
    if mode == "activation":
        register_activation_models()
    elif mode == "steermoe":
        register_steermoe_models()


def register_activation_models():
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "GptOssForCausalLM", "src.activation.modelling.gpt:GptOssForCausalLM"
    )

    ModelRegistry.register_model(
        "OlmoeForCausalLM", "src.activation.modelling.olmoe:OlmoeForCausalLM"
    )

    ModelRegistry.register_model(
        "Qwen3MoeForCausalLM", "src.activation.modelling.qwen:Qwen3MoeForCausalLM"
    )

    ModelRegistry.register_model(
        "MixtralForCausalLM", "src.activation.modelling.mixtral:MixtralForCausalLM"
    )

    ModelRegistry.register_model(
        "PhiMoEForCausalLM", "src.activation.modelling.phi:PhiMoEForCausalLM"
    )


def register_steermoe_models():
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "GptOssForCausalLM", "src.steermoe.modelling.gptmoe20b:GptOssForCausalLM"
    )

    ModelRegistry.register_model(
        "OlmoeForCausalLM", "src.steermoe.modelling.olmoe7b:OlmoeForCausalLM"
    )
