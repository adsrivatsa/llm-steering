def register():
    import os

    mode = os.environ.get("LLM_REGISTRATION", "activation")
    if mode == "activation":
        register_activation_models()
    elif mode == "steermoe":
        register_steermoe_models()
    elif mode == "toksteermoe":
        register_toksteermoe_models()
    elif mode == "steermoe3d":
        register_steermoe3d_models()


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
        "GptOssForCausalLM", "src.steermoe.modelling.gpt:GptOssForCausalLM"
    )

    ModelRegistry.register_model(
        "OlmoeForCausalLM", "src.steermoe.modelling.olmoe:OlmoeForCausalLM"
    )

    ModelRegistry.register_model(
        "Qwen3MoeForCausalLM", "src.steermoe.modelling.qwen:Qwen3MoeForCausalLM"
    )

    ModelRegistry.register_model(
        "MixtralForCausalLM", "src.steermoe.modelling.mixtral:MixtralForCausalLM"
    )

    ModelRegistry.register_model(
        "PhiMoEForCausalLM", "src.steermoe.modelling.phi:PhiMoEForCausalLM"
    )


def register_toksteermoe_models():
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "GptOssForCausalLM", "src.toksteermoe.modelling.gpt:GptOssForCausalLM"
    )

    ModelRegistry.register_model(
        "OlmoeForCausalLM", "src.toksteermoe.modelling.olmoe:OlmoeForCausalLM"
    )

    ModelRegistry.register_model(
        "Qwen3MoeForCausalLM", "src.toksteermoe.modelling.qwen:Qwen3MoeForCausalLM"
    )

    ModelRegistry.register_model(
        "MixtralForCausalLM", "src.toksteermoe.modelling.mixtral:MixtralForCausalLM"
    )

    ModelRegistry.register_model(
        "PhiMoEForCausalLM", "src.toksteermoe.modelling.phi:PhiMoEForCausalLM"
    )

def register_steermoe3d_models():
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "GptOssForCausalLM", "src.steermoe3d.modelling.gpt:GptOssForCausalLM"
    )

    ModelRegistry.register_model(
        "OlmoeForCausalLM", "src.steermoe3d.modelling.olmoe:OlmoeForCausalLM"
    )

    ModelRegistry.register_model(
        "Qwen3MoeForCausalLM", "src.steermoe3d.modelling.qwen:Qwen3MoeForCausalLM"
    )

    ModelRegistry.register_model(
        "MixtralForCausalLM", "src.steermoe3d.modelling.mixtral:MixtralForCausalLM"
    )

    ModelRegistry.register_model(
        "PhiMoEForCausalLM", "src.steermoe3d.modelling.phi:PhiMoEForCausalLM"
    )
