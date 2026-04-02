import pytest
from transformers import PretrainedConfig


def _make_dummy_hf_config(model, *args, **kwargs):
    """Return a minimal PretrainedConfig for offline unit tests.

    Inspects *model* so that multimodal-aware tests (Qwen2-VL, etc.)
    get a config whose architecture is resolved correctly by vLLM.
    """
    cfg = PretrainedConfig()
    cfg.hidden_size = 768
    cfg.num_hidden_layers = 12
    cfg.num_attention_heads = 12
    cfg.intermediate_size = 3072
    cfg.max_position_embeddings = 131072
    cfg.vocab_size = 50257

    model_str = str(model).lower()
    if "qwen2-vl" in model_str:
        cfg.model_type = "qwen2_vl"
        cfg.architectures = ["Qwen2VLForConditionalGeneration"]
        cfg.vision_config = PretrainedConfig()
        cfg.vision_config.hidden_size = 768
    else:
        cfg.model_type = "gpt2"
        cfg.architectures = ["GPT2LMHeadForCausalLM"]

    return cfg


@pytest.fixture(autouse=True)
def _skip_offline_model_resolution(request, monkeypatch):
    """Prevent EngineArgs from resolving model paths or loading HF configs
    via network / local snapshot.  These unit tests only exercise
    config-creation logic and never need real weights on disk.

    Tests that need real HF configs (e.g. integration tests that load
    actual model weights) can opt out by using the ``real_hf_config``
    marker::

        @pytest.mark.real_hf_config
        def test_my_integration_test(): ...
    """
    if request.node.get_closest_marker("real_hf_config"):
        return

    def _passthrough(model, revision=None):
        return model

    monkeypatch.setattr(
        "vllm.transformers_utils.repo_utils.get_model_path",
        _passthrough,
    )
    monkeypatch.setattr(
        "vllm.engine.arg_utils.get_model_path",
        _passthrough,
    )
    monkeypatch.setattr(
        "vllm.config.model.get_config",
        _make_dummy_hf_config,
    )
