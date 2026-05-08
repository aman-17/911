import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

_MODEL_NAME = "Qwen/Qwen3-4B"


def load_model(
    checkpoint_path: str | None = None,
    device_map: str = "auto",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    name = checkpoint_path or _MODEL_NAME
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    return model, tokenizer
