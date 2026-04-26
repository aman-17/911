import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

_MODEL_NAME = "allenai/OLMo-2-0425-1B-SFT"


def load_model(device_map: str = "auto") -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_NAME,
        dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    return model, tokenizer
