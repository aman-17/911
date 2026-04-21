from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name: str = "Qwen/Qwen3-0.6B", device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, tokenizer
