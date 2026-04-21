import torch

from post_training.inference.inference_utils import generate_top_p

SYSTEM_PROMPT = (
    "Your role as an assistant involves thoroughly exploring questions through "
    "a systematic long thinking process before providing the final precise and "
    "accurate solutions."
)


def render_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


__all__ = ["render_prompt", "generate_top_p"]


if __name__ == "__main__":
    from post_training.data.data_tokenizer import load_model_and_tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(device=device)

    question = "Half the value of $3x-9$ is $x+37$. What is the value of $x$?"
    prompt = render_prompt(question)

    torch.manual_seed(0)
    response = generate_top_p(model, tokenizer, prompt, device, max_new_tokens=2048)
    print(response)
