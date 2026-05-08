import torch

from post_training.inference.inference_utils import sample_token


@torch.no_grad()
def sample_response(
    model,
    tokenizer,
    prompt: str,
    device,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> dict:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    past_key_values = None
    generated_ids = []
    log_probs = []

    # prefill: run full prompt through once to populate KV-cache
    out = model(input_ids, past_key_values=past_key_values, use_cache=True)
    logits = out.logits[:, -1, :]
    past_key_values = out.past_key_values

    for _ in range(max_new_tokens):
        token_id, log_prob = sample_token(logits[0], temperature, top_p)
        generated_ids.append(token_id)
        log_probs.append(log_prob)

        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break

        # decode step: single new token reusing cached KV
        next_token = torch.tensor([[token_id]], device=device)
        out = model(next_token, past_key_values=past_key_values, use_cache=True)
        logits = out.logits[:, -1, :]
        past_key_values = out.past_key_values

    generated_tensor = torch.tensor(generated_ids, device=device, dtype=input_ids.dtype)
    full_token_ids = torch.cat([input_ids[0], generated_tensor])

    return {
        "text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "token_ids": generated_ids,
        "log_probs": log_probs,
        "full_token_ids": full_token_ids,
        "prompt_len": prompt_len,
    }


if __name__ == "__main__":
    from post_training.data.data_tokenizer import load_model_and_tokenizer
    from post_training.inference.generation import render_prompt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(device=device)

    torch.manual_seed(0)
    raw_prompt = "Half the value of $3x-9$ is $x+37$. What is the value of $x$?"
    prompt = render_prompt(raw_prompt)

    result = sample_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=512,
        temperature=0.9,
        top_p=0.9,
    )
    print(result["text"])
