import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 1.0:
        return logits
    return logits / temperature


def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = (cumulative - sorted_probs) > top_p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum()
    out = torch.zeros_like(probs)
    out.scatter_(-1, sorted_idx, sorted_probs)
    return out


def sample_token(logits: torch.Tensor, temperature: float, top_p: float) -> tuple[int, float]:
    logits = apply_temperature(logits, temperature)
    probs = top_p_filter(F.softmax(logits, dim=-1), top_p)
    token_id = int(torch.multinomial(probs, num_samples=1).item())
    log_prob = float(F.log_softmax(logits, dim=-1)[token_id].item())
    return token_id, log_prob


@torch.no_grad()
def generate_top_p(
    model,
    tokenizer,
    prompt: str,
    device,
    max_new_tokens: int = 2048,
    temperature: float = 0.9,
    top_p: float = 0.9,
) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    out = model(input_ids, use_cache=True)
    logits = out.logits[:, -1, :]
    past_key_values = out.past_key_values
    generated_ids = []

    for _ in range(max_new_tokens):
        token_id, _ = sample_token(logits[0], temperature, top_p)
        generated_ids.append(token_id)

        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break

        next_token = torch.tensor([[token_id]], device=device)
        out = model(next_token, past_key_values=past_key_values, use_cache=True)
        logits = out.logits[:, -1, :]
        past_key_values = out.past_key_values

    return tokenizer.decode(generated_ids, skip_special_tokens=True)
