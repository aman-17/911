import torch

from post_training.data.data_tokenizer import load_model_and_tokenizer


@torch.inference_mode()
def avg_logprob_answer(model, tokenizer, prompt, answer, device="cpu"):
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(answer)
    full_ids = torch.tensor(prompt_ids + answer_ids, device=device)
    logits = model(full_ids.unsqueeze(0)).logits.squeeze(0)
    logprobs = torch.log_softmax(logits, dim=-1)
    start = len(prompt_ids) - 1
    end = full_ids.shape[0] - 1
    t_idx = torch.arange(start, end, device=device)
    next_tokens = full_ids[start + 1 : end + 1]
    next_token_logps = logprobs[t_idx, next_tokens]
    return torch.mean(next_token_logps).item()


def sequence_logprob_draft(model, token_ids, prompt_len):
    logits = model(token_ids.unsqueeze(0)).logits.squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)
    start = prompt_len - 1  # A
    end = token_ids.shape[0] - 1  # A
    t_idx = torch.arange(start, end, device=token_ids.device)
    next_tokens = token_ids[start + 1 : end + 1]
    next_token_logps = logprobs[t_idx, next_tokens]
    return torch.sum(next_token_logps)  # B


def sequence_logprob(model, token_ids, prompt_len):
    logits = model(token_ids.unsqueeze(0)).logits.squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)
    selected = logprobs[:-1].gather(
        1, token_ids[1:].unsqueeze(-1)
    ).squeeze(-1)
    return torch.sum(selected[prompt_len - 1:])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(device=device)

    prompt = "Question: Half the value of 3x-9 is x+37. What is x?\nAnswer:"
    answer_text = r"The final answer is \boxed{83}"

    avg_logprob_val = avg_logprob_answer(
        model, tokenizer,
        prompt=prompt,
        answer=answer_text,
        device=device,
    )
    print(avg_logprob_val)

    sequence_logprob_val = avg_logprob_val * (
        len(tokenizer.encode(answer_text))
    )
    print(sequence_logprob_val)

    token_ids = torch.tensor(
        tokenizer.encode(prompt + " " + answer_text), device=device
    )
    prompt_len = len(tokenizer.encode(prompt))

    print(sequence_logprob_draft(model, token_ids, prompt_len))
    print(sequence_logprob(model, token_ids, prompt_len))

    rollouts = [
        r"\boxed{83}",
        r"The correct answer is \boxed{83}",
        r"The final answer is 83",
        r"We get \boxed{38}",
    ]
    rollout_logps = []
    for text in rollouts:
        token_ids = tokenizer.encode(prompt + " " + text)
        logprob = sequence_logprob(
            model=model,
            token_ids=torch.tensor(token_ids, device=device),
            prompt_len=prompt_len,
        )
        print(f"Answer: {text}")
        print(f"Logprob: {logprob.item():.4f}\n")
        rollout_logps.append(logprob)
