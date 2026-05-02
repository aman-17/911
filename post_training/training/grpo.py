import re
import time
import torch
import torch.nn.functional as F
from pathlib import Path

from post_training.inference.generation import render_prompt
from post_training.inference.rollout import sample_response


def _extract_boxed(text: str) -> str | None:
    match = re.search(r"\\boxed\{", text)
    if not match:
        return None
    start, depth = match.end(), 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def _normalize(ans: str) -> str:
    return ans.strip().replace(" ", "").replace(",", "").lower()


def compute_reward(response_text: str, ground_truth: str) -> float:
    predicted = _extract_boxed(response_text)
    if predicted is None:
        return 0.0
    gt = _extract_boxed(ground_truth) or ground_truth
    return 1.0 if _normalize(predicted) == _normalize(gt) else 0.0


def compute_grpo_loss(
    model,
    tokenizer,
    example: dict,
    device,
    num_rollouts: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> dict:
    question = example["problem"]
    ground_truth = example.get("answer") or example.get("solution", "")
    prompt = render_prompt(question)

    # Phase 1 — rollout (no_grad is set inside sample_response)
    samples = []
    for _ in range(num_rollouts):
        sample = sample_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        sample["reward"] = compute_reward(sample["text"], ground_truth)
        samples.append(sample)

    # Phase 2 — group-relative advantages
    rewards = torch.tensor(
        [s["reward"] for s in samples], dtype=torch.float32, device=device
    )
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # Phase 3 — recompute log probs with grad, build policy loss
    policy_losses = []
    for i, sample in enumerate(samples):
        full_ids = sample["full_token_ids"].unsqueeze(0).to(device)  # [1, L]
        prompt_len = sample["prompt_len"]
        gen_len = full_ids.shape[1] - prompt_len
        if gen_len == 0:
            continue

        # Forward pass over the full sequence (prompt + response) with gradients
        logits = model(full_ids).logits  # [1, L, vocab]

        # logits[:, t, :] is the distribution over token t+1.
        # Response tokens live at positions [prompt_len .. prompt_len+gen_len-1],
        # so their predictors are logits at [prompt_len-1 .. prompt_len+gen_len-2].
        response_logits = logits[:, prompt_len - 1 : prompt_len - 1 + gen_len, :]
        response_ids = full_ids[:, prompt_len:]  # [1, gen_len]

        token_log_probs = (
            F.log_softmax(response_logits, dim=-1)
            .gather(dim=-1, index=response_ids.unsqueeze(-1))
            .squeeze(-1)  # [1, gen_len]
        )

        # Scalar loss contribution for this rollout
        policy_losses.append(-advantages[i] * token_log_probs.mean())

    if not policy_losses:
        dummy = torch.tensor(0.0, device=device, requires_grad=True)
        return {
            "loss_tensor": dummy,
            "loss": 0.0,
            "rewards": rewards.tolist(),
            "samples": [{"gen_len": len(s["token_ids"])} for s in samples],
        }

    loss = torch.stack(policy_losses).mean()
    return {
        "loss_tensor": loss,
        "loss": loss.item(),
        "rewards": rewards.tolist(),
        "samples": [{"gen_len": len(s["token_ids"])} for s in samples],
    }


def train_rlvr_grpo(
    model,
    tokenizer,
    math_data,
    device,
    steps=None,
    num_rollouts=2,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9,
    lr=1e-5,
    checkpoint_every=50,
    checkpoint_dir=".",
    csv_log_path=None,
):
    if steps is None:
        steps = len(math_data)

    # A — optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    current_step = 0
    if csv_log_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_log_path = f"train_rlvr_grpo_metrics_{timestamp}.csv"
    csv_log_path = Path(csv_log_path)

    try:
        # B — training steps
        for step in range(steps):
            # C — zero gradients
            optimizer.zero_grad()
            current_step = step + 1
            example = math_data[step % len(math_data)]

            # D — GRPO loss
            stats = compute_grpo_loss(
                model=model,
                tokenizer=tokenizer,
                example=example,
                device=device,
                num_rollouts=num_rollouts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            # E — backward
            stats["loss_tensor"].backward()

            # F — gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # G — weight update
            optimizer.step()

            # H — logging
            reward_avg = torch.tensor(stats["rewards"]).mean().item()
            step_tokens = sum(s["gen_len"] for s in stats["samples"])
            avg_response_len = (
                step_tokens / len(stats["samples"]) if stats["samples"] else 0.0
            )
            _append_csv_metrics(
                csv_log_path, current_step, steps,
                stats["loss"], reward_avg, avg_response_len,
            )
            print(
                f"[Step {current_step}/{steps}] "
                f"loss={stats['loss']:.4f} "
                f"reward_avg={reward_avg:.3f} "
                f"avg_resp_len={avg_response_len:.1f}"
            )

            # I — periodic checkpoint
            if checkpoint_every and current_step % checkpoint_every == 0:
                ckpt_path = _save_checkpoint(
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    step=current_step,
                )
                print(f"Saved checkpoint to {ckpt_path}")

    # J — interrupt checkpoint
    except KeyboardInterrupt:
        ckpt_path = _save_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            step=max(1, current_step),
            suffix="interrupt",
        )
        print(f"\nKeyboardInterrupt. Saved checkpoint to {ckpt_path}")

    return model


def _save_checkpoint(model, checkpoint_dir, step, suffix=""):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix_str = f"-{suffix}" if suffix else ""
    ckpt_path = checkpoint_dir / f"qwen3-0.6B-rlvr-grpo-step{step:05d}{suffix_str}.pth"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def _append_csv_metrics(csv_log_path, step_idx, total_steps, loss, reward_avg, avg_response_len):
    if not csv_log_path.exists():
        csv_log_path.write_text(
            "step,total_steps,loss,reward_avg,avg_response_len\n",
            encoding="utf-8",
        )
    with csv_log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{step_idx},{total_steps},{loss:.6f},{reward_avg:.6f},{avg_response_len:.6f}\n"
        )
