import time
from pathlib import Path

import torch

from post_training.loss_functions.policy_gradient import compute_grpo_loss


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    current_step = 0
    if csv_log_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_log_path = f"train_rlvr_grpo_metrics_{timestamp}.csv"
    csv_log_path = Path(csv_log_path)
    try:
        for step in range(steps):
            optimizer.zero_grad()
            current_step = step + 1
            example = math_data[step % len(math_data)]
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
            stats["loss_tensor"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            reward_avg = torch.tensor(stats["rewards"]).mean().item()
            step_tokens = sum(
                sample["gen_len"] for sample in stats["samples"]
            )
            avg_response_len = (
                step_tokens / len(stats["samples"]) if stats["samples"] else 0.0
            )
            append_csv_metrics(
                csv_log_path, current_step, steps,
                stats["loss"], reward_avg, avg_response_len,
            )
            print(
                f"[Step {current_step}/{steps}] "
                f"loss={stats['loss']:.4f} "
                f"reward_avg={reward_avg:.3f} "
                f"avg_resp_len={avg_response_len:.1f}"
            )
            if checkpoint_every and current_step % checkpoint_every == 0:
                ckpt_path = save_checkpoint(
                    model=model,
                    checkpoint_dir=checkpoint_dir, step=current_step,
                )
                print(f"Saved checkpoint to {ckpt_path}")
    except KeyboardInterrupt:
        ckpt_path = save_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            step=max(1, current_step),
            suffix="interrupt",
        )
        print(f"\nKeyboardInterrupt. Saved checkpoint to {ckpt_path}")
        return model
    return model


def save_checkpoint(model, checkpoint_dir, step, suffix=""):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-{suffix}" if suffix else ""
    ckpt_path = (
        checkpoint_dir /
        f"qwen3-0.6B-rlvr-grpo-step{step:05d}{suffix}.pth"
    )
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path

def append_csv_metrics(
    csv_log_path, step_idx, total_steps,
    loss, reward_avg, avg_response_len,
):
    if not csv_log_path.exists():
        csv_log_path.write_text(
            "step,total_steps,loss,reward_avg,avg_response_len\n",
            encoding="utf-8",
        )
    with csv_log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{step_idx},{total_steps},{loss:.6f},{reward_avg:.6f},"
            f"{avg_response_len:.6f}\n"
        )


def main():
    from post_training.data.data_tokenizer import load_model_and_tokenizer
    from post_training.data.web_crawling.datasets_from_hf import load_math_train

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(device=device)
    model.to(device)
    math_train = load_math_train()

    torch.manual_seed(0)
    train_rlvr_grpo(
        model=model,
        tokenizer=tokenizer,
        math_data=math_train,
        device=device,
        steps=50,
        num_rollouts=4,
        max_new_tokens=512,
        temperature=0.8,
        top_p=0.9,
        lr=1e-5,
        checkpoint_every=5,
        checkpoint_dir=".",
    )


if __name__ == "__main__":
    main()
