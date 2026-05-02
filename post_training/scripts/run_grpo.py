import argparse
import torch

from post_training.data.data_tokenizer import load_model_and_tokenizer
from post_training.data.web_crawling.datasets_from_hf import load_math_train
from post_training.training.grpo import train_rlvr_grpo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--num_rollouts", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--checkpoint_every", type=int, default=50)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--data_path", type=str, default="math_train.json")
    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(device=device)

    print("Loading data...")
    math_data = load_math_train(local_path=args.data_path)
    math_data = math_data[: args.max_samples]
    print(f"Loaded {len(math_data)} examples")

    train_rlvr_grpo(
        model=model,
        tokenizer=tokenizer,
        math_data=math_data,
        device=device,
        steps=args.steps,
        num_rollouts=args.num_rollouts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        lr=args.lr,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
