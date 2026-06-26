import sys
from pathlib import Path
from pprint import pprint

import torch

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from post_training.inference.generation import render_prompt
from post_training.inference.rollout import sample_response
from post_training.inference.logprobs import sequence_logprob
from post_training.rewards.reward_funtions import reward_rlvr


def compute_grpo_loss(
    model,
    tokenizer,
    example,
    device,
    num_rollouts=2,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9,
):
    assert num_rollouts >= 2
    roll_logps, roll_rewards, samples = [], [], []
    prompt = render_prompt(example["problem"])
    was_training = model.training
    model.eval()
    for _ in range(num_rollouts):
        result = sample_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        token_ids = result["full_token_ids"]
        prompt_len = result["prompt_len"]
        text = result["text"]
        reward = reward_rlvr(text, example["answer"])
        logp = sequence_logprob(model, token_ids, prompt_len)
        roll_logps.append(logp)
        roll_rewards.append(reward)
        samples.append(
            {
                "text": text,
                "reward": reward,
                "gen_len": token_ids.numel() - prompt_len,
            }
        )

    if was_training:
        model.train()
    rewards = torch.tensor(roll_rewards, device=device)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    logps = torch.stack(roll_logps)
    pg_loss = -(advantages.detach() * logps).mean()
    loss = pg_loss
    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "rewards": roll_rewards,
        "advantages": advantages.detach().cpu().tolist(),
        "samples": samples,
        "loss_tensor": loss,
    }


if __name__ == "__main__":
    from post_training.data.data_tokenizer import load_model_and_tokenizer
    from post_training.data.web_crawling.datasets_from_hf import load_math_train

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(device=device)
    math_train = load_math_train()

    torch.manual_seed(123)
    stats = compute_grpo_loss(
        model=model,
        tokenizer=tokenizer,
        example=math_train[4],
        device=device,
        num_rollouts=2,
        max_new_tokens=256,
        temperature=0.8,
        top_p=0.9,
    )
    pprint(stats)
