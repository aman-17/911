import torch

from post_training.data.data_utils import (
    extract_final_candidate, grade_answer
)


def reward_rlvr(answer_text, ground_truth):
    extracted = extract_final_candidate(
        answer_text, fallback=None  # A
    )
    if not extracted:
        return 0.0
    correct = grade_answer(extracted, ground_truth)
    return float(correct)


def advantage_rlvr(answer_text, ground_truth):
    extracted = extract_final_candidate(
        answer_text, fallback=None  # A
    )
    if not extracted:
        return 0.0
    correct = grade_answer(extracted, ground_truth)
    return float(correct) - 0.5  # B


if __name__ == "__main__":
    rollouts = [
        r"Summing the digits gives \boxed{83}.",
        r"After simplifying we obtain \boxed{82}.",
        "I'm fairly sure the answer is 83.",
        r"So the final total is \boxed{83} units.",
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rollout_rewards = []
    for answer in rollouts:
        reward = reward_rlvr(answer_text=answer, ground_truth="83")
        print(f"Answer: {answer!r}")
        print(f"Reward: {reward}\n")
        rollout_rewards.append(reward)

    rewards = torch.tensor(rollout_rewards, device=device)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    print(rewards)
    print(advantages)
