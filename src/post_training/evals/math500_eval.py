import json
import time
from pathlib import Path

import torch

from post_training.inference.generation import render_prompt
from post_training.data.data_utils import extract_final_candidate, grade_answer


def eta_progress_message(  # A
    processed,
    total,
    start_time,
    show_eta=False,
    label="Progress",
):
    progress = f"{label}: {processed}/{total}"
    pad_width = len(f"{label}: {total}/{total} | ETA: 00h 00m 00s")
    if not show_eta or processed <= 0:
        return progress.ljust(pad_width)
    elapsed = time.time() - start_time
    if elapsed <= 0:
        return progress.ljust(pad_width)
    remaining = max(total - processed, 0)
    if processed:
        avg_time = elapsed / processed
        eta_seconds = avg_time * remaining
    else:
        eta_seconds = 0
    eta_seconds = max(int(round(eta_seconds)), 0)
    minutes, rem_seconds = divmod(eta_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        eta = f"{hours}h {minutes:02d}m {rem_seconds:02d}s"
    elif minutes:
        eta = f"{minutes:02d}m {rem_seconds:02d}s"
    else:
        eta = f"{rem_seconds:02d}s"
    message = f"{progress} | ETA: {eta}"
    return message.ljust(pad_width)


@torch.no_grad()
def generate_text_stream_concat(
    model, tokenizer, prompt, device, max_new_tokens=512, verbose=False
):
    """Greedy-decode from ``prompt`` and return the concatenated generated text.

    Greedy (argmax) decoding is used for deterministic, reproducible eval.
    When ``verbose`` is set, tokens are streamed to stdout as they are produced.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model(input_ids, use_cache=True)
    logits = out.logits[:, -1, :]
    past_key_values = out.past_key_values

    pieces = []
    for _ in range(max_new_tokens):
        token_id = int(torch.argmax(logits[0]).item())
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break
        piece = tokenizer.decode([token_id], skip_special_tokens=True)
        pieces.append(piece)
        if verbose:
            print(piece, end="", flush=True)
        next_token = torch.tensor([[token_id]], device=device)
        out = model(next_token, past_key_values=past_key_values, use_cache=True)
        logits = out.logits[:, -1, :]
        past_key_values = out.past_key_values

    return "".join(pieces)


def evaluate_math500_stream(
    model,
    tokenizer,
    device,
    math_data,
    out_path=None,
    max_new_tokens=512,
    verbose=False,
):
    if out_path is None:
        dev_name = str(device).replace(":", "-")  # B
        out_path = Path(f"math500-{dev_name}.jsonl")
    num_examples = len(math_data)
    num_correct = 0
    start_time = time.time()
    with open(out_path, "w", encoding="utf-8") as f:  # C
        for i, row in enumerate(math_data, start=1):
            prompt = render_prompt(row["problem"])  # D
            gen_text = generate_text_stream_concat(  # E
                model, tokenizer, prompt, device,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )
            extracted = extract_final_candidate(  # F
                gen_text
            )
            is_correct = grade_answer(  # G
                extracted, row["answer"]
            )
            num_correct += int(is_correct)
            record = {  # H
                "index": i,
                "problem": row["problem"],
                "gtruth_answer": row["answer"],
                "generated_text": gen_text,
                "extracted": extracted,
                "correct": bool(is_correct),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress_msg = eta_progress_message(
                processed=i,
                total=num_examples,
                start_time=start_time,
                show_eta=True,
                label="MATH-500",
            )
            print(progress_msg, end="\r", flush=True)
            if verbose:  # I
                print(
                    f"\n\n{'='*50}\n{progress_msg}\n"
                    f"{'='*50}\nExtracted: {extracted}\n"
                    f"Expected: {row['answer']}\n"
                    f"Correct so far: {num_correct}\n{'-'*50}"
                )
    seconds_elapsed = time.time() - start_time
    acc = num_correct / num_examples if num_examples else 0.0
    print(f"\nAccuracy: {acc*100:.1f}% ({num_correct}/{num_examples})")
    print(f"Total time: {seconds_elapsed/60:.1f} min")
    print(f"Logs written to: {out_path}")
    return num_correct, num_examples, acc


if __name__ == "__main__":
    from post_training.data.data_tokenizer import load_model_and_tokenizer
    from post_training.data.web_crawling.datasets_from_hf import load_math_train

    WHICH_MODEL = "base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(device=device)

    # NOTE: swap in the MATH-500 test split here; the train set (same
    # {"problem", "answer"} schema) is used as a runnable stand-in.
    math_data = load_math_train()

    print("Model:", WHICH_MODEL)
    num_correct, num_examples, acc = evaluate_math500_stream(
        model, tokenizer, device,
        math_data=math_data[:10],  # A
        max_new_tokens=2048,
        verbose=False,  # B
    )

    # To evaluate the RLVR-finetuned checkpoint instead, load it and re-run:
    # model.load_state_dict(torch.load("qwen3-0.6B-rlvr-grpo-step00050.pth"))
    # evaluate_math500_stream(model, tokenizer, device, math_data=math_data[:10])

# to run the evaluation, use:
# python evaluate_math500.py \
#   --dataset_size 500 \
#   --which_model base \
#   --checkpoint_path "qwen3-0.6B-rlvr-grpo-step00050.pth"
