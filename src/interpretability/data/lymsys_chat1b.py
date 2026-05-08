import torch
from datasets import load_dataset
from tqdm import tqdm

from interpretability.models.olmo2_1b import load_model
from interpretability.nn.activations import ActivationCollector

DATASET = "lmsys/lmsys-chat-1m"
LAYER_IDX = 8
BATCH_SIZE = 8
MAX_SEQ_LEN = 2048
TARGET_TOKENS = 50_000_000
TOKENS_PER_CHUNK = 200_000


def _flatten_conversation(conv: list[dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in conv)


def collect_activations() -> None:
    model, tokenizer = load_model()
    model.eval()

    dataset = load_dataset(DATASET, split="train")
    tokens_collected = 0
    chunk_idx = 0
    flat_buffer: list[torch.Tensor] = []

    with ActivationCollector(model.model.layers[LAYER_IDX]) as collector:
        with torch.no_grad():
            batch_texts: list[str] = []
            pbar = tqdm(total=TARGET_TOKENS, desc="Collecting activations", unit="tok", unit_scale=True)

            for example in dataset:
                batch_texts.append(_flatten_conversation(example["conversation"]))
                if len(batch_texts) < BATCH_SIZE:
                    continue

                inputs = tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=MAX_SEQ_LEN,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)
                batch_texts = []

                model(**inputs, output_hidden_states=False)

                # strip padding — only keep real token activations
                mask = inputs["attention_mask"].cpu().bool()
                for hidden, m in zip(collector.pop()[0].cpu(), mask):
                    flat_buffer.append(hidden[m])  # [n_real_tokens, d_model]
                    tokens_collected += int(m.sum())

                pbar.update(tokens_collected - pbar.n)

                if sum(len(t) for t in flat_buffer) >= TOKENS_PER_CHUNK:
                    torch.save(torch.cat(flat_buffer), f"activations_chunk_{chunk_idx:04d}.pt")
                    flat_buffer.clear()
                    chunk_idx += 1

                if tokens_collected >= TARGET_TOKENS:
                    break

            if flat_buffer:
                torch.save(torch.cat(flat_buffer), f"activations_chunk_{chunk_idx:04d}.pt")
                chunk_idx += 1

            pbar.close()

    print(f"Done. {tokens_collected:,} tokens saved across {chunk_idx} chunks.")


if __name__ == "__main__":
    collect_activations()
