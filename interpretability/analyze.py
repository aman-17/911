"""
Pre-compute top activating token examples per SAE feature.
Run this once after SAE training — produces feature_analysis.json used by the web app.
"""
import heapq
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

from interpretability.models.olmo2_1b import load_model
from interpretability.nn.activations import ActivationCollector
from interpretability.nn.sae import SAEConfig, SparseAutoencoder

N_EXAMPLES = 2000
TOP_K = 10
LAYER_IDX = 8
CHECKPOINT = "olmo2_1b_sae_layer8.pt"
OUTPUT = "feature_analysis.json"
CONTEXT_WINDOW = 8  # tokens of context around each activating token


def run_analysis() -> None:
    model, tokenizer = load_model()
    model.eval()

    sae = SparseAutoencoder(SAEConfig()).to(model.device)
    sae.load_state_dict(torch.load(CHECKPOINT, map_location=model.device, weights_only=True))
    sae.eval()

    dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")

    # top_examples[feature_idx] = min-heap of (activation_value, token_str, context_str)
    dict_size = SAEConfig().dict_size
    top_examples: list[list] = [[] for _ in range(dict_size)]

    with ActivationCollector(model.model.layers[LAYER_IDX]) as collector:
        with torch.no_grad():
            for i, example in enumerate(tqdm(dataset, total=N_EXAMPLES, desc="Analyzing")):
                if i >= N_EXAMPLES:
                    break

                text = "\n".join(f"{m['role']}: {m['content']}" for m in example["conversation"])
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
                token_ids = inputs["input_ids"][0].tolist()

                model(**inputs, output_hidden_states=False)

                hidden = collector.pop()[0].squeeze(0).float().to(model.device)  # [seq_len, d_model]
                features = sae(hidden).features  # [seq_len, dict_size]

                for tok_idx, feat_row in enumerate(features):
                    active_indices = feat_row.nonzero(as_tuple=False).squeeze(-1)
                    if active_indices.numel() == 0:
                        continue

                    ctx_start = max(0, tok_idx - CONTEXT_WINDOW)
                    ctx_end = min(len(token_ids), tok_idx + CONTEXT_WINDOW + 1)
                    context = tokenizer.decode(token_ids[ctx_start:ctx_end], skip_special_tokens=True)
                    token_str = tokenizer.decode([token_ids[tok_idx]], skip_special_tokens=True)

                    for fidx in active_indices.tolist():
                        val = float(feat_row[fidx])
                        heap = top_examples[fidx]
                        entry = (val, token_str, context)
                        if len(heap) < TOP_K:
                            heapq.heappush(heap, entry)
                        elif val > heap[0][0]:
                            heapq.heapreplace(heap, entry)

    result = {
        str(fidx): [
            {"activation": round(val, 4), "token": tok, "context": ctx}
            for val, tok, ctx in sorted(heap, reverse=True)
        ]
        for fidx, heap in enumerate(top_examples)
        if heap
    }

    Path(OUTPUT).write_text(json.dumps(result))
    print(f"Saved {len(result):,} features → {OUTPUT}")


if __name__ == "__main__":
    run_analysis()
