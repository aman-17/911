import os
import argparse
import multiprocessing as mp
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def get_dtype_for_vocab_size(vocab_size):
    if vocab_size < 2**8:
        return np.uint8
    elif vocab_size < 2**16:
        return np.uint16
    elif vocab_size < 2**32:
        return np.uint32
    else:
        return np.uint64

def process_dataset(dataset, tokenizer, local_dir="processed_data", shard_size=int(1e8)):
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    vocab_size = tokenizer.vocab_size
    dtype = get_dtype_for_vocab_size(vocab_size)
    max_val = np.iinfo(dtype).max
    
    print(f"Vocab size: {vocab_size}")
    print(f"Using dtype: {dtype}")
    print(f"Max value for dtype: {max_val}")
    
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        eot = tokenizer.eos_token_id
    elif hasattr(tokenizer, 'sep_token_id') and tokenizer.sep_token_id is not None:
        eot = tokenizer.sep_token_id
    else:
        eot = 0
        print(f"Warning: Could not find end-of-text token, using {eot}")
    
    def tokenize(doc):
        tokens = [eot]
        encoded = tokenizer.encode(doc["text"], add_special_tokens=False)
        tokens.extend(encoded)
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np <= max_val).all(), \
            f"Token values exceed {dtype} range. Max token: {tokens_np.max()}, Max allowed: {max_val}"
        
        tokens_np_typed = tokens_np.astype(dtype)
        return tokens_np_typed
    
    def write_datafile(filename, tokens_np):
        np.save(filename, tokens_np)
    
    nprocs = max(1, os.cpu_count()//2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=dtype)
        token_count = 0
        progress_bar = None
        
        for tokens in pool.imap(tokenize, dataset, chunksize=16):
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"processed_{split}_{shard_index:06d}")
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder
        
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"processed_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])

def main():
    parser = argparse.ArgumentParser(description="Process dataset with tokenizer and save as .npy files")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., 'HuggingFaceFW/fineweb-edu')")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Tokenizer name (e.g., 'gpt2', 'bert-base-uncased')")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset configuration name (e.g., 'sample-10BT')")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to use (default: 'train')")
    parser.add_argument("--output_dir", type=str, default="processed_data",
                        help="Output directory for processed files (default: 'processed_data')")
    parser.add_argument("--shard_size", type=int, default=int(1e8),
                        help="Number of tokens per shard (default: 100M)")
    parser.add_argument("--num_proc", type=int, default=None,
                        help="Number of processes to use (default: half of CPU cores)")
    
    args = parser.parse_args()
    
    print(f"Loading dataset: {args.dataset}")
    if args.dataset_config:
        dataset = load_dataset(args.dataset, name=args.dataset_config, split=args.split)
        print(f"Dataset config: {args.dataset_config}")
    else:
        dataset = load_dataset(args.dataset, split=args.split)
    
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Shard size: {args.shard_size:,} tokens")
    if args.num_proc:
        original_cpu_count = os.cpu_count
        os.cpu_count = lambda: args.num_proc * 2
    
    process_dataset(dataset, tokenizer, args.output_dir, args.shard_size)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()