import os
import numpy as np
from torch.utils.data import DataLoader
import tiktoken
from data.data_loader import IterableDatasetTargaV1


def load_npy_files_lazy(data_dir, split="train"):
    npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy') and split in f]
    npy_files.sort()
    file_paths = [os.path.join(data_dir, f) for f in npy_files]
    return file_paths


def load_npy_data_generator(file_paths):
    for file_path in file_paths:
        print(f"Loading {file_path}")
        data = np.load(file_path)
        if data.dtype == np.uint16:
            data = data.astype(np.int32)
        yield data


def create_train_loader(cfg):
    data_path = cfg["train_data"]
    batch_size = cfg.get("batch_size", 4)
    max_length = cfg["max_seq_length"]
    stride = cfg.get("stride", max_length // 2)
    tokenizer = tiktoken.get_encoding("gpt2")
    
    if os.path.isdir(data_path):
        files = os.listdir(data_path)
        npy_files = [f for f in files if f.endswith('.npy')]
        txt_files = [f for f in files if f.endswith('.txt')]
        
        if npy_files:
            file_paths = load_npy_files_lazy(data_path, split="train")
            file_paths = file_paths[:min(10, len(file_paths))]
            
            tokenized_data = list(load_npy_data_generator(file_paths))
            
            dataset = IterableDatasetTargaV1(
                tokenized_data=tokenized_data,
                tokenizer=None,
                context_length=max_length,
                stride=stride,
                shuffle=True,
                shuffle_buffer_size=500,
                return_tensors=True
            )
            
        elif txt_files:
            all_text = ""
            for txt_file in txt_files:
                file_path = os.path.join(data_path, txt_file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                all_text += text + "\n"
            
            dataset = IterableDatasetTargaV1(
                tokenized_data=[all_text],
                tokenizer=tokenizer,
                context_length=max_length,
                stride=stride,
                shuffle=True,
                shuffle_buffer_size=500,
                return_tensors=True
            )
        else:
            raise ValueError(f"No .npy or .txt files found in directory: {data_path}")
    
    elif os.path.isfile(data_path):
        if data_path.endswith('.npy'):
            data = np.load(data_path)
            if data.dtype == np.uint16:
                data = data.astype(np.int32)
            tokenized_data = [data]
            
            dataset = IterableDatasetTargaV1(
                tokenized_data=tokenized_data,
                tokenizer=None,
                context_length=max_length,
                stride=stride,
                shuffle=True,
                shuffle_buffer_size=500,
                return_tensors=True
            )
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            dataset = IterableDatasetTargaV1(
                tokenized_data=[text],
                tokenizer=tokenizer,
                context_length=max_length,
                stride=stride,
                shuffle=True,
                shuffle_buffer_size=500,
                return_tensors=True
            )
    else:
        raise ValueError(f"Invalid data path: {data_path}")
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, tokenizer
