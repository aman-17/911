import os
import numpy as np
from torch.utils.data import DataLoader
import torch.distributed as dist
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


def create_train_loader(cfg, distributed=True):
    data_path = cfg["train_data"]
    batch_size = cfg.get("batch_size", 4)
    max_length = cfg["max_seq_length"]
    stride = cfg.get("stride", max_length // 2)
    num_workers = cfg.get("num_workers", 0)
    pin_memory = cfg.get("pin_memory", True)
    shuffle_buffer_size = cfg.get("shuffle_buffer_size", 500)

    tokenizer = tiktoken.get_encoding("gpt2")
    rank = 0
    world_size = 1
    if distributed and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    
    if os.path.isdir(data_path):
        files = os.listdir(data_path)
        npy_files = [f for f in files if f.endswith('.npy')]
        txt_files = [f for f in files if f.endswith('.txt')]
        
        if npy_files:
            file_paths = load_npy_files_lazy(data_path, split="train")
            file_paths = file_paths[:min(10, len(file_paths))]
            if distributed and world_size > 1:
                files_per_rank = len(file_paths) // world_size
                start_idx = rank * files_per_rank
                end_idx = start_idx + files_per_rank
                if rank == world_size - 1:
                    end_idx = len(file_paths)
                file_paths = file_paths[start_idx:end_idx]
                
                if rank == 0:
                    print(f"Distributed mode: Total files: {len(file_paths) * world_size}, "
                          f"Files per rank: {len(file_paths)}")
            
            tokenized_data = list(load_npy_data_generator(file_paths))
            
            dataset = IterableDatasetTargaV1(
                tokenized_data=tokenized_data,
                tokenizer=None,
                context_length=max_length,
                stride=stride,
                shuffle=True,
                shuffle_buffer_size=shuffle_buffer_size,
                return_tensors=True,
                distributed=distributed
            )
            
        elif txt_files:
            if distributed and world_size > 1:
                files_per_rank = len(txt_files) // world_size
                start_idx = rank * files_per_rank
                end_idx = start_idx + files_per_rank
                if rank == world_size - 1:
                    end_idx = len(txt_files)
                txt_files = txt_files[start_idx:end_idx]
            
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
                shuffle_buffer_size=shuffle_buffer_size,
                return_tensors=True,
                distributed=distributed
            )
        else:
            raise ValueError(f"No .npy or .txt files found in directory: {data_path}")
            
    elif os.path.isfile(data_path):
        if data_path.endswith('.npy'):
            data = np.load(data_path)
            if data.dtype == np.uint16:
                data = data.astype(np.int32)
            if distributed and world_size > 1:
                data_len = len(data)
                chunk_size = data_len // world_size
                start_idx = rank * chunk_size
                end_idx = start_idx + chunk_size
                if rank == world_size - 1:
                    end_idx = data_len
                data = data[start_idx:end_idx]
                
                if rank == 0:
                    print(f"Distributed mode: Total tokens: {data_len}, "
                          f"Tokens per rank: ~{chunk_size}")
            
            tokenized_data = [data]
            
            dataset = IterableDatasetTargaV1(
                tokenized_data=tokenized_data,
                tokenizer=None,
                context_length=max_length,
                stride=stride,
                shuffle=True,
                shuffle_buffer_size=shuffle_buffer_size,
                return_tensors=True,
                distributed=distributed
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
                shuffle_buffer_size=shuffle_buffer_size,
                return_tensors=True,
                distributed=distributed
            )
    else:
        raise ValueError(f"Invalid data path: {data_path}")

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
        sampler=None
    )
    # if rank == 0:
    #     print(f"Created DataLoader with batch_size={batch_size}, "
    #           f"num_workers={num_workers}, distributed={distributed}")
    #     if hasattr(dataset, '__len__'):
    #         try:
    #             dataset_len = len(dataset)
    #             # print(f"Dataset length: {dataset_len} samples per rank")
    #             # print(f"Steps per epoch: {dataset_len // batch_size}")
    #         except:
    #             raise Exception("IterableDataset len problem")

    return train_loader, tokenizer