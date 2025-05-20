import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset
import os
import math
import random
from typing import List, Dict, Iterator, Optional
import numpy as np


class DatasetTargaV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


class IterableDatasetTargaV1(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        # data_files: Optional[List[str]],
        tokenized_data: List[str],
        tokenizer,
        context_length: int = 2048,
        shuffle: bool = True,
        shuffle_buffer_size: int = 1000,
        return_tensors: bool = True
    ):
        super(IterableDatasetTargaV1, self).__init__()
        # self.data_files = data_files
        if not isinstance(tokenized_data, list):
            tokenized_data = [tokenized_data]
        self.tokenized_data = []
        for item in tokenized_data:
            if isinstance(item, (list, np.ndarray)):
                self.tokenized_data.append(np.array(item, dtype=np.int32))
            else:
                self.tokenized_data.append(np.array(tokenizer.encode(item), dtype=np.int32))

        self.tokenizer = tokenizer
        self.context_length = context_length
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.return_tensors = return_tensors
        # if data_files:
        #     self.total_size = sum(os.path.getsize(f) for f in data_files)
        # else:
        self.total_size = sum(len(tokens) * 4 for tokens in self.tokenized_data)
        print(f"IterableDatasetTargaV1 initialized with {len(self.tokenized_data)} sequences")
        for i, seq in enumerate(self.tokenized_data):
            print(f"Sequence {i} type: {type(seq)}, length: {len(seq)}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            data_to_process = self.tokenized_data
        else:
            per_worker = int(math.ceil(len(self.tokenized_data) / float(worker_info.num_workers))) # Multi-process data loading-split files among workers
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, len(self.tokenized_data))
            data_to_process = self.tokenized_data[start_idx:end_idx]
        
        buffer = []
        for token_seq in data_to_process:
            # tokens = self._load_tokens(file_path)
            if not isinstance(token_seq, np.ndarray):
                token_seq = np.array(token_seq, dtype=np.int32)
                print(f"Token sequence type: {type(token_seq)}, length: {len(token_seq)}")
            
                if len(token_seq) <= self.context_length:
                    print(f"Warning: Token sequence length {len(token_seq)} is less than context length {self.context_length}")
                    continue

                for i in range(0, len(token_seq) - self.context_length, self.context_length):
                    chunk = token_seq[i:i + self.context_length]
                    if len(chunk) == self.context_length:
                        if self.shuffle:
                            buffer.append(chunk)
                            if len(buffer) >= self.shuffle_buffer_size:
                                random.shuffle(buffer)
                                for item in buffer[:self.shuffle_buffer_size//2]:  # Yield half the buffer
                                    yield self._create_sample(item)
                                buffer = buffer[self.shuffle_buffer_size//2:] # Keep remaining items
                        else:
                            yield self._create_sample(chunk) # Yield directly without shuffling
            
        # Yield any remaining items in buffer
        if buffer and self.shuffle:
            random.shuffle(buffer)
            for item in buffer:
                yield self._create_sample(item)
    
    # def _load_tokens(self, file_path: str) -> np.ndarray:
    #     try:
    #         return np.load(file_path)
    #     except:
    #         with open(file_path, 'r', encoding='utf-8') as f:
    #             text = f.read()
    #         return np.array(self.tokenizer.encode(text))
    
    def _create_sample(self, tokens) -> Dict[str, torch.Tensor]:
        x = tokens[:-1]  # Input: all but last token
        y = tokens[1:]   # Target: all but first token
        
        if self.return_tensors:
            x = torch.tensor(x, dtype=torch.long)
            y = torch.tensor(y, dtype=torch.long)
            
        return {'input_ids': x, 'labels': y}
