import math
import random
from typing import Iterator, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset


class DatasetTargaV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


class IterableDatasetTargaV1(torch.utils.data.IterableDataset):
    def __init__(
        self,
        tokenized_data: Union[List[str], List[List[int]], List[np.ndarray]],
        tokenizer,
        context_length: int = 2048,
        stride: Optional[int] = None,
        shuffle: bool = True,
        shuffle_buffer_size: int = 1000,
        return_tensors: bool = True,
        distributed: bool = False,
    ):
        super(IterableDatasetTargaV1, self).__init__()

        if not isinstance(tokenized_data, list):
            tokenized_data = [tokenized_data]

        self.tokenized_data = []
        for item in tokenized_data:
            if isinstance(item, (list, np.ndarray)):
                self.tokenized_data.append(np.array(item, dtype=np.int32))
            else:
                if tokenizer is None:
                    raise ValueError("Tokenizer required for text data")
                token_ids = tokenizer.encode(item)
                self.tokenized_data.append(np.array(token_ids, dtype=np.int32))

        self.tokenizer = tokenizer
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.return_tensors = return_tensors
        self.distributed = distributed

        self.total_tokens = sum(len(tokens) for tokens in self.tokenized_data)
        self._calculate_length()
        self.rank = 0
        self.world_size = 1
        if distributed and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def _calculate_length(self):
        total_samples = 0
        for token_seq in self.tokenized_data:
            if len(token_seq) > self.context_length:
                num_samples = (len(token_seq) - self.context_length + self.stride - 1) // self.stride
                total_samples += num_samples
        self._length = total_samples

    def __len__(self):
        if self.distributed:
            samples_per_rank = self._length // self.world_size
            if self.rank < self._length % self.world_size:
                samples_per_rank += 1
            return samples_per_rank
        return self._length

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterator:
        worker_info = torch.utils.data.get_worker_info()
        if self.distributed and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            sequences_per_rank = len(self.tokenized_data) // world_size
            start_seq = rank * sequences_per_rank
            end_seq = start_seq + sequences_per_rank
            if rank == world_size - 1:
                end_seq = len(self.tokenized_data)

            data_to_process = self.tokenized_data[start_seq:end_seq]
        else:
            data_to_process = self.tokenized_data

        if worker_info is not None:
            per_worker = int(math.ceil(len(data_to_process) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, len(data_to_process))
            data_to_process = data_to_process[start_idx:end_idx]

        if self.shuffle and self.distributed and hasattr(self, "epoch"):
            random.seed(42 + self.epoch + (self.rank if self.distributed else 0))

        buffer = []

        for token_seq in data_to_process:
            if len(token_seq) <= self.context_length:
                continue

            for i in range(0, len(token_seq) - self.context_length, self.stride):
                chunk = token_seq[i : i + self.context_length + 1]
                if len(chunk) == self.context_length + 1:
                    if self.shuffle:
                        buffer.append(chunk)
                        if len(buffer) >= self.shuffle_buffer_size:
                            random.shuffle(buffer)
                            for item in buffer[: self.shuffle_buffer_size // 2]:
                                yield self._create_sample(item)
                            buffer = buffer[self.shuffle_buffer_size // 2 :]
                    else:
                        yield self._create_sample(chunk)

        if buffer and self.shuffle:
            random.shuffle(buffer)
            for item in buffer:
                yield self._create_sample(item)

    def _create_sample(self, tokens):
        x = tokens[:-1]
        y = tokens[1:]
        if self.return_tensors:
            x = torch.tensor(x, dtype=torch.long)
            y = torch.tensor(y, dtype=torch.long)
        return x, y
