from typing import Tuple

import tiktoken
from torch.utils.data import DataLoader

from data.data_loader import DatasetTargaV1, IterableDatasetTargaV1
from data.data_tokenizer import TokenizerV0, load_txt_file


def create_dataloader_v0(
    txt, batch_size=4, max_length=2048, stride=128, shuffle=True, drop_last=True
) -> DataLoader:
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = DatasetTargaV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last)
    return dataloader


def create_dataloader_v1(
    txt, batch_size=4, max_length=2048, stride=128, shuffle=True, drop_last=True
) -> DataLoader:
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = IterableDatasetTargaV1(
        tokenized_data=txt,
        tokenizer=tokenizer,
        context_length=max_length,
        shuffle=shuffle,
        shuffle_buffer_size=1000,
        return_tensors=True,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last)
    return dataloader


def create_train_loader(train_config) -> Tuple[DataLoader, TokenizerV0]:
    with open(train_config["train_data"], "r", encoding="utf-8") as f:
        raw_text = f.read()

    vocab = load_txt_file(raw_text)
    tokenizer = TokenizerV0(vocab)

    train_ratio = 0.90
    split_idx = int(train_ratio * len(raw_text))
    train_data = raw_text[:split_idx]
    # torch.manual_seed(123)
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=train_config["max_seq_length"],
        stride=train_config["max_seq_length"],
        drop_last=True,
        shuffle=True,
    )
    return train_loader, tokenizer
