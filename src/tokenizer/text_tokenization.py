
from pathlib import Path

import torch
from tiktoken import Encoding, get_encoding
from torch.utils.data import DataLoader, Dataset

file_path = Path(__file__).with_name("the-verdict.txt")

with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

#now let's create the input-target pairs for training for the next word prediction task
""" for i in range(1, context_size + 1):
    context = encoded_text[:i]
    target = encoded_text[i]
    print(context, "--->", target)

    print(tokenizer.decode(context), "--->", tokenizer.decode([target])) """

class GPTDatasetV1(Dataset):
    def __init__(self: GPTDatasetV1, text: str, tokenizer: Encoding, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self: GPTDatasetV1) -> int:
        return len(self.input_ids)

    def __getitem__(self: GPTDatasetV1, idx: int):
        return self.input_ids[idx], self.target_ids[idx]



def create_dataloader_v1(
    text: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
):
    tokenizer = get_encoding("cl100k_base")
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

# Lets test the dataloader with a batch size of 1 for an LLM context size pf 4 
# to develop an intuition of how the GPTDataseV1 works 

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=8, stride=8, shuffle=False)

data_iter = iter(dataloader)
first_batch = next(data_iter)
# print(first_batch)
"""
output = [tensor([[  40,  473, 1846, 2744]]), tensor([[ 473, 1846, 2744, 3463]])]
first_batch var contains 2 tensors: input_ids and target_ids
max_length is set to 4, so each input and target tensor contains 4 tokens
"""

"""
# lets understand the stride
output: 
[tensor([[ 473, 1846, 2744, 3463]]), tensor([[1846, 2744, 3463, 7762]])]
if we compare the input_ids of the first and second batch, we can see that the second batch starts one token after the first batch, which is consistent with stride=1.
so stride setting dictates the number of tokens to move forward for the next batch emulating a sliding window over the token sequence.

Best-practice choices by goal:
    stride=max_length: efficient default for large pretraining datasets.
    stride=max_length//2: useful for overlapping chunks or smaller datasets.
    stride=1: maximum examples, but usually impractical and very redundant.
    Your max_length=256, stride=128 is therefore a reasonable
"""
second_batch = next(data_iter)
# print(second_batch)