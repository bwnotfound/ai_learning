from functools import partial
import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import tiktoken
from datasets import load_dataset, load_from_disk

from configs.config import LLMTransformerConfig


class TranslateDataset(Dataset):
    def __init__(self, encoder: tiktoken.Encoding, data_path="./datasets/fra.txt"):
        self.dataset = pd.read_csv(data_path, delimiter='\t', header=None)
        self.dataset = self.dataset.iloc[:, :2]
        self.encoder = encoder

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.iloc[idx, 0], self.dataset.iloc[idx, 1]

    def collate_fn(self, batchs, len_align=True):
        src = []
        tgt = []
        for src_, tgt_ in batchs:
            src.append(self.encoder.encode_ordinary(src_))
            tgt.append(self.encoder.encode_ordinary(tgt_))
        if len_align:
            max_len1 = max([len(s) for s in src])
            max_len2 = max([len(t) for t in tgt])
            max_len = max(max_len1, max_len2)
        src = [s + [self.encoder.eot_token] * (max_len - len(s)) for s in src]
        src = torch.tensor(src)
        if not len_align:
            max_len = max([len(t) for t in tgt])
        tgt = [t + [self.encoder.eot_token] * (max_len - len(t)) for t in tgt]
        tgt = torch.tensor(tgt)
        return src, tgt


class OpenOrcaDataset(Dataset):
    def __init__(
        self,
        config: LLMTransformerConfig,
        enc: tiktoken.Encoding,
        is_val=False,
        cache_dir="D:/AI/HuggingFace/datasets",
    ):
        super().__init__()
        self.config = config
        self.eot_token = enc.eot_token
        self.dataset = load_dataset("Open-Orca/OpenOrca", cache_dir=cache_dir)
        self.split_dataset = self.dataset["train"].train_test_split(
            test_size=0.0005, seed=2357, shuffle=True
        )
        if is_val:
            self.dataset = self.split_dataset["test"]
        else:
            self.dataset = self.split_dataset["train"]
        self.dataset = self.dataset.to_list()
        self.dataset = map(partial(self.handle, enc=enc), self.dataset)

    def handle(self, x, enc: tiktoken.Encoding):
        ids = []
        ids.append(enc._special_tokens[self.config.sys_prompt_start])
        ids.extend(enc.encode_ordinary(x['system_prompt']))
        ids.append(enc._special_tokens[self.config.sys_prompt_end])
        ids.append(enc._special_tokens[self.config.mem_start])
        ids.append(enc._special_tokens[self.config.mem_end])
        ids.append(enc._special_tokens[self.config.question_start])
        ids.extend(enc.encode_ordinary(x['question']))
        ids.append(enc._special_tokens[self.config.question_end])
        answer_start_idx = len(ids)
        ids.append(enc._special_tokens[self.config.answer_start])
        ids.extend(enc.encode_ordinary(x['response']))
        ids.append(enc._special_tokens[self.config.answer_end])
        ids = ids[: self.config.max_len]
        answer_start_idx = min(answer_start_idx, self.config.max_len)
        return ids, answer_start_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]

    def collate_fn(self, batchs):
        ids_list = []
        len_list = []
        answer_start_idx_list = []
        for b in batchs:
            ids, answer_start_idx = b
            ids_list.append(ids)
            len_list.append(len(ids))
            answer_start_idx_list.append(answer_start_idx)

        max_len = max(len_list)
        ids_tensor = torch.zeros(len(batchs), max_len, dtype=torch.long)
        ids_tensor = torch.fill_(ids_tensor, self.eot_token)
        for i, ids in enumerate(ids_list):
            ids_tensor[i, : len(ids)] = torch.tensor(ids)

        len_tensor = torch.tensor(len_list)
        answer_start_idx_tensor = torch.tensor(answer_start_idx_list)

        return ids_tensor, len_tensor, answer_start_idx_tensor


class BookCorpus(Dataset):
    def __init__(
        self,
        enc: tiktoken.Encoding,
        is_val=False,
        max_len=None,
        split=False,
        val_rate=0.005,
        seed=1437,
        cache_dir="D:/AI/HuggingFace/datasets",
    ):
        super().__init__()
        self.eot_token = enc.eot_token
        self.max_len = max_len
        self.dataset = load_dataset("bookcorpus", cache_dir=cache_dir)
        self.split_dataset = self.dataset["train"].train_test_split(
            test_size=val_rate, seed=seed, shuffle=True
        )
        if is_val:
            self.dataset = self.split_dataset["test"]
        else:
            self.dataset = self.split_dataset["train"]
        self.dataset = self.dataset.to_list()
        self.dataset = list(map(lambda x: enc.encode_ordinary(x['text']), self.dataset))
        if max_len is not None:
            if not split:
                self.dataset = list(
                    filter(lambda x: len(x) <= max_len - 1, self.dataset)
                )
            else:
                new_dataset = []
                for d in self.dataset:
                    for i in range(0, len(d), max_len - 1):
                        new_dataset.append(d[i : i + max_len - 1])
                self.dataset = new_dataset
        self.dataset = [d for d in new_dataset if len(d) > 0]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]

    def collate_fn(self, batchs):
        ids_list = []
        len_list = []
        for b in batchs:
            ids = b + [self.eot_token]
            ids_list.append(ids)
            len_list.append(len(ids))

        max_len = max(len_list)
        ids_tensor = torch.zeros(len(batchs), max_len, dtype=torch.long)
        ids_tensor = torch.fill_(ids_tensor, self.eot_token)
        for i, ids in enumerate(ids_list):
            ids_tensor[i, : len(ids)] = torch.tensor(ids)

        len_tensor = torch.tensor(len_list)

        return ids_tensor, len_tensor


class ImageNetSmall(Dataset):

    def __init__(
        self,
        is_val=False,
        val_rate=0.005,
        seed=1437,
        cache_dir="D:/AI/HuggingFace/datasets",
        transforms=None,
        need_transform=True,
        disk_dir=None,
    ):
        if disk_dir is not None:
            disk_path = os.path.join(disk_dir, "image_net_small_dataset")
            if not os.path.exists(disk_dir):
                os.makedirs(disk_dir)
            if os.path.exists(disk_path):
                self.dataset = load_from_disk(disk_path, keep_in_memory=True).with_format("torch")
                return
        self.dataset = load_dataset("israfelsr/mm_tiny_imagenet", cache_dir=cache_dir)
        self.split_dataset = self.dataset["train"].train_test_split(
            test_size=val_rate, seed=seed, shuffle=True
        )
        if is_val:
            self.dataset = self.split_dataset["test"]
        else:
            self.dataset = self.split_dataset["train"]
        if need_transform:
            if transforms is None:
                from torchvision.transforms import (
                    Compose,
                    ToTensor,
                    Resize,
                    Normalize,
                    ColorJitter,
                )

                transforms = Compose(
                    [
                        Resize((224, 224)),
                        ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
                        ),
                        ToTensor(),
                        # Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ]
                )

            def _transforms(examples):
                examples["image"] = [
                    transforms(image.convert("RGB")) for image in examples["image"]
                ]
                return examples

            if disk_dir is None:
                self.dataset.set_transform(_transforms)
            else:
                self.dataset = self.dataset.map(
                    _transforms, batched=True, batch_size=64
                )
        if disk_dir is not None:
            self.dataset.save_to_disk(disk_path)
            self.dataset = load_from_disk(disk_path).with_format("torch")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]

    def collate_fn(self, batchs):
        images = []
        labels = []
        for b in batchs:
            images.append(b["image"])
            labels.append(b["label"])
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        return images, labels
