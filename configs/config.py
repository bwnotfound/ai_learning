import os

import torch

class LLMTransformerConfig:
    seed = 1234
    epochs = 1000
    batch_size = 2
    save_per_steps = 100
    precision = "32"
    num_workers = 4
    n_layers = 3
    n_heads = 8
    embedding_dim = 512
    hidden_dim = 512
    dropout = 0.0
    max_len = 512
    lr = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "./output"

    sys_prompt_start = "<|sys_prompt_start|>"
    sys_prompt_end = "<|sys_prompt_end|>"
    mem_start = "<|mem_start|>"
    mem_end = "<|mem_end|>"
    question_start = "<|question_start|>"
    question_end = "<|question_end|>"
    answer_start = "<|answer_start|>"
    answer_end = "<|answer_end|>"

    def __init__(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)