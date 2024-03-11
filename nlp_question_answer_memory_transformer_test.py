import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import tiktoken

from noobai.module.nn.resblock import SimpleResWrapper
from noobai.datasets.nlp.qa import OpenOrcaDataset
from noobai.util.trainer import SimpleTrainer
from noobai.module.nn.lr_scheduler import WarmupCosineLRSchedule
from noobai.module.nn.loss import MaskedCrossEntropyLoss

import torch
import torch.nn as nn
from typing import Iterable
from tqdm import tqdm

from noobai.data.util import save_model


class Trainer(SimpleTrainer):

    def train(
        self,
        step=0,
        epochs=100,
        save_iter=200,
        eval_iter=None,
        has_target=True,
        simple_accuracy=True,
        clip_grad=None,
        load_model=True,
    ):
        r'''
        accuracy_mode: None=no acc. ""
        '''
        if load_model:
            step = self.load_model()
        
        for epoch in range(epochs):
            t_bar = tqdm(
                total=len(self.dataloader),
                ncols=100,
                desc=f"Epoch {epoch}",
                colour="green",
            )
            for ids_tensor, len_tensor, answer_start_idx_tensor in self.dataloader:
                if self.scheduler is not None:
                    self.scheduler.step()
                ids_tensor, len_tensor, answer_start_idx_tensor = (
                    ids_tensor.to(self.device, non_blocking=True),
                    len_tensor.to(self.device, non_blocking=True),
                    answer_start_idx_tensor.to(self.device, non_blocking=True),
                )
                ids_tensor = ids_tensor.permute(1, 0)
                if torch.max(len_tensor - answer_start_idx_tensor).item() == 0:
                    continue
                self.optimizer.zero_grad()
                output, loss = self.model(
                    ids_tensor, len_tensor, answer_start_idx_tensor
                )
                loss.backward()
                if clip_grad is not None:
                    nn.utils.clip_grad_value_(self.model.parameters(), clip_grad)
                self.optimizer.step()

                msg = f"loss: {loss.item():.5f}"
                if self.scheduler is not None:
                    try:
                        lr = self.scheduler.get_lr()
                    except NotImplementedError:
                        lr = self.scheduler.get_last_lr()
                    if isinstance(lr, Iterable):
                        lr = lr[0]
                    msg += f", lr: {lr:.3e}"
                t_bar.set_postfix_str(msg)
                t_bar.update()
                step += 1

                if step % save_iter == 0:
                    save_model(
                        self.save_dir,
                        self.model,
                        self.optimizer,
                        step,
                        scheduler=self.scheduler,
                        model_name=self.model_name,
                    )
                if (
                    eval_iter is not None
                    and self.eval_func is not None
                    and step % eval_iter == 0
                ):
                    self.eval_func()
            t_bar.close()


class Model(nn.Module):
    def __init__(
        self,
        n_vocab,
        hidden_dim,
        n_layer,
        n_head,
        mem_size=32,
        dropout=0,
        forget_rate=0.2,
        loop_time=3,
        chunk_size=32,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.emb = nn.Embedding(n_vocab, hidden_dim)
        self.pos_emb = nn.Embedding(256, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, n_head, dropout=dropout), n_layer
        )
        self.hidden_dim = hidden_dim
        self.mem_size = mem_size
        self.loop_time = loop_time
        self.chunk_size = chunk_size
        self.forget_rate = forget_rate
        self.forget_param = nn.Parameter(torch.zeros(hidden_dim))
        self.init_param = nn.Parameter(torch.zeros(hidden_dim))
        self.start_param = nn.Parameter(torch.zeros(hidden_dim))
        self.empty_param = nn.Parameter(torch.zeros(hidden_dim))
        self.l_out = nn.Sequential(
            SimpleResWrapper(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
            ),
            nn.Linear(hidden_dim, n_vocab),
        )
        self.loss_func = MaskedCrossEntropyLoss()

    def forward(self, x, len_tensor, answer_start_idx_tensor):
        device, batch_size = x.device, x.size(1)

        target = []
        for i in range(len_tensor.size(0)):
            target.append(x[answer_start_idx_tensor[i] : len_tensor[i], i])
        target_len_tensor = len_tensor - answer_start_idx_tensor
        t_max_len = torch.max(target_len_tensor).item()
        if t_max_len % self.chunk_size != 0:
            t_max_len += self.chunk_size - t_max_len % self.chunk_size
        for i in range(len(target)):
            target[i] = torch.cat(
                [
                    target[i],
                    torch.zeros(
                        t_max_len - target[i].size(0), dtype=torch.long, device=device
                    ),
                ],
                dim=0,
            )
        target = torch.stack(target, dim=1)
        mem = (
            self.init_param.unsqueeze(0)
            .unsqueeze(0)
            .repeat(self.mem_size, batch_size, 1)
        )

        for i in range(self.loop_time):
            for j in range(0, x.size(0), self.chunk_size):
                chosen_indices = []
                for k in range(batch_size):
                    if j >= answer_start_idx_tensor[k]:
                        continue
                    chosen_indices.append(k)
                if len(chosen_indices) == 0:
                    break
                inp_x = self.emb(x[:, chosen_indices][j : j + self.chunk_size])
                inp_mem = mem[:, chosen_indices]
                inp = torch.cat([inp_mem, inp_x], dim=0)
                inp = inp + self.pos_emb(
                    torch.arange(inp.size(0), device=device)
                ).unsqueeze(1)
                output = self.transformer(inp)
                mem[:, chosen_indices] = output[: self.mem_size]
            if i < self.loop_time - 1:
                forget_num = int(self.mem_size * self.forget_rate)
                if forget_num == 0:
                    continue
                forget_idx_list = torch.arange(0, self.mem_size).numpy()
                np.random.shuffle(forget_idx_list)
                forget_idx_list = forget_idx_list[:forget_num]
                mem[forget_idx_list] = self.forget_param.unsqueeze(0).repeat(
                    batch_size, 1
                )
        chunked_target_list = [
            self.start_param.unsqueeze(0)
            .unsqueeze(0)
            .repeat(self.chunk_size, batch_size, 1)
        ]
        for i in range(0, target.size(0), self.chunk_size):
            chunked_target_list.append(target[i : i + self.chunk_size])
        chunked_target_list = chunked_target_list[:-1]
        out_list = []
        for i, chunked_target in enumerate(chunked_target_list):
            chosen_indices = []
            for j in range(batch_size):
                if i * self.chunk_size >= len_tensor[j] - answer_start_idx_tensor[j]:
                    continue
                chosen_indices.append(j)
            tem = torch.zeros(self.chunk_size, batch_size, self.n_vocab, device=device)
            if len(chosen_indices) == 0:
                out_list.append(tem)
                continue
            chunked_target = chunked_target[:, chosen_indices]
            if len(chunked_target.shape) == 2:
                chunked_target = self.emb(chunked_target)
            inp = torch.cat(
                [mem[:, chosen_indices], chunked_target],
                dim=0,
            )
            inp = inp + self.pos_emb(
                torch.arange(inp.size(0), device=device)
            ).unsqueeze(1)
            output = self.transformer(inp)
            mem[:, chosen_indices], output = (
                output[: self.mem_size],
                output[self.mem_size :],
            )
            tem[:, chosen_indices] = self.l_out(output)
            out_list.append(tem)
        result = torch.cat(out_list, dim=0)
        loss = self.loss_func(result, target, mask=target_len_tensor)
        return result, loss


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = OpenOrcaDataset.init_encoder(tiktoken.get_encoding("gpt2"))
    dataset = OpenOrcaDataset(
        enc,
        cache_dir="D:/AI/HuggingFace/datasets",
        is_val=True,
        max_len=512,
        val_rate=0.0005,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        collate_fn=dataset.collate_fn,
    )
    model = Model(enc.n_vocab, 768, 3, 8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    scheduler = WarmupCosineLRSchedule(
        optimizer,
        1e-6,
        2e-4,
        3e-5,
        warmup_steps=2000,
        total_steps=20000,
        current_step=0,
    )
    trainer = Trainer(
        model,
        dataloader,
        optimizer,
        scheduler=scheduler,
        model_name="nlp_qa_mem_transformer",
    )
    trainer.train()
