import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import tiktoken
from tqdm import tqdm
from utils import load_model, save_model

from bw_dataset import OpenOrcaDataset
from configs.config import LLMTransformerConfig as Config


class Net(nn.Module):
    def __init__(
        self, embedding_dim, hidden_dim, n_vocabs, n_layers, n_heads, dropout, max_len
    ):
        super().__init__()
        assert embedding_dim == hidden_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_vocabs = n_vocabs
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.max_len = max_len
        self.text_emb = nn.Embedding(n_vocabs, embedding_dim)
        self.pos_emb = nn.Embedding(max_len, embedding_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, n_heads, dropout=dropout),
            num_layers=n_layers,
        )
        self.post_proj = nn.Linear(embedding_dim, n_vocabs)

    def forward(self, ids_tensor, answer_start_idx_tensor, batch_first=True):
        if batch_first:
            ids_tensor = ids_tensor.transpose(0, 1)
        batch_size = ids_tensor.size(1)
        length = ids_tensor.size(0)
        device = ids_tensor.device
        attn_mask_list = []
        for i in range(batch_size):
            attn_mask_list.append(
                torch.cat(
                    [
                        torch.zeros(
                            (length, answer_start_idx_tensor[i]), dtype=torch.bool
                        ),
                        torch.cat(
                            [
                                torch.ones(
                                    answer_start_idx_tensor[i],
                                    length - answer_start_idx_tensor[i],
                                    dtype=torch.bool,
                                ),
                                torch.triu(
                                    torch.ones(
                                        length - answer_start_idx_tensor[i],
                                        length - answer_start_idx_tensor[i],
                                        dtype=torch.bool,
                                    ),
                                    diagonal=1,
                                ),
                            ],
                            dim=0,
                        ),
                    ],
                    dim=1,
                )
                .unsqueeze(0)
                .repeat(self.n_heads, 1, 1)
            )
        attn_mask = torch.cat(attn_mask_list, dim=0).to(device)
        input_emb = self.text_emb(ids_tensor)
        pos_ids = (
            torch.arange(length).unsqueeze(1).expand(length, batch_size).to(device)
        )
        pos_emb = self.pos_emb(pos_ids)
        input_emb = input_emb + pos_emb
        output = self.encoder(
            input_emb,
            mask=attn_mask,
            is_causal=True,
        )
        logits = self.post_proj(output)
        if batch_first:
            logits = logits.transpose(0, 1)
        return logits


if __name__ == '__main__':
    model_name = "llm_transformer_model"
    config = Config()
    enc = tiktoken.get_encoding("gpt2")
    enc = tiktoken.Encoding(
        name="gpt2",
        pat_str=enc._pat_str,
        mergeable_ranks=enc._mergeable_ranks,
        special_tokens={
            **enc._special_tokens,
            config.sys_prompt_start: enc.max_token_value + 1,
            config.sys_prompt_end: enc.max_token_value + 2,
            config.mem_start: enc.max_token_value + 3,
            config.mem_end: enc.max_token_value + 4,
            config.question_start: enc.max_token_value + 5,
            config.question_end: enc.max_token_value + 6,
            config.answer_start: enc.max_token_value + 7,
            config.answer_end: enc.max_token_value + 8,
        },
    )

    model = Net(
        config.embedding_dim,
        config.hidden_dim,
        enc.n_vocab,
        config.n_layers,
        config.n_heads,
        config.dropout,
        config.max_len,
    ).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    step = load_model(config.save_dir, model, optimizer, model_name=model_name)

    dataset = OpenOrcaDataset(config, enc)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    for epoch in range(config.epochs):
        t_bar = tqdm(
            total=len(dataloader), desc=f"epoch {epoch}", ncols=120, colour="green"
        )
        for ids_tensor, len_tensor, answer_start_idx_tensor in dataloader:
            ids_tensor, len_tensor, answer_start_idx_tensor = (
                ids_tensor.to(config.device, non_blocking=True),
                len_tensor.to(config.device, non_blocking=True),
                answer_start_idx_tensor.to(config.device, non_blocking=True),
            )
            optimizer.zero_grad()
            logits = model(ids_tensor, answer_start_idx_tensor)
            mask = (
                torch.arange(logits.size(1)).unsqueeze(0).to(config.device)
                < len_tensor.unsqueeze(1)
            ) * (
                torch.arange(logits.size(1)).to(config.device)
                >= answer_start_idx_tensor.unsqueeze(1)
            )
            masked_ids_tensor = torch.masked_fill(
                ids_tensor, torch.logical_not(mask.to(torch.bool)), -1
            )[:, 1:]
            masked_logits = logits[:, :-1].transpose(1, 2)
            loss = F.cross_entropy(masked_logits, masked_ids_tensor, ignore_index=-1)

            loss.backward()
            optimizer.step()
            t_bar.set_postfix_str("loss: {:.4f}".format(loss.item()))
            t_bar.update()
            if step % config.save_per_steps == 0:
                save_model(
                    config.save_dir, model, optimizer, step, model_name=model_name
                )
                acc = (
                    (torch.argmax(masked_logits, dim=1) == masked_ids_tensor)
                    * mask[:, 1:]
                ).sum() / mask[:, 1:].sum()
                tqdm.write(f"model_{step}.pt saved. Accuracy: {acc:.4f}")
                source_answer = masked_ids_tensor[0]
                predict_answer = torch.cat(
                    [source_answer[:1], torch.argmax(masked_logits[0], dim=0)[:-1]]
                )
                new_mask = torch.logical_not(source_answer == -1)
                source_answer = source_answer[new_mask]
                predict_answer = predict_answer[new_mask]
                source_answer = enc.decode(source_answer.cpu().detach().tolist())
                predict_answer = enc.decode(predict_answer.cpu().detach().tolist())
                with open("./test.txt", "w", encoding="utf-8") as f:
                    f.write(
                        "---{}---\n\n---{}---".format(source_answer, predict_answer)
                    )
            step += 1
