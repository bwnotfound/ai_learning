import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tiktoken
from tqdm import tqdm

from bw_dataset import BookCorpus
from utils import save_model, load_model, accuracy_calc


class Net(nn.Module):

    def __init__(
        self,
        n_vocab,
        in_dim,
        hidden_dim,
        output_dim,
        n_layers,
        start_gen_index,
        dropout=0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.start_gen_index = start_gen_index
        self.embedding = nn.Embedding(n_vocab, in_dim)
        self.rnn = nn.LSTM(in_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_state = None

    def init_hidden(self, batch_size):
        self.hidden_state = (
            torch.zeros(self.n_layers, batch_size, self.hidden_dim),
            torch.zeros(self.n_layers, batch_size, self.hidden_dim),
        )

    def encode(self, src, batch_first=True):
        is_batch = len(src.shape) == 2
        if not is_batch:
            src = src.unsqueeze(0)
        if batch_first:
            src = src.permute(1, 0)
        if self.hidden_state is None:
            self.init_hidden(src.size(1))
        self.hidden_state = [h.to(src.device) for h in self.hidden_state]
        src = self.embedding(src)
        _, self.hidden_state = self.rnn(src, self.hidden_state)
        return self.hidden_state

    def decode(self, hidden_state=None, max_len=32):
        if hidden_state is None:
            hidden_state = self.hidden_state
        assert hidden_state is not None
        batch_size = hidden_state[0].shape[1]
        device = hidden_state[0].device
        output = [
            torch.tensor([self.start_gen_index] * batch_size, dtype=torch.long).to(
                device
            )
        ]
        for i in range(max_len):
            in_tensor = output[-1].unsqueeze(0)
            o, hidden_state = self.rnn(in_tensor, hidden_state)
            output.append(self.fc(o[-1]).argmax(dim=-1))
        return torch.stack(output, dim=0)

    def forward(self, src, batch_first=True):
        if batch_first:
            src = src.permute(1, 0)
        if self.hidden_state is None:
            self.init_hidden(src.size(1))
        self.hidden_state = [h.to(src.device) for h in self.hidden_state]
        src = self.embedding(src)
        _, self.hidden_state = self.rnn(src, self.hidden_state)
        start_tensor = self.embedding(
            torch.tensor([self.start_gen_index], dtype=torch.long)
            .to(src.device)
            .expand(1, src.size(1))
        )
        src = torch.cat([start_tensor, src[:-1]], dim=0)
        output, self.hidden_state = self.rnn(src, self.hidden_state)
        prediction = self.fc(output)
        if batch_first:
            prediction = prediction.permute(1, 0, 2)
        return prediction


if __name__ == '__main__':
    model_name = "rnn_model_predict"
    start_gen_token = "<|startofgen|>"
    encoder = tiktoken.get_encoding("gpt2")
    encoder = tiktoken.Encoding(
        name="gpt2",
        pat_str=encoder._pat_str,
        mergeable_ranks=encoder._mergeable_ranks,
        special_tokens={
            **encoder._special_tokens,
            start_gen_token: encoder.max_token_value + 1,
        },
    )
    start_gen_index = encoder._special_tokens[start_gen_token]
    dataset = BookCorpus(encoder, is_val=True, max_len=32, split=True, val_rate=0.2)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=2,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        drop_last=True,
        batch_sampler=None,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {
        "n_vocab": encoder.n_vocab,
        "in_dim": 32,
        "hidden_dim": 256,
        "output_dim": encoder.n_vocab,
        "n_layers": 2,
        "start_gen_index": start_gen_index,
        "dropout": 0,
    }
    model = Net(**model_kwargs).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    step = load_model("./output", model, optimizer, model_name)
    for epoch in range(10):
        t_bar = tqdm(
            len(dataloader), desc=f"Epoch {epoch + 1}", ncols=100, colour="green"
        )
        for src, len_tensor in dataloader:
            src = src.to(device)
            mask = torch.arange(src.size(1)).unsqueeze(0) >= len_tensor.unsqueeze(1)
            mask = mask.to(device)
            label = torch.clone(src).to(device)
            label[mask] = -1
            model.init_hidden(src.size(0))
            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output.permute(0, 2, 1), label)
            loss.backward()
            optimizer.step()
            t_bar.set_postfix_str(f"loss: {loss.item():.5f}")
            t_bar.update()
            step += 1
            if step % 500 == 0:
                acc, _, _ = accuracy_calc(output, src, ignore_token=-1)
                tqdm.write(f"Step: {step}, Accuracy: {acc:.5f}")

                def run(src):
                    src_ = []
                    flag = False
                    for i in reversed(src):
                        if i == encoder.eot_token and not flag:
                            continue
                        flag = True
                        src_.append(i)
                    src_.reverse()
                    if flag:
                        src_.append(encoder.eot_token)
                    return encoder.decode(src_)

                src_text = run(src[0].cpu().detach().tolist())
                pred_text = run(output.argmax(dim=-1)[0].cpu().detach().tolist())
                with open("./output/test.txt", "w", encoding="utf-8") as f:
                    f.write("输入：{}\n\n预测：{}".format(src_text, pred_text))
                save_model("./output", model, optimizer, step, model_name)
