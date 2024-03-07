import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tiktoken
from tqdm import tqdm

from bw_dataset import TranslateDataset
from utils import save_model, load_model, accuracy_calc


class Net(nn.Module):

    def __init__(self, n_vocab, in_dim, hidden_dim, output_dim, n_layers, dropout=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(n_vocab, in_dim)
        self.rnn = nn.LSTM(in_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_state = None

    def init_hidden(self, batch_size):
        self.hidden_state = (
            torch.zeros(self.n_layers, batch_size, self.hidden_dim),
            torch.zeros(self.n_layers, batch_size, self.hidden_dim),
        )

    def forward(self, src, batch_first=True):
        if batch_first:
            src = src.permute(1, 0)
        if self.hidden_state is None:
            self.init_hidden(src.size(1))
        self.hidden_state = [h.to(src.device) for h in self.hidden_state]
        src = self.embedding(src)
        output, self.hidden_state = self.rnn(src, self.hidden_state)
        prediction = self.fc(output)
        if batch_first:
            prediction = prediction.permute(1, 0, 2)
        return prediction


if __name__ == '__main__':
    model_name = "rnn_model"
    encoder = tiktoken.get_encoding("gpt2")
    dataset = TranslateDataset(encoder)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=0,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        drop_last=True,
        batch_sampler=None,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(encoder.n_vocab, 64, 256, encoder.n_vocab, 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    step = load_model("./output", model, optimizer, model_name)
    for epoch in range(10):
        t_bar = tqdm(
            len(dataloader), desc=f"Epoch {epoch + 1}", ncols=100, colour="green"
        )
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            model.init_hidden(src.size(0))
            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output.permute(0, 2, 1), tgt)
            loss.backward()
            optimizer.step()
            t_bar.set_postfix_str(f"loss: {loss.item():.5f}")
            t_bar.update()
            step += 1
            if step % 500 == 0:
                acc, _, _ = accuracy_calc(output, tgt, ignore_token=encoder.eot_token)
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
                tgt_text = run(tgt[0].cpu().detach().tolist())
                with open("./output/test.txt", "w", encoding="utf-8") as f:
                    f.write(
                        "输入：{}\n\n预测：{}\n\n目标：{}".format(
                            src_text, pred_text, tgt_text
                        )
                    )
                save_model("./output", model, optimizer, step, model_name)
