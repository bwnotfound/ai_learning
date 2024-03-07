import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from simpleai.task.memory.number_recall import NumberRecallDataset
from simpleai.util.trainer import SimpleTrainer
from simpleai.data.util import load_model


class RecallModel(nn.Module):

    def __init__(self, n_vocab, hidden_dim, n_layer):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.emb = nn.Embedding(n_vocab, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layer)
        self.fc = nn.Linear(hidden_dim, n_vocab)
        self.question_emb = nn.Parameter(torch.randn(1, hidden_dim))

    def init_hidden(self, batch_size):
        return [
            torch.zeros(self.n_layer, batch_size, self.hidden_dim),
            torch.zeros(self.n_layer, batch_size, self.hidden_dim),
        ]

    def forward(self, x, hidden=None, batch_first=True):
        if batch_first:
            x = x.transpose(0, 1)
        device = x.device
        if hidden is None:
            hidden = self.init_hidden(x.size(1))
        hidden = [h.to(device) for h in hidden]
        x = self.emb(x)
        repeat_time = torch.randint(1, 4, (1,)).item()
        x = torch.cat(
            [
                x[:-1].repeat(repeat_time, 1, 1),
                # self.question_emb.unsqueeze(1).repeat(1, x.size(1), 1),
                x[-1:],
            ],
            dim=0,
        )
        x, _ = self.lstm(x, hidden)
        x = x[-1]
        x = self.fc(x)
        return x


if __name__ == '__main__':
    dataset = NumberRecallDataset()
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True
    )
    model = RecallModel(dataset.n_vocab, 256, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = SimpleTrainer(model, dataloader, optimizer)
    trainer.train()
