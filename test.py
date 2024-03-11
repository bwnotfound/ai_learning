# from time import perf_counter
# import torch
# import torch.nn as nn

# from simpletool.log.time import TimeConsumeLog

# model = nn.Sequential(
#     nn.TransformerEncoder(nn.TransformerEncoderLayer(1024, 8), 3),
# )
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# with TimeConsumeLog('分开'):
#     length, epo = 32, 10
#     optimizer.zero_grad()
#     x = torch.randn(length, 32, 1024).to(device)
#     for i in range(epo):
#         x = model(x)
#     loss = x.sum()
#     loss.backward()
#     optimizer.step()

# with TimeConsumeLog('合并'):
#     optimizer.zero_grad()
#     x = torch.randn(length * epo, 32, 1024).to(device)
#     y = model(x)
#     loss = y.sum()
#     loss.backward()
#     optimizer.step()


import torch

x = torch.arange(24).reshape(2, 3, 4)
chosen = torch.tensor([0, 1])
x[:, chosen] = torch.arange(8).reshape(2, 4) - 10
print(x)
