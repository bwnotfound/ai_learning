import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL.Image import Image, fromarray

from torchvision.datasets import MNIST
from torchvision.transforms import (
    Compose,
    ToTensor,
    Resize,
    Normalize,
    ColorJitter,
)

from bw_dataset import ImageNetSmall
from utils import load_model, save_model


class Net(nn.Module):

    def __init__(
        self,
        in_channels=3,
        embed_dim=512,
        n_layer=4,
        n_head=8,
        mlp_ratio=2,
    ):
        r'''
        should be 224x224 and patch_size should be divisible by 224
        '''
        super(Net, self).__init__()
        self.feat_extract_conv_seq = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 16, 7, 2, 3),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(embed_dim // 16, embed_dim // 8, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 8, embed_dim // 4, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 4, embed_dim, 3, 2, 1),
        )
        self.pitch_encoder = nn.Sequential(  # 32x32 -> 1x1
            nn.Conv2d(in_channels, embed_dim // 16, 5, 2, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(embed_dim // 16, embed_dim // 8, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 8, embed_dim // 4, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(embed_dim, embed_dim // 4),
        )
        self.decoder_pre = nn.Sequential(
            nn.Linear(int(embed_dim * (1 + 1 / 4)), embed_dim),
            nn.ReLU(),
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_head,
                dim_feedforward=mlp_ratio * embed_dim,
            ),
            num_layers=n_layer,
        )

        self.pos_embed = nn.Embedding(7 * 7 + 3, embed_dim)
        self.mask_embed = nn.Parameter(torch.randn(embed_dim) * 0.01)
        self.restore_embed = nn.Parameter(torch.randn(embed_dim) * 0.01)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, 2, 2, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 8, embed_dim // 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 16, embed_dim // 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 32, in_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        mask = torch.zeros([N, L], device=x.device, dtype=ids_keep.dtype)
        mask.scatter_add_(1, ids_keep, torch.ones_like(ids_keep, device=x.device))
        x_masked = torch.clone(x)
        x_masked[mask == 0] = self.mask_embed
        return x_masked, mask

    def forward_encode(self, imgs, mask_ratio=None):
        imgs = self.feat_extract_conv_seq(imgs)
        imgs = imgs.permute(0, 2, 3, 1).reshape(imgs.shape[0], -1, imgs.shape[1])
        if mask_ratio is not None:
            imgs, mask = self.random_masking(imgs, mask_ratio)
        imgs = imgs.transpose(0, 1)
        imgs = torch.cat(
            [self.restore_embed.view(1, 1, -1).repeat(1, imgs.shape[1], 1), imgs], dim=0
        )
        imgs = imgs + self.pos_embed(
            torch.arange(imgs.shape[0], device=imgs.device)
        ).unsqueeze(1)
        imgs = self.encoder(imgs)[0]
        if mask_ratio is not None:
            return imgs, mask
        return imgs

    def forward_decode(self, latent, pitch_latent):
        latent = torch.cat([latent, pitch_latent], dim=1)
        latent = self.decoder_pre(latent)
        latent = latent.view(latent.shape[0], latent.shape[1], 1, 1)
        pitch_pred = self.decoder(latent)
        return pitch_pred

    def forward_loss(self, imgs, pred, mask=None):
        """
        imgs/pred: [N, 3, H, W]
        mask: [N, 7*7]
        """
        loss = (pred - imgs) ** 2
        loss = loss.mean(dim=1)  # [N, L], mean loss per patch
        # if mask is not None:
        #     mask = mask.reshape(-1, 7, 7).repeat(1, 32, 32)
        #     loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        # else:
        loss = loss.mean()
        return loss

    def forward(self, imgs):
        latent, mask = self.forward_encode(imgs, mask_ratio=0)
        pitch_imgs = torch.einsum(
            "bdhpwq->bpqdhw", imgs.reshape(imgs.shape[0], imgs.shape[1], 32, 7, 32, 7)
        ).reshape(-1, imgs.shape[1], 32, 32)
        pitch_imgs_latent = self.pitch_encoder(pitch_imgs)
        pitch_pred = self.forward_decode(
            torch.repeat_interleave(latent, 7 * 7, dim=0), pitch_imgs_latent
        )
        imgs_pred = torch.einsum(
            "bpqdhw->bdhpwq",
            pitch_pred.reshape(imgs.shape[0], 7, 7, pitch_pred.shape[1], 32, 32),
        ).reshape_as(imgs)
        return imgs_pred, self.forward_loss(imgs, imgs_pred)

### TODO: 接下来可以针对不同的pitch设置不同的匹配度label。离目标pitch越近，越接近1，反之越接近0。这样可以算作匹配度。然后需要从其他图片截取一些pitch作为0 label。
### 结论：可行，但是没有验证是否可以作为backbone改善下游任务。
if __name__ == '__main__':
    model_name = "img_feature_match_model"
    model_kwargs = {
        "in_channels": 3,
        "embed_dim": 512,
        "n_layer": 3,
        "n_head": 8,
        "mlp_ratio": 2,
    }
    dataset = ImageNetSmall()
    # transforms = Compose(
    #     [
    #         Resize((224, 224)),
    #         ToTensor(),
    #     ]
    # )
    # dataset = MNIST("./datasets", train=True, download=True, transform=transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    step = load_model("./output", model, optimizer, model_name)
    for epoch in range(100000):
        t_bar = tqdm(
            len(dataloader), desc=f"Epoch {epoch + 1}", ncols=100, colour="green"
        )
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            # label = labels.to(device)
            optimizer.zero_grad()
            output, loss = model(imgs)
            loss.backward()
            optimizer.step()
            t_bar.set_postfix_str(f"loss: {loss.item():.5f}")
            t_bar.update()
            step += 1
            if step % 200 == 0:
                fromarray(
                    (
                        torch.cat([imgs[0], output[0]], dim=2)
                        .permute(1, 2, 0)
                        # .repeat(1, 1, 3)
                        .cpu()
                        .detach()
                        .numpy()
                        * 255
                    ).astype("uint8")
                ).save("./output/test.png")
                save_model("./output", model, optimizer, step, model_name)
