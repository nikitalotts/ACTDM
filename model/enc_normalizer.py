import torch

from torch import nn, FloatTensor
import os


class EncNormalizer(nn.Module):
    def __init__(self, enc_mean_path: str, enc_std_path: str):
        super().__init__()
        location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.enc_mean = nn.Parameter(
            torch.load(enc_mean_path, map_location=location)[None, None, :],
            requires_grad=False
        )
        self.enc_std = nn.Parameter(
            torch.load(enc_std_path, map_location=location)[None, None, :],
            requires_grad=False
        )
        # self.enc_mean = nn.Parameter(
        #     torch.tensor([1]),
        #     requires_grad=False
        # )
        # self.enc_std = nn.Parameter(
        #     torch.tensor([1]),
        #     requires_grad=False
        # )
        # m=1

    def forward(self, *args, **kwargs):
        return nn.Identity()(*args, **kwargs)

    def normalize(self, encoding: FloatTensor) -> FloatTensor:
        return (encoding - self.enc_mean) / self.enc_std

    def denormalize(self, pred_x_0: FloatTensor) -> FloatTensor:
        return pred_x_0 * self.enc_std + self.enc_mean
