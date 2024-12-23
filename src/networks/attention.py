import torch
from torch import nn


class Self_Attention(nn.Module):
    """
    Self Attention layer according to the SASEGAN paper
    https://arxiv.org/abs/2010.09132
    """

    def __init__(self, in_channels, channel_reduction=8, pooling_kernel=4):
        super(Self_Attention, self).__init__()

        reduced_channels = in_channels // channel_reduction

        self.wq = nn.Conv1d(
            in_channels=in_channels, out_channels=reduced_channels, kernel_size=1
        )
        self.wk = nn.Conv1d(
                in_channels=in_channels,
                out_channels=reduced_channels,
                kernel_size=1,
            )

        self.wv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=reduced_channels,
                kernel_size=1,
            )

        self.wo = nn.Conv1d(
            in_channels=reduced_channels, out_channels=in_channels, kernel_size=1
        )

        self.max_pool = nn.MaxPool1d(kernel_size=pooling_kernel, stride=pooling_kernel)
        self.softmax = nn.Softmax(dim=-1)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        q = self.wq(x)
        kT = self.wk(x).permute(0, 2, 1)
        vT = self.wv(x).permute(0, 2, 1)

        kT = self.max_pool(kT)
        v = self.max_pool(vT).permute(0, 2, 1)

        a = self.softmax(q @ kT)
        o = self.wo(a @ v)

        return x + self.beta * o
