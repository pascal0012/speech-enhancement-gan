from typing import Optional

import torch
import torch.nn as nn

from src.networks.attention import Self_Attention
from src.networks.spectral_norm import SpectralNorm
from src.networks.vbatch_norm import VirtualBatchNorm
from src.util.consts import N_FILTERS
from src.util.diffuser import Diffuser


class Discriminator(nn.Module):
    def __init__(
        self,
        reference_batch: torch.Tensor,
        device: str,
        diffuser: Optional[Diffuser] = None,
        attention: bool = True,
        spectral_norm: bool = True,
    ):
        super(Discriminator, self).__init__()

        self.reference_batch = reference_batch
        self.attention = attention
        self.spectral_norm = spectral_norm

        self.conv_layers = nn.ModuleList(
            [self._conv_layer(N_FILTERS[0] * 2, N_FILTERS[1])]
        )
        self.conv_layers.extend(
            [
                self._conv_layer(N_FILTERS[i], N_FILTERS[i + 1])
                for i in range(1, 11)
            ]
        )
        self.vbatch_norms = nn.ModuleList(
            [VirtualBatchNorm(num_features=N_FILTERS[i + 1]) for i in range(11)]
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.03)

        if attention:
            self.attention_layer = Self_Attention(N_FILTERS[11])

        self.final_layers = self._final_layers()
        self.fc = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

        # Diffusion GAN tools
        self.diffuser = diffuser
        self.last_out = None

        self.to(device)

    def forward(self, clean: torch.Tensor, recorded: torch.Tensor, real: bool = False):
        x = torch.cat([clean, recorded], dim=1)

        # Step for diffusion GAN
        if self.diffuser is not None:
            x = self.diffuser.diffuse(x)

        # Reference pass
        reference_batch = self.reference_batch
        means = []
        mean_sqs = []
        for i in range(len(self.conv_layers)):
            reference_batch = self.conv_layers[i](reference_batch)
            reference_batch, mean, mean_sq = self.vbatch_norms[i](reference_batch)
            reference_batch = self.lrelu(reference_batch)

            means.append(mean)
            mean_sqs.append(mean_sq)

        # Discriminator pass
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = self.vbatch_norms[i](x, means[i], mean_sqs[i])
            x = self.lrelu(x)

            if i == 10 and self.attention:
                x = self.attention_layer(x)

        x = self.final_layers(x)
        x = torch.squeeze(x)
        x = self.fc(x)
        x = self.sigmoid(x)

        if real:
            self.last_out = x
            
        return x

    def update_diffuser(self):
        assert self.diffuser is not None, "Diffusion is not enabled"
        assert self.last_out is not None, "No output pass before updating diffuser"

        self.diffuser.update_curr_t(self.last_out)

    def _conv_layer(self, in_channels: int, out_channels: int):
        conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=31,
                stride=2,
                padding=15,
            )
        
        return SpectralNorm(conv) if self.spectral_norm else conv

    @staticmethod
    def _final_layers():
        return nn.Sequential(
            nn.Conv1d(in_channels=N_FILTERS[11], out_channels=1, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.03),
        )
