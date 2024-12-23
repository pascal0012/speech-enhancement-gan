from typing import Optional

import torch
import torch.nn as nn

from src.networks.attention import Self_Attention
from src.networks.spectral_norm import SpectralNorm
from src.util.consts import N_FILTERS


class Generator(nn.Module):
    def __init__(self, device: str, attention: bool = True, spectral_norm: bool = True):
        super(Generator, self).__init__()
        self.attention = attention
        self.spectral_norm = spectral_norm

        # Encoder
        self.encoders = nn.ModuleList(
            [self._encode_layer(N_FILTERS[i], N_FILTERS[i + 1]) for i in range(11)]
        )

        # Attention layers
        if attention:
            self.encode_attention = Self_Attention(N_FILTERS[11])
            self.decode_attention = Self_Attention(N_FILTERS[10])

        # Decoder
        # Decoder levels have double the number of filters due to skip connections
        self.decoders = nn.ModuleList(
            [
                self._decode_layer(N_FILTERS[i + 1] * 2, N_FILTERS[i])
                for i in range(11)
            ]
        )

        self.prelu = nn.PReLU()
        self.tanh = nn.Tanh()

        self.to(device)

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None):
        if z is None:
            z = nn.init.normal_(torch.Tensor(x.shape[0], 1024, 8).to(x.device))

        # Encoder
        encoder_outputs = []
        for i in range(11):
            x = self.encoders[i](x)
            encoder_outputs.append(x)

            if i == 10 and self.attention:
                x = self.encode_attention(x)

            x = self.prelu(x)

        # Combine encoded features with latent variable
        x = torch.cat([x, z], dim=1)

        # Decoder
        x = self.decoders[10](x)
        for i in range(9, -1, -1):
            x = torch.cat([x, encoder_outputs[i]], dim=1)
            x = self.prelu(x)

            if i == 10 and self.attention:
                x = self.decode_attention(x)

            x = self.decoders[i](x)

        x = self.tanh(x)

        return x

    def _encode_layer(self, in_channels: int, out_channels: int):
        conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=31,
                stride=2,
                padding=15,
            )
        
        return SpectralNorm(conv) if self.spectral_norm else conv

    def _decode_layer(self, in_channels: int, out_channels: int):
        conv_t = nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=31,
                stride=2,
                padding=15,
                output_padding=1,
            )
        return SpectralNorm(conv_t) if self.spectral_norm else conv_t
