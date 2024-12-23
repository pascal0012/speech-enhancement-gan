import torch
import torch.nn as nn

from src.networks.spectral_norm import SpectralNorm
from src.util.consts import N_FILTERS


class UNet_1D(nn.Module):
    def __init__(self, device: str):
        super(UNet_1D, self).__init__()

        # Encoder
        self.encoders = nn.ModuleList(
            [self._encode_layer(N_FILTERS[i], N_FILTERS[i + 1]) for i in range(11)]
        )

        # Decoder
        # Decoder levels have double the number of filters due to skip connections
        self.decoders = nn.ModuleList(
            [self._decode_layer(N_FILTERS[i + 1] * 2, N_FILTERS[i]) for i in range(10)]
            + [self._decode_layer(N_FILTERS[-1], N_FILTERS[-2])]
        )

        self.prelu = nn.PReLU()
        self.tanh = nn.Tanh()

        self.to(device)

    def forward(self, x: torch.Tensor):
        # Encoder
        encoder_outputs = []
        for i in range(11):
            x = self.encoders[i](x)
            encoder_outputs.append(x)

            x = self.prelu(x)

        # Combine encoded features with latent variable

        # Decoder
        x = self.decoders[10](x)
        for i in range(9, -1, -1):
            x = torch.cat([x, encoder_outputs[i]], dim=1)
            x = self.prelu(x)

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

        return conv

    def _decode_layer(self, in_channels: int, out_channels: int):
        conv_t = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=31,
            stride=2,
            padding=15,
            output_padding=1,
        )
        return conv_t
