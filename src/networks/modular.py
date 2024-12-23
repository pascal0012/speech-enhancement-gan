import torch
from torch import nn

from src.util.signals import istft, stft


class Lineator(nn.Module):
    def __init__(self, n: int, device: str):
        super(Lineator, self).__init__()

        self.alphas = nn.Parameter(torch.randn([n], device=device) / 100)
        self.betas = nn.Parameter(torch.randn([n], device=device) / 100)
        self.gammas = nn.Parameter(torch.randn([n], device=device) / 100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        betas = self.betas[None, :, None].broadcast_to(1, 5, x.shape[2])
        gammas = self.gammas[None, :, None].broadcast_to(1, 5, x.shape[2])
        out = betas * x
        out = out + gammas
        return torch.sum(
            self.alphas[None, :, None] * nn.functional.tanh(out), dim=1, keepdim=True
        )


class Resonator(nn.Module):
    def __init__(self, n_fft: int, device: str, num_matrices: int = 1):
        super(Resonator, self).__init__()

        self.n_fft = n_fft
        self.num_matrices = num_matrices

        # Initialise matrix with some random noise
        # First dimension are the different shifted matrices
        self.As = nn.Parameter(
            torch.randn(num_matrices, n_fft, n_fft, dtype=torch.cfloat, device=device)
            / 100
            + 0j
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        As = self.As[None, :, :, :].broadcast_to(x.shape[0], -1, -1, -1)
        spectogram = stft(x, self.n_fft)
        #spectograms = As @ spectogram[:, None, :, :]

        shifted_result = torch.zeros(
            spectogram.shape, dtype=torch.cfloat, device=x.device
        )
        for i in range(self.num_matrices):
            s = As[:,i] @ spectogram
            shifted = torch.roll(s, shifts=-i, dims=2)
            shifted_result += shifted

        x = istft(shifted_result, self.n_fft).real

        return x


class Model(nn.Module):
    def __init__(
        self, device: str, n_lin_functions: int, n_fft: int, num_matrices: int
    ):
        super(Model, self).__init__()

        self.lineator = Lineator(n_lin_functions, device)
        
        self.resonator = Resonator(n_fft, device, num_matrices)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.resonator(x)
        x = self.lineator(x)

        return x + identity
