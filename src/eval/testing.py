import os
from typing import Dict, List, Tuple

import torch
from torch import nn
from torchaudio.functional import deemphasis

from src.data.files import create_dir, get_output_dir, write_file, write_model
from src.eval.evaluate_integration import evaluate_parameters
from src.networks.discriminator import Discriminator
from src.networks.generator import Generator
from src.util.signals import CombinedChunks, patch_signals
from src.data.filesampler import PathPair


class Tester:
    def __init__(
        self,
        run_name: str,
        paths: List[PathPair],
        chunks: List[CombinedChunks],
        device: str,
        write_all: bool = True,
        disable_testing: bool = False,
        verbose: bool = False,
    ) -> None:
        self.base_path = run_name
        self.loss = nn.MSELoss()
        self.episode = -1
        self.paths = paths
        self.chunks = chunks
        self.verbose = verbose
        self.device = device
        self.write_all = write_all
        self.disable_testing = disable_testing

        self.num_files = len(self.chunks)

        # Stack all chunks used for validation
        self.recorded_chunks = torch.cat(
            [recorded_chunks for _, recorded_chunks in self.chunks], dim=0
        ).to(self.device)

    def test(
        self,
        episode: int,
        generator: Generator,
    ) -> Tuple[float, float, Dict[str, float], List[str], List[str]]:
        self.create_episode_dir(episode)

        if self.disable_testing:
            return 0, 0, {}, [], []

        chunk_recon_loss = 0
        sample_paths = []
        essential_levels = [
            "task_1_level_1",
            "task_1_level_4",
            "task_1_level_7",
            "task_2_level_1",
        ]

        # Pass all recorded chunks through generator at once
        all_result_chunks = torch.cat(
            [
                generator(self.recorded_chunks[i * 128 : (i + 1) * 128])
                for i in range(self.recorded_chunks.shape[0] // 128 + 1)
            ],
            dim=0,
        )

        for (clean_chunks, recorded_chunks), (_, recorded_path) in zip(
            self.chunks, self.paths
        ):
            clean_chunks = clean_chunks.to(self.device)
            # Get result chunks for current file and remove them from chunk list
            result_chunks = all_result_chunks[: len(recorded_chunks)]
            all_result_chunks = all_result_chunks[len(recorded_chunks) :]

            # Calculate loss for chunks
            chunk_recon_loss += self.loss(result_chunks, clean_chunks) / self.num_files

            # Calculate loss for patched files
            result_file = patch_signals(result_chunks)

            level = "_".join(os.path.basename(recorded_path).split("_")[:4])
            if (level in essential_levels) or self.write_all:
                # Regulate high frequencies to original levels
                result_file = deemphasis(result_file, 0.95)

                # Create directory for episode and write file
                write_path = self.get_path(recorded_path)
                write_file(result_file, write_path)

                if level in essential_levels:
                    essential_levels.remove(level)
                    sample_paths.append(write_path)

        mean_cer, cers, transcriptions = evaluate_parameters(
            get_output_dir(self.get_episode_path()), verbose=self.verbose
        )

        return (
            chunk_recon_loss,
            mean_cer,
            cers,
            sample_paths,
            transcriptions,
        )

    def create_episode_dir(self, episode: int):
        self.episode = episode
        create_dir(self.get_episode_path())

    def get_episode_path(self) -> str:
        episode = (
            f"{self.episode}"
            if self.episode >= 1000
            else f"0{self.episode}"
            if self.episode >= 100
            else f"00{self.episode}"
            if self.episode >= 10
            else f"000{self.episode}"
        )
        return os.path.join(self.base_path, f"episode_{episode}")

    def get_path(self, filepath: str) -> str:
        """
        Creates the directory for the current episodes and returns the path
        """
        filename = os.path.basename(filepath)
        episode_path = self.get_episode_path()

        return os.path.join(episode_path, filename)
