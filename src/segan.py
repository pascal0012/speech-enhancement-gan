from collections import OrderedDict
from typing import Dict, List, Union

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

from src.data.dataloader import get_loader
from src.data.dataset import SpeechData
from src.data.files import write_model
from src.data.filesampler import sample_filepaths
from src.eval.testing import Tester
from src.networks.discriminator import Discriminator
from src.networks.generator import Generator
from src.util.consts import LOG_INTERVAL, TEST_TASK_1
from src.util.diffuser import Diffuser
from src.util.logger import Logger
from src.util.losses import segan_discriminator_loss, segan_generator_loss
from src.util.signals import load_chunks_pair_list


class SEGAN:
    def __init__(
        self,
        levels: List[str],
        hyperparameters: Dict[str, Union[int, float]],
        device: torch.device,
        val_paths: List[str] = None,
        diffusion: bool = True,
        attention: bool = True,
        spectral_norm: bool = True,
    ) -> None:
        self.levels = levels
        self.hyperparameters = hyperparameters
        self.device = device
        self.diffusion = diffusion
        self.tags = []
        if attention:
            self.tags.append("attention")
        if spectral_norm:
            self.tags.append("spectral_norm")

        # Initialize dataset
        self.val_paths = (
            sample_filepaths(tasks=levels, sample_rate=0.005)
            if val_paths is None
            else val_paths
        )
        self.val_chunks = load_chunks_pair_list(self.val_paths)

        dataset = SpeechData(tasks=levels, ignore_paths=self.val_paths)

        self.train_loader = get_loader(
            dataset, batch_size=hyperparameters["batch_size"], device=device
        )

        # Initialize networks
        self.generator = Generator(device=device)
        reference_batch = dataset.get_reference_batch(
            batch_size=hyperparameters["batch_size"], device=device
        )

        # Add diffuser if diffusion GAN is enabled
        diffuser = None
        if diffusion:
            diffuser = Diffuser(
                device=device,
                sigma=hyperparameters["diffuser_sigma"]
                if "diffuser_sigma" in hyperparameters
                else 0.05,
                C=hyperparameters["diffuser_C"]
                if "diffuser_C" in hyperparameters
                else 3,
                d_target=hyperparameters["diffuser_d_target"]
                if "diffuser_d_target" in hyperparameters
                else 0.6,
            )

            self.tags.append("diffusion")

        self.discriminator = Discriminator(
            reference_batch=reference_batch,
            device=device,
            diffuser=diffuser,
        )

        # Compile models if on GPU
        if device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            self.generator = torch.compile(self.generator)
            self.discriminator = torch.compile(self.discriminator)
            print("Compiled generator and discriminator!")

        # Initialize optimizers
        self.generator_optimizer = torch.optim.AdamW(
            self.generator.parameters(), lr=hyperparameters["lr"]
        )
        self.discriminator_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(), lr=hyperparameters["lr"]
        )

        self.reconstruction_loss = torch.nn.MSELoss()

        # Initialize schedulers
        self.gen_scheduler = StepLR(self.generator_optimizer, step_size=1500, gamma=0.1)
        self.disc_scheduler = StepLR(
            self.discriminator_optimizer, step_size=1500, gamma=0.1
        )

    def load(self, path: str):
        # Fix state names from parallel training
        state_dicts = [
            torch.load(f"{path}_generator.pt", map_location=self.device),
            torch.load(f"{path}_discriminator.pt", map_location=self.device),
            torch.load(f"{path}_generator_optimizer.pt", map_location=self.device),
            torch.load(f"{path}_discriminator_optimizer.pt", map_location=self.device)
        ]

        for i, state_dict in enumerate(state_dicts):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_key = k.replace("_orig_mod.", "")
                new_state_dict[new_key] = v

                if i == 0:
                    self.generator.load_state_dict(new_state_dict, strict=True)
                elif i == 1:
                    self.discriminator.load_state_dict(new_state_dict, strict=True)
                elif i == 2:
                    self.generator_optimizer.load_state_dict(new_state_dict, strict=True)
                elif i == 3:
                    self.discriminator_optimizer.load_state_dict(new_state_dict, strict=True)

    def learn(
        self,
        num_episodes: int = 2000,
        sweep: bool = False,
        disable_testing: bool = False,
    ):
        # Initialize logger
        self.logger = Logger(
            tasks=self.levels,
            hyperparameters=self.hyperparameters,
            tags=["segan"] + self.tags,
            sweep=sweep,
            val_paths=self.val_paths,
        )
        self.tester = Tester(
            run_name=self.logger.run_name,
            paths=self.val_paths,
            chunks=self.val_chunks,
            device=self.device,
            disable_testing=disable_testing,
            write_all=True,
        )

        for episode in range(num_episodes):
            for i, (clean_files, recorded_files, _) in enumerate(self.train_loader):
                z = nn.init.normal_(
                    torch.Tensor(clean_files.shape[0], 1024, 8).to(device=self.device)
                )

                clean_files = clean_files.to(self.device, non_blocking=True)
                recorded_files = recorded_files.to(self.device, non_blocking=True)

                # Update discriminator
                self.discriminator.zero_grad(set_to_none=True)
                g_out = self.generator(recorded_files, z)
                d_out_fake = self.discriminator(g_out, recorded_files)
                d_out_real = self.discriminator(clean_files, recorded_files, real=True)
                discriminator_loss = segan_discriminator_loss(
                    d_out_real=d_out_real, d_out_fake=d_out_fake
                )
                discriminator_loss.backward()
                self.discriminator_optimizer.step()

                # Update generator
                self.generator.zero_grad(set_to_none=True)
                g_out = self.generator(recorded_files, z)
                d_out_fake = self.discriminator(g_out, recorded_files)
                generator_loss = segan_generator_loss(
                    g_out=g_out,
                    d_out=d_out_fake,
                    x_clean=clean_files,
                    l1_mag=self.hyperparameters["l1_mag"],
                )

                generator_loss.backward()
                self.generator_optimizer.step()

                # Update diffusion step list
                if i % 4 == 0 and self.diffusion:
                    self.discriminator.update_diffuser()

                if i % LOG_INTERVAL == 0:
                    self.logger.log_metrics(
                        generator_loss=generator_loss.item(),
                        discriminator_loss=discriminator_loss.item(),
                        episode=episode,
                        iteration=i,
                        lr=self.gen_scheduler.get_last_lr()[0],
                    )

                    # Terminate if generator loss diverges
                    if generator_loss > self.hyperparameters["l1_mag"]:
                        self.write()
                        raise ValueError("Generator loss diverged")

            # Update learning rates
            # self.gen_scheduler.step()
            # self.disc_scheduler.step()

            if episode % 80 == 0:
                (
                    chunk_recon_loss,
                    mean_cer,
                    cers,
                    sample_paths,
                    transcriptions,
                ) = self.tester.test(
                    generator=self.generator,
                    episode=episode,
                )

                self.logger.log_metrics(
                    chunk_recon_loss=chunk_recon_loss,
                    mean_cer=mean_cer,
                    cers=cers,
                    episode=episode,
                    iteration=i,
                    lr=self.gen_scheduler.get_last_lr()[0],
                    audio_paths=sample_paths,
                    transcriptions=transcriptions,
                )

                self.write(episode=episode)

        self.logger.finish()

    def test(self, id: int = None):
        # Validation result
        val_results = self.tester.test(
            generator=self.generator,
            paths=self.val_paths,
            chunks=self.val_chunks,
            episode=f"final_validation_{id}" if id is not None else "final_validation",
            device=self.device,
            write_all=True,
        )

        # Test results
        self.test_paths = sample_filepaths(TEST_TASK_1, sample_rate=1)
        test_chunks = load_chunks_pair_list(sampled_paths=self.test_paths)
        test_results = self.tester.test(
            generator=self.generator,
            paths=self.test_paths,
            chunks=test_chunks,
            episode=f"final_testing_{id}" if id is not None else "final_testing",
            device=self.device,
            write_all=True,
        )

        return val_results, test_results

    def write(self, episode: int = None):
        name_prefix = f"episode_{episode}_" if episode is not None else ""

        write_model(
            model=self.generator,
            run_name=self.logger.run_name,
            model_name=f"{name_prefix}generator",
        )
        write_model(
            model=self.discriminator,
            run_name=self.logger.run_name,
            model_name=f"{name_prefix}discriminator",
        )
        write_model(
            self.generator_optimizer,
            run_name=self.logger.run_name,
            model_name=f"{name_prefix}generator_optimizer",
        )
        write_model(
            self.discriminator_optimizer,
            run_name=self.logger.run_name,
            model_name=f"{name_prefix}discriminator_optimizer",
        )
