import argparse
import os

from src.segan import SEGAN
from src.util.consts import TASK_1, TASK_2
from src.util.device import set_device


def main():
    parser = argparse.ArgumentParser(
        description="Trains a (Diffusion-)SEGAN model on the provided levels."
    )

    parser.add_argument(
        "--levels",
        type=str,
        nargs="+",
        help="Levels to train on.",
        required=True,
    )

    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--epochs", type=int, default=4_000, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--recon_mag",
        type=float,
        default=100,
        help="Magnitude of reconstruction in generator loss",
    )

    parser.add_argument(
        "--no_diffusion",
        action=argparse.BooleanOptionalAction,
        help="Whether to enable noise on discriminator inputs",
    )
    parser.add_argument(
        "--no_attention",
        action=argparse.BooleanOptionalAction,
        help="Whether to disable attention in the generator and discriminator",
    )
    parser.add_argument(
        "--no_spectral_norm",
        action=argparse.BooleanOptionalAction,
        help="Whether to disable spectral normalization in the generator and discriminator",
    )

    parser.add_argument(
        "--load_val_paths",
        action=argparse.BooleanOptionalAction,
        help="Whether to load validation paths from val_paths.txt",
    )

    args = parser.parse_args()

    # Set device
    device = set_device()

    # Decode levels
    if args.levels[0] == "Task_1":
        args.levels = TASK_1
    elif args.levels[0] == "Task_2":
        args.levels = TASK_2
    elif args.levels[0] == "All":
        args.levels = TASK_1 + TASK_2

    # Load validation paths
    val_paths = None
    if args.load_val_paths:
        with open('../val_paths.txt', "r") as f:
            val_paths = [line.split(',') for line in f.read().splitlines()]

    segan = SEGAN(
        levels=args.levels,
        hyperparameters={
            "batch_size": args.batch_size,
            "lr": args.lr,
            "l1_mag": args.recon_mag,
        },
        val_paths=val_paths,
        diffusion=not(args.no_diffusion),
        attention=not(args.no_attention),
        spectral_norm=not(args.no_spectral_norm),
        device=device,
    )

    segan.learn(num_episodes=args.epochs)

    segan.write()

    print_results(segan.test())


def print_results(test_results):
    recon_loss, mean_cer, cers, sample_paths, transcriptions = test_results

    print(f"Reconstruction loss: {recon_loss}")
    print(f"Mean CER: {mean_cer}")
    print("----------------------------------")

    for level, cer in cers:
        print(f"CER {level}: {cer}")

    print("----------------------------------")

    for path, transcription in zip(sample_paths, transcriptions):
        print(f"Transcription {os.path.basename(path)}: {transcription}")


if __name__ == "__main__":
    main()
