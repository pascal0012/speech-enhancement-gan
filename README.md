# Diffusion-SEGAN: Speech Enhancement Generative Adversarial Network (trained with diffusion)

This repository contains an implementation of the [Speech Enhancement Generative Adversarial Network](https://arxiv.org/abs/1703.09452) introduced by S. Pascual et. al in 2017 combined with [Diffusion-GAN](https://arxiv.org/abs/2206.02262) (Z. Wang, 2023).

## Authors

Implemented by [Pascal Makossa](mailto:pascal.makossa@tu-dortmund.de) under the supervision of Sebastian
Konietzny and Prof. Dr. Stefan Harmeling (Artifical Intelligence Chair, Department of Computer Science, TU Dortmund University).

## About

This project was created to participate in the [Helsinki Speech Challenge 2024](https://arxiv.org/abs/2406.04123) and is therefore designed to run on that dataset, however it should be easily adaptable to other datasets.

## Architecture

This implementation is mostly based on the SEGAN paper. It is based on training a discriminator $D$ that is trained to distinguish between generated and clean audio samples by optimizing for

$`\min_D V_\text{LSGAN}(D,G)=\frac{1}{2}\mathbb{E}_{x,x_c\sim p_{data}(x,x_c)}[(D(x,x_c)-1)^2]+\frac{1}{2}\mathbb{E}_{z\sim p_z(z),x_c\sim p_{data}(x_c)}[D(G(z,x_c),x_c)^2]`$

while the generator $G$ is trained to create samples that can not be recognised as fake. A weighted reconstruction loss term is added to ensure the generated samples represent the input file:

$`\min_G V_\text{LSGAN}(D,G)=\frac{1}{2}\mathbb{E}_{z\sim p_z(z),\tilde{x}\sim p_{data}(\tilde{x})}[(D(G(z,\tilde{x}),\tilde{x})-1)^2]+\lambda\|G(z,\tilde{x})-x\|_1`$

To improve training performance, additional noise is added to the discriminator input $x$ according to:

$`y=\sqrt{\overline{a_t}}x-\sqrt{1-\overline{a_t}}\sigma\varepsilon`$ with $`\overline{a_t}=\prod_{s=1}^ta_s`$ and $`\varepsilon\sim\mathcal{N}(0,\mathcal{I})`$

## Installation

Python 3.9 is required for evaluation with the DeepSpeech model, for training and enhancement newer versions work aswell. It is recommended to create a **virtual environment** to install compatible versions of the following packages via `pip`:

- torch (with torchaudio)
- python-dotenv
- numpy

After that simply install the package itself by running `pip install .` in the root directory.

### Training Requirements

Furthermore the training loop requires some additional packages to be installed mostly for calculating the CER. For that, the deepspeech model and scorer need to be stored in the `model` folder.

- wandb
- librosa
- pandas
- jiwer
- deepspeech
- numpy < 2.0

On top of that, the `.env` file needs to be modified with the following values (All paths have to be relative from the `train.py` script).

|Key |Description |
|----|------|
|`MODEL_PATH`|Path to model|
|`SCORER_PATH`|Path to scorer|
|`TEXT_FILE_PATH`|Path to file with transcriptions|

The correct transcriptions need to be stored under the provided path in the following format:

```txt
file_name.wav   Original text
```

Furthermore, the following existing variables can be modified if a different dataset is used.

|Key |Description |
|----|------|
|`DATA_PATH`|Path to input dataset structured in tasks|
|`CLEAN_PATH`|Path to clean data in each task folder|
|`RECORDED_PATH`|Path to noisy data in each task folder|
|`OUTPUT_PATH`|Path to folder for testing results|

## Running

For all scripts, navigate to the `scripts` subfolder before executing them.

### Training

To train your own models run the `train.py` file  with the following (optional) command line arguments:

| Argument | Default | Description |
|----------|---------|-------------|
|`levels` | - | `Task_1`, `Task_2` or `All` |
|`epochs` | 4,000 | Training epochs |
|`batch_size`|50 | Batch size per network pass |
|`lr` |0.0001 | Learning Rate |
|`recon_mag`|100|Magnitude of reconstruction loss term|

### Enhancement

To enhance all audio files contained in a directory, run the `main.py` file with the following command line arguments:

| Argument | Description |
|----------|-------------|
|`input_dir`|Relative path to recorded files|
|`output_dir`|Relative path to store enhanced files|
|`task_id`|Task the files belong to (Format: `TXLY`)|

## Results

Visible below are some examples of enhanced audio files:

TODO
