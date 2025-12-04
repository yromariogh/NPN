# NPN: Non-Linear Projections of the Null-Space for Imaging Inverse Problems

This repository accompanies the paper "NPN: Non-Linear Projections of the Null-Space for Imaging Inverse Problems" NeurIPS 2025. NPN learns a neural projection onto a compact, task-aware subspace of the sensing operator's null-space and uses it as a prior for plug-and-play and unrolled reconstruction of inverse problems such as deblurring and super-resolution.

## Method in brief
- Learn a projection matrix S that spans informative directions of the null-space of the sensing matrix H.
- Train a network $G(\mathbf{y})$ to predict the coefficients of Sx from measurements $\mathbf{y}$ instead of regressing the full image; corrections are confined to components the measurement process cannot see.
- Combine the learned prior with classical solvers (PnP-ADMM, FISTA), deep image prior, or other denoisers to improve convergence and fidelity across sensing operators.

## Repository layout
- models.py: UNet, ConvNeXt-style blocks, spectral-norm MLPs, autoencoders, and helper modules used for the null-space projector and reconstruction nets.
- train_g_deconv.py: trains the null-space projection network G on blurred/downsampled images using paired low-pass/high-pass measurements.
- pnp_admm.py: plug-and-play ADMM or FISTA reconstruction that injects the learned null-space projection during optimization.
- deep_image_prior.py: deep image prior baseline regularized by the NPN projection.
- utils.py: dataset loaders, seeding, logging helpers, and metric utilities.
- colibri/: sensing operators (blur, super-resolution), PnP/ADMM solvers, and fidelity/prior terms.
- deepinv/: bundled dependency providing datasets, priors, and physics layers used by the scripts.
- weights/: checkpoints expected by the training and reconstruction scripts.

## Setup
1. Create a Python environment (Python >=3.10 recommended) with GPU-enabled PyTorch.
2. Install dependencies:
   `pip install torch torchvision torchmetrics matplotlib pillow tqdm opencv-python wandb`
   The local `deepinv` and `colibri` packages are imported directly from this repo.
3. Prepare data:
   - For training G, place grayscale PNG/JPG images under `data/` (the script splits train/test automatically and resizes to `n x n`).
   - For PnP reconstruction, point `pnp_admm.py` to your evaluation set (default is `data/places/test`; update the `dataset = ImageDataset(...)` line).
   - For the deep image prior experiment, update `image_dir` inside `deep_image_prior.py` to your validation images.

## Training the null-space projector
The projector is trained to map low-pass measurements to their high-pass counterparts, learning how null-space energy should look for a given sensing model.
Example (deblurring with Gaussian kernel, downsample factor RF=2):
```bash
python train_g_deconv.py --n 128 --sigma 6.0 --factor 4 --RF 2 \
  --batch_size 64 --epochs 100 --lr 1e-3 --save_path weights/deb/ --device cuda
```
Outputs: model weights plus logs and metrics inside the directory created under `--save_path` (see the script for the exact naming convention).

## Plug-and-play ADMM reconstruction
Uses the learned G to guide PnP-ADMM (or FISTA) with both low-pass (range-space) and high-pass (null-space) measurements.
- Ensure a trained checkpoint exists at `weights/sr/Places_G_srf_2.pth` (super-resolution) or `weights/deb/Places_G_sigma_<sigma>.pth` (deblurring).
- Adjust the dataset path and weight file if your data/checkpoints are elsewhere.
Example (super-resolution, RF=4):
```bash
python pnp_admm.py --task sr --n 128 --RF 4 --max_iter 200 \
  --sigma 1.0 --local_path outputs/pnp_sr --device cuda
```
The script logs PSNR, MSE traces, reconstructions, and intermediate projections in `outputs/pnp_sr`.

## Deep Image Prior with NPN regularization
Runs a DIP that reconstructs from noise while matching both low-pass fidelity and the learned null-space projection.
```bash
python deep_image_prior.py --n 128 --sigma 4 --gamma 0.1 \
  --epochs 300 --save_path weights/dip_runs --device cuda
```
`gamma` controls the strength of the null-space regularization term.

## Notes on measurement models
- Blur kernels are built in `create_high_pass_low_pass_kernel` with `sigma` (std) and `factor` (secondary blur scale) and are applied via `colibri.optics.blur.Convolution` or `ConvolutionSR` (with downsampling factor `RF`).
- The learned projector is inserted after the reconstruction network via `R_linear_module` (linear, MLP, spectral-norm MLP, or identity) set in `args`.

## Citing
If you use this code or build on the ideas, please cite:
```bibtex
@inproceedings{
jacome2025npn,
title={{NPN}: Non-Linear Projections of the Null-Space for Imaging Inverse Problems},
author={Roman Jacome and Romario Gualdr{\'o}n-Hurtado and Le{\'o}n Su{\'a}rez-Rodr{\'\i}guez and Henry Arguello},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=G67ZNmeWJ5}
}
```
