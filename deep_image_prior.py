import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os
from colibri.optics.blur import Convolution
from utils import (
    save_metrics,
    AverageMeter,
    save_npy_metric,
    save_coded_apertures,
    set_seed,
    print_dict,
)
import logging
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
from models import FullyConnected, CNN, CNNwithAttention, MultiScale, ViT, ConvNeXt, UNet, IndiUnet
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

import argparse
from torch.utils.data import DataLoader, random_split
import numpy as np



def create_high_pass_low_pass_kernel(kernel_size=5, sigma=1.0, factor=1.0):
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    gaussian = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    gaussian = gaussian / torch.sum(gaussian)
    delta = torch.zeros_like(gaussian)
    center = kernel_size // 2
    delta[center, center] = 1.0

    gaussian_2 = torch.exp(-(xx**2 + yy**2) / (2. * (sigma*factor)**2))
    gaussian_2 = gaussian_2 / torch.sum(gaussian_2)

    high_pass = delta - gaussian_2
    return high_pass.view(1, 1, kernel_size, kernel_size), gaussian.view(1, 1, kernel_size, kernel_size)

def main(args):

    set_seed(args.seed)
    imgs_idx= [400,500,600,700,10000]
    image_dir = r"C:\Roman\NPN_Clean\NPN\data\val_256\test" # Replace with your image path
    img_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.png') or img.endswith('.jpg')]
    max_epochs = 1000

    psnr_vals = np.zeros((len(imgs_idx),max_epochs))
    tt = -1
    for idx in imgs_idx:
        tt += 1
        transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((args.n,args.n), antialias=True)])

        # load a single 

        image_path = f"{img_paths[idx]}"  
        image = Image.open(image_path).convert("L")

        image = transform(image).unsqueeze(0).to(args.device)  # Add batch dimension

        G = UNet(in_channels=1, out_channels=1,args=args).to(args.device)

        hp_kernel, lp_kernel = create_high_pass_low_pass_kernel(kernel_size=args.n, sigma=args.sigma,factor=args.factor)

        # Apply convolution
        path_name = f"weights/deb/Places_G_sigma_4.pth"
        conv_lp = Convolution(lp_kernel.to(args.device))
        conv_hp = Convolution(hp_kernel.to(args.device))
        args.save_path = path_name

        G.load_state_dict(torch.load(f"{args.save_path}"))

        D =  UNet(in_channels=1, out_channels=1,args=args).to(args.device)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        PSNR = PeakSignalNoiseRatio(data_range=1.0).to(device)

        im_size = (1, args.n, args.n)
        optimizer = torch.optim.AdamW(D.parameters(), lr=1e-3)

        criterion = nn.MSELoss()
        # wandb.login(key="13977ceba9b05d840afc8127f3e542f41809ffff")
        # wandb.init(project=args.project_name, name=path_name, config=args)
        z = torch.randn_like(image).to(device)
        epoch_loop = tqdm(range(max_epochs), desc="Epochs", unit="epoch")

        for epoch in epoch_loop:

                train_loss = AverageMeter()
                train_psnr = AverageMeter()


                y = conv_lp(image)
                ys =G(y)

                x_hat = D(z)
                y_hat = conv_lp(x_hat)
                ys_hat = conv_hp(x_hat)

                fid_loss = criterion(y_hat, y)
                reg_loss = criterion(ys_hat, ys)
                loss_train =fid_loss + args.gamma *reg_loss

                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

                train_psnr = PSNR(image, x_hat).item()
                ssim_val = SSIM(image, x_hat).item()
                
                epoch_loop.set_description(f"Epoch: {epoch + 1}/{max_epochs} - Loss: {loss_train.item():.4f} - PSNR: {train_psnr:.2f} - SSIM: {ssim_val:.4f}")
                epoch_loop.set_postfix(loss=loss_train.item(), psnr=train_psnr, 
                                         reg_loss=reg_loss.item(), fid_loss=fid_loss.item())

                psnr_vals[tt,epoch] = train_psnr
        
        np.save(f"{args.save_path}/metrics_psnr_ssim_{args.gamma}.npy", np.array(psnr_vals))
  

    # wandb.finish()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Processing")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=160)
    parser.add_argument("--save_path", type=str, default="weights/")
    parser.add_argument("--seed", type=int, default=10) 
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=4) # standard deviation of the Gaussian blur kernel
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--features', type=int, default=128, help='Numero de canales base en ConvNeXt.')
    parser.add_argument('--grad_acc', type=int, default=1, help='Si no es 1, usa gradient accumulation.')
    parser.add_argument('--blocks', type=int, default=5, help='Numero de bloques base en ConvNeXt.')
    parser.add_argument('--res_scaler', type=float, default=1.0)
    parser.add_argument('--drop_path_prob', type=float, default=0.0)
    parser.add_argument('--WSConv', type=str, default='False', help='Si True, WSConv en init y final layer.')
    parser.add_argument('--use_SE', type=str, default='False', help='Si True, activa bloques SE en ConvNeXtBlock.')
    parser.add_argument('--use_PE', type=str, default='False', help='Activar Positional Encoding')    
    parser.add_argument('--use_cosPE', type=str, default='False', help='Activar cosine/sine Positional Encoding')
    parser.add_argument('--time_dim', type=int, default=256, help='Dimension of time for cosine/sine Positional Encoding')
    parser.add_argument('--R_linear_module', type=str, default='S')
    parser.add_argument('--exp_mode', type=str, default='Residual', help='Use wandb')
    parser.add_argument('--factor', type=int, default=1, help='Acceleration factor')
    parser.add_argument('--gamma', type=float, default=0.1, help='Regularization parameter for the prior term') # put this value to 0.0 for base DIP, best value was \gamma = 0.1
    args = parser.parse_args()


    args_dict = vars(args)

    print_dict(args_dict)
    main(args)
