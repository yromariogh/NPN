import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os
from colibri.optics.blur import Convolution, ConvolutionSR
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
from utils import ImageDataset
from torch.utils.data import DataLoader, random_split
import numpy as np
    # Create kernels


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

 
    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((args.n,args.n), antialias=True)])

    dataset = ImageDataset(r'data', transform=transform) # [Replace with your image path]
    # make split for train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    G = UNet(in_channels=1, out_channels=1,args=args).to(args.device)

    hp_kernel, lp_kernel = create_high_pass_low_pass_kernel(kernel_size=args.n, sigma=args.sigma,factor=args.factor)

    
    # Apply convolution
    path_name = f"weights/deb/Places_G_sigma_4.pt"
    conv_lp = ConvolutionSR(lp_kernel.to(args.device),RF = args.RF)
    conv_hp = ConvolutionSR(hp_kernel.to(args.device),RF=1)
    args.save_path = args.save_path + path_name

    images_path, model_path, metrics_path = save_metrics(f"{args.save_path}")

    args.save_path = args.save_path + path_name
    if os.path.exists(f"{args.save_path}/metrics/metrics.npy"):
        print("Experiment already done")
        exit()

    logging.basicConfig(
            filename=f"{metrics_path}/training.log",
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
        )

    logging.info(f"Starting training with parameters: {args}")

    loss_train_record = np.zeros(args.epochs)
    psnr_train_record = np.zeros(args.epochs)
    loss_val_record = np.zeros(args.epochs)
    psnr_val_record = np.zeros(args.epochs)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    PSNR = PeakSignalNoiseRatio(data_range=1.0).to(device)

    optimizer = torch.optim.AdamW(G.parameters(), lr=args.lr)

    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
            G.train()

            train_loss = AverageMeter()
            train_psnr = AverageMeter()

            data_loop_train = tqdm(enumerate(trainloader), total=len(trainloader), colour="red")
            for _, train_data in data_loop_train:

                img = train_data
                img = img.to(device)

                y = conv_lp(img)
                ys = conv_hp(img)
                ys_pred = G(conv_lp(y,type_calculation="backward"))
                loss_train = criterion(ys_pred, ys)

                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

                train_loss.update(loss_train.item())
                train_psnr.update(PSNR(ys_pred, ys).item())

                data_loop_train.set_description(f"Epoch: {epoch + 1}/{args.epochs}")
                data_loop_train.set_postfix(loss=train_loss.avg, psnr=train_psnr.avg)

            logging.info(
                f"Epoch: {epoch} - Train Loss: {np.round(train_loss.avg, 4)} - Train PSNR: {np.round(train_psnr.avg, 4)}"
            )

           
            torch.save(G.state_dict(), f"{model_path}/G.pth")


            loss_train_record[epoch] = train_loss.avg
            psnr_train_record[epoch] = train_psnr.avg
           
        # TEST BEST VAL MODEl

            val_loss = AverageMeter()

            val_psnr = AverageMeter()
        
            data_loop_val = tqdm(enumerate(testloader), total=len(testloader), colour="green")
            with torch.no_grad():
                G.eval()

                for _, val_data in data_loop_val:
                    img = val_data
                    img = img.to(device)

                    y = conv_lp(img)
                    ys = conv_hp(img)

                    ys_pred = G(conv_lp(y,type_calculation="backward"))

                    loss_val = criterion(ys_pred, ys)

                    val_loss.update(loss_val.item())
                    val_psnr.update(PSNR(ys_pred, ys).item())

                    data_loop_val.set_description("VAL")
                    data_loop_val.set_postfix(loss=val_loss.avg, psnr=val_psnr.avg)



    # # save data
    save_npy_metric(
        dict(
            loss_train_record=loss_train_record,
            psnr_train_record=psnr_train_record,
            loss_val_record=loss_val_record,
            psnr_val_record=psnr_val_record,
        ),
        f"{metrics_path}/metrics",
    )

  

    # wandb.finish()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Processing")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--save_path", type=str, default="weights/")
    parser.add_argument("--project_name", type=str, default="NPN_Deconv")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=6.0)
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
    parser.add_argument('--factor', type=float, default=4.0, help='Acceleration factor')
    parser.add_argument('--sigma_y', type=float, default=0.0, help='Standard deviation of the noise')
    parser.add_argument('--RF', type=float, default=2.0, help='Downsampling factor for the convolutional layer')
    args = parser.parse_args()

    args_dict = vars(args)

    print_dict(args_dict)
    main(args)
