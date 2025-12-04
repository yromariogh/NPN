import os
import torch
import numpy as np
import torch.nn as nn
from torchmetrics.image import PeakSignalNoiseRatio
from models import UNet
# from utils_pnp import get_dataloaders, remove_directory_from_path
# from colibri.optics.spc import SPC
from colibri.optics.blur import Convolution, ConvolutionSR
from colibri.recovery.fista import Fista, MultiRegFista
from colibri.recovery.pnp import PnP_ADMM, MultiRegPnP_ADMM
from colibri.recovery.terms.fidelity import L2
from colibri.recovery.terms.prior import Denoiser, Sparsity, DenoiserRED
from models import ConvNeXt
import argparse
# from utils_pnp import JPGImageDataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from utils import ImageDataset, get_dataloaders, psnr_fun

def create_high_pass_low_pass_kernel(kernel_size=5, sigma=1.0):
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    gaussian = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    gaussian = gaussian / torch.sum(gaussian)
    delta = torch.zeros_like(gaussian)
    center = kernel_size // 2
    delta[center, center] = 1.0
    high_pass = delta - gaussian
    return high_pass.view(1, 1, kernel_size, kernel_size), gaussian.view(1, 1, kernel_size, kernel_size)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_measurement_operators(args):

    hp_kernel, lp_kernel = create_high_pass_low_pass_kernel(kernel_size=args.n, sigma=args.sigma)
    if args.task == 'sr':

        lp_conv = ConvolutionSR(lp_kernel.to(args.device),RF=args.RF).to(args.device)
        hp_conv = ConvolutionSR(hp_kernel.to(args.device),RF=1.0).to(args.device)
    elif args.task == 'deconv':
        lp_conv = Convolution(lp_kernel.to(args.device)).to(args.device)
        hp_conv = Convolution(hp_kernel.to(args.device)).to(args.device)
    return lp_conv, hp_conv





def run_fista_and_log_metrics(x, yt, lp_conv, hp_conv, ys, ys_hat,args):
    psnr = PeakSignalNoiseRatio().to(args.device)
    mse = nn.MSELoss()
    l2 = L2()

    # Examples of other priors
    # prior = Denoiser({'in_channels': 1, 'out_channels': 1, 'pretrained': "download", 'device': args.device},denoiser='DRUNet').to(args.device)
    # prior = Denoiser({'in_chans': 1, 'pretrained': "download", 'device': args.device,'pretrained_noise_level': 50},denoiser='SwinIR').to(args.device)
    # prior = Sparsity('dct')
    prior = Denoiser({'in_channels': 1, 'out_channels': 1, 'pretrained': "download_lipschitz", 'device': args.device},denoiser='DnCNN').to(args.device) # You can the denoiser here
    gammas = [0.4]

    shape = (len(gammas), args.max_iter)
    metrics = {
        'ys_Sx_loss': np.zeros(shape),
        'Gy_Sx_loss': np.zeros(shape),
        'ratio_ys': np.zeros(shape),
        'Stys_StSx_loss': np.zeros(shape),
        'StGy_StSx_loss': np.zeros(shape),
        'ratio_Stys': np.zeros(shape),
        'psnrs': np.zeros(shape),
        'psnr_base': np.zeros(shape),
        'Recon': np.zeros(x.shape),
        'mse_x_hat': np.zeros(shape),
        'mse_x_hat_base': np.zeros(shape),
        'Recon_base': np.zeros(x.shape),
        'ys_hat': np.zeros(x.shape),
        'ys': np.zeros(x.shape),
        'base_Sx_Sxl': np.zeros(shape),
        'ratio_convergence': np.zeros(shape),
        'base_ratio_convergence': np.zeros(shape)  
    }

    for t1, gamma in enumerate(gammas):
        alpha1 =0.8
        # You can change it to FISTA Algorithm 
        # fista = Fista(lp_conv, fidelity=l2, prior=prior, max_iters=args.max_iter, alpha=alpha1, _lambda=0.005).eval()
        admm_pnp = PnP_ADMM(lp_conv, fidelity=l2, prior=prior, max_iters=args.max_iter, alpha=alpha1, _lambda=0.005,rho=0.01,solver='gradient').eval()
        admm_pnp_npn = MultiRegPnP_ADMM(acquisition_model=[lp_conv, hp_conv], fidelity=[l2,l2], prior=prior, max_iters=args.max_iter, alpha=[alpha1,gamma*alpha1], _lambda=0.005,rho=1,solver='gradient').eval()
        x0_base = lp_conv(yt[0], type_calculation='backward')
        x0_base = x0_base / x0_base.max()
        print('Runing ADMM Prop')
        _, x_hats_base = admm_pnp_npn(x0=x0_base, y=yt, freq=1, xgt=x)
        print('Runing ADMM BASE')
        _,x_hats = admm_pnp(x0=x0_base, y=yt[0], freq=1, xgt=x)
        for i in range(args.max_iter - 1):
            x_hat_i = x_hats_base[i]
            Sx_i = hp_conv(x_hat_i)
            Bx_i = hp_conv(Sx_i, type_calculation='backward')
            Sx_i_base = hp_conv(x_hats[i])
            metrics['ys_Sx_loss'][t1, i] = mse(Sx_i, ys).item()
            metrics['Gy_Sx_loss'][t1, i] = mse(Sx_i, ys_hat).item()
            metrics['base_Sx_Sxl'][t1, i] = mse(Sx_i_base, ys).item()
            # Compute convergence ratio: ||x^{i+1} - x|| / ||x^t - x||
            if i < args.max_iter - 2:
                num = torch.norm(x_hats_base[i+1] - x)
                denom = torch.norm(x_hats_base[i] - x)
                metrics['ratio_convergence'][t1, i] = (num / denom).item() if denom != 0 else 0.0

                num = torch.norm(x_hats[i+1] - x)
                denom = torch.norm(x_hats[i] - x)
                metrics['base_ratio_convergence'][t1, i] = (num / denom).item() if denom != 0 else 0.0
            else:
                
                metrics['base_ratio_convergence'][t1, i] = 0.0
                metrics['ratio_convergence'][t1, i] = 0.0


            metrics['ratio_ys'][t1, i] = (mse(Sx_i, ys) - mse(Sx_i, ys_hat)) / mse(Sx_i, ys)
            metrics['psnrs'][t1, i] = psnr(x_hat_i, x).item()
            metrics['mse_x_hat'][t1, i] = mse(x_hat_i, x).item()
            metrics['Stys_StSx_loss'][t1, i] = mse(Bx_i, hp_conv(ys, type_calculation="backward")).item()
            metrics['StGy_StSx_loss'][t1, i] = mse(Bx_i, hp_conv(ys_hat, type_calculation="backward")).item()
            metrics['ratio_Stys'][t1, i] = (
                mse(Bx_i, hp_conv(ys, type_calculation="backward")) -
                mse(Bx_i, hp_conv(ys_hat, type_calculation="backward"))
            ) / mse(Bx_i, hp_conv(ys, type_calculation="backward"))

            metrics['psnr_base'][t1, i] = psnr(x_hats[i], x).item()
    print('========================')
    print('####### RESULTS #######')
    print('========================')
    print("Best PSNR :", metrics['psnrs'][t1, :].max())
    print("Best PSNR base:", metrics['psnr_base'][t1, :].max())
    metrics['Recon'] = x_hats_base[-1].detach().cpu().numpy()
    metrics['Recon_base'] = x_hats[-1].detach().cpu().numpy()   
    metrics['ys_hat'] = ys_hat.detach().cpu().numpy()
    metrics['ys'] = ys.detach().cpu().numpy()
    return x_hats_base, metrics


def save_metrics(metrics, args, ys, ys_hat, spc_s):

    print("Saving metrics to", args.local_path)
    np.save(os.path.join(args.local_path, f'metrics.npy'), metrics)
    np.save(os.path.join(args.local_path, f'psnr.npy'), metrics['psnrs'])
    np.save(os.path.join(args.local_path, f'ys_Sx_loss.npy'), metrics['ys_Sx_loss'])
    np.save(os.path.join(args.local_path, f'Gy_Sx_loss.npy'), metrics['Gy_Sx_loss'])
    np.save(os.path.join(args.local_path, f'ratio_ys.npy'), metrics['ratio_ys'])
    np.save(os.path.join(args.local_path, f'Stys_StSx_loss.npy'), metrics['Stys_StSx_loss'])
    np.save(os.path.join(args.local_path, f'StGy_StSx_loss.npy'), metrics['StGy_StSx_loss'])
    np.save(os.path.join(args.local_path, f'ratio_Stys.npy'), metrics['ratio_Stys'])
    np.save(os.path.join(args.local_path, f'psnr_base.npy'), metrics['psnr_base'])
    np.save(os.path.join(args.local_path, f'ys_hat.npy'), metrics['ys_hat'])
    np.save(os.path.join(args.local_path, f'ys.npy'), metrics['ys'])
    np.save(os.path.join(args.local_path, f'Recon.npy'), metrics['Recon'])
    np.save(os.path.join(args.local_path, f'Recon_base.npy'), metrics['Recon_base'])
    
    
    mse = nn.MSELoss()
    mse_ys = np.ones((args.max_iter - 1, 1)) * mse(ys, ys_hat).item()
    mse_Stys = np.ones((args.max_iter - 1, 1)) * mse(spc_s(ys, type_calculation="backward"), spc_s(ys_hat, type_calculation="backward")).item()

    np.save(os.path.join(args.local_path, f'mse_ys_net.npy'), mse_ys)
    np.save(os.path.join(args.local_path, f'mse_Stys_net.npy'), mse_Stys)


def main(args):
    set_seed(args.seed)
    lp_conv, hp_conv= load_measurement_operators(args)
    
    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((args.n,args.n), antialias=True)])
    # dataset = ImageDataset(r'C:\Roman\datasets\CelebA2\test', transform=transform) # [Replace with your image path]
    dataset = ImageDataset(r'C:\Roman\NPN_Clean\NPN\data\places\test', transform=transform) # [Replace with your image path]
    # make split for train and test

    testloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    PSNR = PeakSignalNoiseRatio().to(args.device)
    

    
    G = UNet(in_channels=1, out_channels=1,args=args).to(args.device)
    if args.task == 'sr':
        path_name = r"weights\sr\Places_G_srf_2.pth"
    elif args.task == 'deconv':
        path_name = f"weights/deb/Places_G_sigma_{args.sigma}.pth"

    G.load_state_dict(torch.load(path_name))
    x = next(iter(testloader)).to(args.device)
    y = lp_conv(x, type_calculation='forward')
    if args.task == 'sr':
        ys_hat = G(lp_conv(y, type_calculation='backward'))
    else:
        ys_hat = G(y)
    
    ys = hp_conv(x, type_calculation='forward')
    


    yt = [y,ys_hat]

    print(psnr_fun(ys_hat, ys).item())

    
    x_hats_base, metrics = run_fista_and_log_metrics(x, yt, lp_conv, hp_conv, ys, ys_hat,args)
    args.local_path = os.path.join(args.local_path, f"admm_fac_UNET_HP_LP_PnP_sigma_{args.sigma}")
    os.makedirs(args.local_path, exist_ok=True)
    save_metrics(metrics, args, ys, ys_hat, hp_conv)

    print("Final PSNR:", psnr_fun(x_hats_base[-1], x).item())

                                                

        
if __name__ == '__main__':
     
    parser = argparse.ArgumentParser(description='Learning cost')
    #----------------------------------- Dataset parameters ---------------------------------
    parser.add_argument('--n', type=int, default=128, help='Size of the image')
    parser.add_argument('--batch_size', type=int, default=40, help='Batch size')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset') 
    #----------------------------------- FISTA parameters -----------------------------------
    parser.add_argument('--algo', type=str, default='fista', help='Algorithm')
    parser.add_argument('--max_iter', type=int, default=200, help='Maximum number of iterations')
    parser.add_argument('--alpha', type=float, default=1e-3, help='Step size')
    parser.add_argument('--_lambda', type=float, default=0.1, help='Regularization parameter')
    #----------------------------------- Other parameters -----------------------------------
    parser.add_argument('--seed', type=int, default=0, help='Seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    # ---------------------------------- DeepNorm parameters --------------------------------
    parser.add_argument('--s_path', type=str, default=r'', help='Path to the S matrix')
    parser.add_argument('--h_path', type=str, default=r'matrices\crA0.3_crB0.1_n32.pt', help='Path to the H matrix')
    parser.add_argument('--cr_s', type=float, default=0.0, help='Compression ratio for S')
    parser.add_argument('--cr_h', type=float, default=0.0, help='Compression ratio for H')
    # ---------------------------------- Network parameters --------------------------------
    parser.add_argument('--hidden_size_base', type=int, default=2048, help='Hidden size base for the network')
    parser.add_argument('--dim_mid', type=int, default=1024, help='Mid dimension for the network')
    parser.add_argument('--dim_hid', type=int, default=102, help='Hidden size for the network')

    parser.add_argument('--net', type=str, default='ConvNeXt')
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
    parser.add_argument('--sigma', type=int, default=1.0, help='Project name for wandb')
    parser.add_argument('--path', type=str, default='weights\HP_LP_lr_0.0001_b_32_e_500_sd_1_bc_16_im_128_sigma_5.0\model')
    parser.add_argument('--gamma', type=float, default=10, help='Use wandb')
    parser.add_argument('--local_path', type=str, default=r'', help='Path to the local folder')
    parser.add_argument('--sigma_y', type=float, default=0.0, help='Standard deviation of the noise')
    parser.add_argument('--RF', type=float, default=4.0, help='Downsampling factor for the convolutional layer')
    parser.add_argument('--task', type=str, default='sr', help='Type of inverse problem')
    args = parser.parse_args()

    
    main(args)

    