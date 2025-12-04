import torch

from .transforms import DCT2D
import sys
sys.path.append("....")
from deepinv.models import DnCNN, Restormer, SCUNet, SwinIR, UNet, DRUNet



class Denoiser(torch.nn.Module):

    def __init__(self, denoiser_args=None,denoiser='DnCNN'):
        super().__init__()

        if denoiser == 'DnCNN':
            self.denoiser = DnCNN(**denoiser_args)
        if denoiser == 'Restormer':
            self.denoiser = Restormer(**denoiser_args)
            
        if denoiser == 'SCUNet':
            self.denoiser = SCUNet(**denoiser_args)
        if denoiser == 'SwinIR':
            self.denoiser = SwinIR(**denoiser_args)
        if denoiser == 'UNet':
            self.denoiser = UNet(**denoiser_args)
        if denoiser == 'DRUNet':
            self.denoiser = DRUNet(**denoiser_args)

    def prox(self, x, _lambda):
        with torch.no_grad():
            if x.shape[1] == 1:
                x = self.denoiser(x,_lambda)
            
            if x.shape[1] == 2:
                x_c1 = x[:, 0:1, :, :]  
                x_c2 = x[:, 1:2, :, :]

                d_c1 = self.denoiser(x_c1)
                d_c2 = self.denoiser(x_c2)

                x = torch.cat((d_c1, d_c2), dim=1)
        return x

class DenoiserRED(torch.nn.Module):

    def __init__(self, denoiser_args=None):
        super().__init__()

        self.denoiser = DnCNN(**denoiser_args)

    def prox(self, x, _lambda):
        with torch.no_grad():
            if x.shape[1] == 1:
                xd = self.denoiser(x)
            
            if x.shape[1] == 2:
                x_c1 = x[:, 0:1, :, :]  
                x_c2 = x[:, 1:2, :, :]

                d_c1 = self.denoiser(x_c1)
                d_c2 = self.denoiser(x_c2)

                xd = torch.cat((d_c1, d_c2), dim=1)
        return _lambda*(x - xd)
    

class DenoirserRED_ULA(torch.nn.Module):
    def __init__(self, denoiser_args=None):
        super().__init__()

        self.denoiser = Restormer(**denoiser_args)

    def prox(self, x, _lambda, alpha=1e-3):
        with torch.no_grad():
            if x.shape[1] == 1:
                xd = self.denoiser(x)
            
            if x.shape[1] == 2:
                x_c1 = x[:, 0:1, :, :]  
                x_c2 = x[:, 1:2, :, :]

                d_c1 = self.denoiser(x_c1)
                d_c2 = self.denoiser(x_c2)

                xd = torch.cat((d_c1, d_c2), dim=1)
        e = torch.randn_like(xd)*torch.sqrt(2*alpha)
        
        return _lambda*(x - xd) - e
class DenoiserRestormer(torch.nn.Module):
    def __init__(self, denoiser_args=None):
        super().__init__()

        self.denoiser = Restormer(**denoiser_args)

    def prox(self, x, _lambda):
        with torch.no_grad():
            if x.shape[1] == 1:
                x = self.denoiser(x)
            
            if x.shape[1] == 2:
                x_c1 = x[:, 0:1, :, :]  
                x_c2 = x[:, 1:2, :, :]

                d_c1 = self.denoiser(x_c1)
                d_c2 = self.denoiser(x_c2)

                x = torch.cat((d_c1, d_c2), dim=1)
        return x

class Sparsity(torch.nn.Module):
    r"""
        Sparsity prior 
        
        .. math::
        
            g(\mathbf{x}) = \| \transform \textbf{x}\|_1
        
        where :math:`\transform` is the sparsity basis and :math:`\textbf{x}` is the input tensor.

    """
    def __init__(self, basis=None):
        r"""
        Args:
            basis (str): Basis function. 'dct', 'None'. Default is None.
        """
        super(Sparsity, self).__init__()

        if basis == 'dct':
            self.transform = DCT2D()
        else:
            self.transform = None

    def forward(self, x):
        r"""
        Compute sparsity term.

        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Sparsity term.
        """
        x = self.transform.forward(x)
        return torch.norm(x, 1)**2
    
    def prox(self, x, _lambda, type="soft"):
        r"""
        Compute proximal operator of the sparsity term.

        Args:
            x (torch.Tensor): Input tensor.
            _lambda (float): Regularization parameter.
            type (str): String, it can be "soft" or "hard".
        
        Returns:
            torch.Tensor: Proximal operator of the sparsity term.
        """
        
        x = x.requires_grad_()
        x = self.transform.forward(x)

        if type == 'soft':
            x = torch.sign(x)*torch.max(torch.abs(x) - _lambda, torch.zeros_like(x))
        elif type == 'hard':
            x = x*(torch.abs(x) > _lambda)
        
        x = self.transform.inverse(x)
        return x
        
    def transform(self, x):
        
        if self.transform is not None:
            return self.transform.forward(x)
        else:
            return x
    
    def inverse(self, x):
        
        if self.transform is not None:
            return self.transform.inverse(x)
        else:
            return x
        
        
    
def soft_threshold(x, lmbda):
    return torch.sign(x) * torch.clamp(torch.abs(x) - lmbda, min=0)

import torch

class TVPrior(torch.nn.Module):
    r"""
    Total‐Variation prior (anisotropic)

    .. math::

        g(\mathbf{x}) \;=\; \sum_{i,j}
          \bigl|\,x_{i+1,j} - x_{i,j}\bigr|
        \;+\;\bigl|\,x_{i,j+1} - x_{i,j}\bigr|

    where \(\mathbf{x}\) is assumed to be a 2D (or batched) image tensor.
    """
    def __init__(self):
        super().__init__()
        # no learnable parameters

    def forward(self, x):
        """
        Compute the (anisotropic) TV norm of x.

        Args:
            x (torch.Tensor): shape (B, C, H, W) or (C, H, W) or (H, W)

        Returns:
            torch.Tensor: scalar TV(x)
        """
        # ensure 4D: B x C x H x W
        orig_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)

        # horizontal and vertical differences
        dh = x[..., 1:, :] - x[..., :-1, :]   # shape (B, C, H-1, W)
        dw = x[..., :, 1:] - x[..., :, :-1]   # shape (B, C, H, W-1)

        tv = dh.abs().sum() + dw.abs().sum()
        return tv

    def prox(self, x, lam, n_iter=10):
        """
        Proximal operator of TV:
            prox_{lam * TV}(x) = argmin_u  1/2||u - x||^2 + lam * TV(u)

        This is typically implemented via Chambolle’s algorithm.

        Args:
            x (torch.Tensor): input image, shape (B,C,H,W) or (C,H,W) or (H,W)
            lam (float): regularization weight
            n_iter (int): number of Chambolle iterations

        Returns:
            torch.Tensor: the proximal result u
        """
        # For brevity, we leave this unimplemented.
        # You can plug in a standard Chambolle‐type solver here.
        with torch.no_grad():

            x_prox = tv_prox_soft_threshold(x, lam=lam)
        return x_prox

    def transform(self, x):
        # identity
        return x

    def inverse(self, x):
        # identity
        return x



import torch
import torch.nn.functional as F
def tv_prox_soft_threshold(x, lam=0.1):
    """
    TV proximal operator via soft-thresholding on image differences.
    Approximate anisotropic TV prox:
        prox_{lam TV}(x) = x - div( soft_threshold(grad(x), lam) )
    
    Args:
        x (torch.Tensor): Input image, shape (B,C,H,W) or (C,H,W) or (H,W)
        lam (float): Regularization weight

    Returns:
        torch.Tensor: Denoised output
    """
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    elif x.dim() == 3:
        x = x.unsqueeze(0)  # (1,C,H,W)

    B, C, H, W = x.shape

    # Compute gradients
    dx = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))  # vertical
    dy = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))  # horizontal

    # Soft-threshold gradients
    dx_th = soft_threshold(dx, lam)
    dy_th = soft_threshold(dy, lam)

    # Compute divergence (adjoint of gradient)
    ddx = dx_th - F.pad(dx_th[:, :, :-1, :], (0, 0, 1, 0))
    ddy = dy_th - F.pad(dy_th[:, :, :, :-1], (1, 0, 0, 0))
    div = ddx + ddy

    # Proximal update
    x_prox = x - div

    return x_prox.squeeze().detach()
def tv_prox_chambolle(x, lam=0.1, n_iter=10):
    """
    Proximal operator for anisotropic Total Variation using Chambolle's algorithm.
    
    Args:
        x (torch.Tensor): Input image (B, C, H, W) or (C, H, W) or (H, W)
        lam (float): Regularization parameter
        n_iter (int): Number of iterations
    
    Returns:
        torch.Tensor: Denoised output via TV proximity operator
    """
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif x.dim() == 3:
        x = x.unsqueeze(0)               # (1, C, H, W)

    B, C, H, W = x.shape
    p = torch.zeros((B, C, 2, H, W), device=x.device)
    with torch.no_grad():
        for _ in range(n_iter):
            # divergence
            div = (
                F.pad(p[:, :, 0, 1:, :], (0, 0, 0, 1)) - F.pad(p[:, :, 0, :-1, :], (0, 0, 1, 0)) +
                F.pad(p[:, :, 1, :, 1:], (0, 1, 0, 0)) - F.pad(p[:, :, 1, :, :-1], (1, 0, 0, 0))
            )
            u = x - lam * div

            # gradients
            grad_x = u[:, :, 1:, :] - u[:, :, :-1, :]
            grad_y = u[:, :, :, 1:] - u[:, :, :, :-1]
            grad_x = F.pad(grad_x, (0, 0, 0, 1))
            grad_y = F.pad(grad_y, (0, 1, 0, 0))

            norm = torch.maximum(torch.ones_like(grad_x), torch.sqrt(grad_x**2 + grad_y**2))
            p[:, :, 0] = (p[:, :, 0] + (1.0 / (8 * lam)) * grad_x) / norm
            p[:, :, 1] = (p[:, :, 1] + (1.0 / (8 * lam)) * grad_y) / norm

        # final result
        div = (
            F.pad(p[:, :, 0, 1:, :], (0, 0, 0, 1)) - F.pad(p[:, :, 0, :-1, :], (0, 0, 1, 0)) +
            F.pad(p[:, :, 1, :, 1:], (0, 1, 0, 0)) - F.pad(p[:, :, 1, :, :-1], (1, 0, 0, 0))
        )
        u = x - lam * div
    return u.squeeze().detach()

