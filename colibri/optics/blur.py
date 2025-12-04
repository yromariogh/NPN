import torch
from colibri.optics.functional import psf_single_doe_spectral, convolutional_sensing, wiener_filter, ideal_panchromatic_sensor
from colibri.optics.sota_does import conventional_lens, nbk7_refractive_index
from .utils import BaseOpticsLayer


import torch.nn.functional as F

class ConvolutionSR(BaseOpticsLayer):
    def __init__(self, kernel: torch.Tensor, RF: int = 1):
        """
        Args:
            kernel: PSF / kernel tensor (assumed shape compatible with convolutional_sensing / wiener_filter)
            RF: resolution factor. After sensing we downsample by RF; before deconvolution we upsample by RF.
        """
        self.kernel = kernel
        self.RF = RF
        super(ConvolutionSR, self).__init__(
            learnable_optics=kernel,
            sensing=self.convolution,
            backward=self.deconvolution,
        )

    def _downsample(self, x: torch.Tensor):
        if self.RF == 1:
            return x
        # Downsample spatial dims by factor RF. Assume x shape (B, C, H, W)
        return F.interpolate(x, scale_factor=1 / self.RF, mode="bilinear", align_corners=False, recompute_scale_factor=True)

    def _upsample(self, x: torch.Tensor):
        if self.RF == 1:
            return x
        return F.interpolate(x, scale_factor=self.RF, mode="bilinear", align_corners=False)

    def convolution(self, x, kernel):
        r"""
        Forward operator: apply sensing and then (non-trainable) downsampling by RF.

        Args:
            x (torch.Tensor): Input tensor with shape (B, L, M, N)
            kernel: convolution kernel
        Returns:
            torch.Tensor: downsampled field tensor
        """
        field = convolutional_sensing(x, kernel, domain="fourier")
        field_ds = self._downsample(field)
        return field_ds

    def deconvolution(self, x, kernel, alpha=1e-3):
        r"""
        Backward operator: upsample input by RF (non-trainable) then apply Wiener filter.

        Args:
            x (torch.Tensor): Input tensor with shape (B, 1, Mh, Nh) where Mh,Nh are possibly reduced
        Returns:
            torch.Tensor: Reconstructed tensor with shape (B, L, M, N)
        """
        x_up = self._upsample(x)
        out = wiener_filter(x_up, kernel, alpha)
        return out
    
class Convolution(BaseOpticsLayer):
    def __init__(self, kernel: torch.Tensor):
        # Store kernel as a buffer (not trainable, but moves with model to device)
        self.kernel = kernel
        # self.learnable_optics = kernel
        super(Convolution, self).__init__(learnable_optics=kernel, sensing=self.convolution, backward=self.deconvolution)

    def convolution(self, x, kernel):
        r"""
        Forward operator of the SingleDOESpectral layer.

        Args:
            x (torch.Tensor): Input tensor with shape (B, L, M, N)
        Returns:
            torch.Tensor: Output tensor with shape (B, 1, M, N) 
        """

        # psf = self.kernel
        field = convolutional_sensing(x, kernel, domain='fourier')
        return field


    def deconvolution(self, x, kernel, alpha=1e-3):
        r"""
        Backward operator of the SingleDOESpectral layer.

        Args:
            x (torch.Tensor): Input tensor with shape (B, 1, M, N)
        Returns:
            torch.Tensor: Output tensor with shape (B, L, M, N) 
        """

        # psf = self.kernel
        # print(kernel.shape)
        # out = torch.nn.functional.conv_transpose2d(x, kernel,padding=((kernel.shape[2]-1)//2))
        # # print(out.shape)
        # out = out[:,:,:x.shape[2],:x.shape[3]]
        # # print(out.shape)
        out = wiener_filter(x, kernel, alpha)
        return out
    
    
    def forward(self, x, type_calculation="forward"):
        r"""
        Performs the forward or backward operator according to the type_calculation

        Args:
            x (torch.Tensor): Input tensor with shape (B, L, M, N)
            type_calculation (str): String, it can be "forward", "backward" or "forward_backward"
        Returns:
            torch.Tensor: Output tensor with shape (B, L, M, N) 
        Raises:
            ValueError: If type_calculation is not "forward", "backward" or "forward_backward"
        """

        return super(Convolution, self).forward(x, type_calculation)


