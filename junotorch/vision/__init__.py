import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
	def __init__(self, fn):
		super().__init__()
		self.fn = fn

	def forward(self, x):
		return self.fn(x) + x

class Shuffle(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.dim = dim
		self.permutation = torch.randperm(dim)
		
	def forward(self, x):
		return x[:, self.permutation, ...]

def ConvMixerOriginal(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
	# credit : https://github.com/locuslab/convmixer
	return nn.Sequential(
		nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
		nn.GELU(),
		nn.BatchNorm2d(dim),
		*[nn.Sequential(
				Residual(nn.Sequential(
					nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=kernel_size//2),
					nn.GELU(),
					nn.BatchNorm2d(dim)
				)),
				nn.Conv2d(dim, dim, kernel_size=1),
				nn.GELU(),
				nn.BatchNorm2d(dim)
		) for i in range(depth)],
		nn.AdaptiveAvgPool2d((1,1)),
		nn.Flatten(),
		nn.Linear(dim, n_classes)
	)

def ConvNeXt(dim, kernel_size, dim_mult=4, groups=4):
	if groups == 1:
		return Residual(nn.Sequential(
					nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim//8),
					nn.BatchNorm2d(dim),
					nn.Conv2d(dim, dim*dim_mult, kernel_size=1),
					nn.GELU(),
					nn.Conv2d(dim*dim_mult, dim, kernel_size=1),
				))
	else:
		return Residual(nn.Sequential(
					nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim//8),
					nn.BatchNorm2d(dim),
					nn.Conv2d(dim, dim*dim_mult, kernel_size=1, groups=groups),
					nn.GELU(), Shuffle(dim*dim_mult),
					nn.Conv2d(dim*dim_mult, dim, kernel_size=1, groups=groups),
				))

def Upsampler(d_in, d_out, window_size, stride=2):
	return nn.Sequential(
			ConvNeXt( d_in, kernel_size=window_size),
			nn.GELU(),
			nn.Conv2d( d_in, d_out, window_size, stride=2, padding=window_size//2 )
		)

def Downsampler(d_in, d_out, window_size, stride=2):
	return nn.Sequential(
			ConvNeXt( d_in, kernel_size=window_size),
			nn.GELU(),
			nn.ConvTranspose2d( d_in, d_out, window_size, stride=2, padding=window_size//2 )
		)