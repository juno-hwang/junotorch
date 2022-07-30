import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch.nn.utils import spectral_norm
from junotorch.vision import *


def SpecConv2d( *args, **kwargs):
    return spectral_norm(nn.Conv2d( *args, **kwargs))
    
def SpecConvTranspose2d( *args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d( *args, **kwargs))
        
class SpecResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fn = nn.Sequential(
            nn.GroupNorm(dim//4, dim), nn.GELU(), SpecConv2d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(dim//4, dim), nn.GELU(), SpecConv2d(dim, dim, kernel_size=3, padding=1)
        )
        self.proj = SpecConv2d(dim, dim, kernel_size=1)
        
    def forward(self, x):
        return (self.fn(x) + self.proj(x)) / np.sqrt(2)
    
def SpecConvMixer(dim, depth, kernel_size=9, patch_size=7, d_in=3, d_out=2):
    return nn.Sequential(
        SpecConv2d(d_in, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.GroupNorm(dim//4, dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    SpecConv2d(dim, dim, kernel_size, groups=dim, padding=kernel_size//2),
                    nn.GELU(),
                    nn.GroupNorm(dim//4, dim)
                )),
                SpecConv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.GroupNorm(dim//4, dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, d_out)
    )

def SpecConvNeXt(dim, kernel_size, dim_mult=4):
    return Residual(nn.Sequential(
                SpecConv2d(dim, dim * dim_mult, kernel_size=1),
                nn.GroupNorm(dim*dim_mult//4, dim * dim_mult), nn.GELU(),
                SpecConv2d(dim * dim_mult, dim, kernel_size=1),
                SpecConv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim),
            ))

def SpecDownsampler(d_in, d_out, window_size, stride=2):
    return nn.Sequential(
            SpecConvNeXt( d_in, kernel_size=window_size),
            nn.GELU(),
            SpecConv2d( d_in, d_out, window_size, stride=2, padding=window_size//2 )
        )

def SpecUpsampler(d_in, d_out, window_size, stride=2):
    return nn.Sequential(
            SpecConvNeXt( d_in, kernel_size=window_size),
            nn.GELU(),
            SpecConvTranspose2d( d_in, d_out, window_size, stride=2, padding=window_size//2 )
        )

class SpecNormUNet(nn.Module):
    def __init__(self, dim, mid_depth, n_downsample, base_dim=128, n_resblock=1, d_in=4, d_out=3):
        super().__init__()
        self.dim = dim
        self.n_downsample =  n_downsample
        base_dim = base_dim
        dims =  [ min(base_dim*2**i, dim) for i in range(n_downsample+1) ]
        dims[-1] = dim
        
        self.enc = SpecConv2d(d_in, base_dim, kernel_size=3, padding=1)
        self.downs = nn.ModuleList([
            nn.Sequential(
                SpecConv2d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1),
                *[SpecResBlock(dims[i+1]) for n in range(n_resblock)]
            )for i in range(self.n_downsample)
        ])
        self.middle = nn.ModuleList([ SpecConvNeXt(dim, kernel_size=9) for i in range(mid_depth) ])
        self.ups = nn.ModuleList([
            nn.Sequential(
                *[SpecResBlock(dims[i+1]) for n in range(n_resblock)],
                SpecConvTranspose2d(dims[i+1], dims[i], kernel_size=4, stride=2, padding=1)
            )for i in range(self.n_downsample)[::-1]
        ])
        self.dec = SpecConv2d(base_dim, 3, kernel_size=1)
        
        n_params = sum( [np.prod(p.shape) for p in self.parameters()] )
        for p in self.parameters():
            print(np.prod(p.shape)/1e6)
        print(f'{n_params/1e6:.1f}M params')

    def forward(self, x):
        self.device = next(self.parameters()).device
        x = x.to(self.device)
        x = self.enc(x)
        for down in self.downs:
            x = down(x)
            xs.append(x)
        for mid in self.middle:
            x = mid(x)
        for up in self.ups:
            x = (x + xs.pop()) / np.sqrt(2)
            x = up(x)
        return self.dec(x)