import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from .specnorm import *

def drawImage(img, scale=2, interpolation='nearest'):
    img = img.cpu().detach().clamp(min=0, max=1)
    if len(img.shape)==4 :
        plt.figure(figsize=(img.shape[0]*scale, scale))
        for i in range(img.shape[0]):
            plt.subplot(1,img.shape[0], i+1); plt.imshow(img[i].permute(1,2,0), interpolation=interpolation);
            plt.xticks([]); plt.yticks([]);
        plt.show()
    else:
        plt.imshow(img.permute(1,2,0)); plt.xticks([]); plt.yticks([]);

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return (self.fn(x) + x) / 1.414

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fn = nn.Sequential(
            nn.GroupNorm(dim//4, dim), nn.GELU(), nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(dim//4, dim), nn.GELU(), nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        
    def forward(self, x):
        return (self.fn(x) + self.proj(x)) / np.sqrt(2)

class ConditionalResidual(nn.Module):
    def __init__(self, d_condition, d_x, window_size, scale_factor, groups=1):
        super().__init__()
        self.scale_factor=scale_factor
        self.conv = nn.Conv2d(d_condition, d_x, 1, groups=1)
        
    def forward(self, x, condition):
        condition = F.interpolate(condition, scale_factor=self.scale_factor, mode='bicubic')
        return x + self.conv(condition) 

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

def ConvNeXt(dim, kernel_size, dim_mult=4):
    return Residual(nn.Sequential(
                nn.Conv2d(dim, dim * dim_mult, kernel_size=1),
                nn.BatchNorm2d(dim * dim_mult), nn.GELU(),
                nn.Conv2d(dim * dim_mult, dim, kernel_size=1),
                nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim),
            ))

def Downsampler(d_in, d_out, window_size, stride=2):
    return nn.Sequential(
            ConvNeXt( d_in, kernel_size=window_size),
            nn.GELU(),
            nn.Conv2d( d_in, d_out, window_size, stride=2, padding=window_size//2 )
        )

def Upsampler(d_in, d_out, window_size, stride=2):
    return nn.Sequential(
            ConvNeXt( d_in, kernel_size=window_size),
            nn.GELU(),
            nn.ConvTranspose2d( d_in, d_out, window_size, stride=2, padding=window_size//2 )
        )

class AdaGN(nn.Module):
    def __init__(self, dim_in, dim_out, chunk):
        super().__init__()
        self.gn = nn.GroupNorm(dim_in//chunk, dim_out)
        self.mean = nn.Sequential(nn.Linear(dim_in, dim_out), nn.GELU(), nn.Linear(dim_out, dim_out))
        self.std = nn.Sequential(nn.Linear(dim_in, dim_out), nn.GELU(), nn.Linear(dim_out, dim_out))
        
    def forward(self, x, c):
        mu = self.mean(c)
        sig = self.std(c)
        return (sig.exp()[:,:,None,None] * self.gn(x) + mu[:,:,None,None])

class AdaConvNeXt(nn.Module):
    def __init__(self, dim, kernel_size, dim_mult=4):
        super().__init__()
        self.dim = dim
        self.adagn = AdaGN(dim, dim*dim_mult, chunk=4)
        self.conv1 = nn.Conv2d(dim, dim * dim_mult, kernel_size=1)
        self.conv2 = nn.Conv2d(dim * dim_mult, dim, kernel_size=1)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.residual = nn.Conv2d(dim, dim, kernel_size=1)
        
    def forward(self, x, c):
        y = self.conv1(x)
        y = F.gelu(self.adagn(y, c))
        y = self.conv3(self.conv2(y))
        return (self.residual(x) + y) / 1.414

class EfficientUNet(nn.Module):
    def __init__(self, dim, image_size, mid_depth, n_downsample, T, base_dim=128, n_resblock=1):
        super().__init__()
        self.dim, self.T = dim, T
        self.image_size = image_size
        self.n_downsample =  n_downsample
        base_dim = base_dim
        dims =  [ min(base_dim*2**i, dim) for i in range(n_downsample+1) ]
        dims[-1] = dim
        
        self.enc = nn.Conv2d(3, base_dim, kernel_size=3, padding=1)
        self.downs = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1),
                nn.Sequential(nn.Linear(dim, dims[i+1]*2), nn.GELU(), nn.Linear(dims[i+1]*2, dims[i+1])),
                nn.Sequential( *[ResBlock(dims[i+1]) for n in range(n_resblock)] ),
            ])for i in range(self.n_downsample)
        ])
        self.middle = nn.ModuleList([ AdaConvNeXt(dim, kernel_size=9) for i in range(mid_depth) ])
        self.ups = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential( *[ResBlock(dims[i+1]) for n in range(n_resblock)] ),
                nn.Sequential(nn.Linear(dim, dims[i+1]*2), nn.GELU(), nn.Linear(dims[i+1]*2, dims[i+1])),
                nn.ConvTranspose2d(dims[i+1], dims[i], kernel_size=4, stride=2, padding=1)
            ])for i in range(self.n_downsample)[::-1]
        ])
        self.dec = nn.Conv2d(base_dim, 3, kernel_size=1)
        
        t = np.linspace(0, np.pi, self.T+1)
        self.t_sinusoid = np.array([ np.cos(t*np.exp(freq))
                           for freq
                           in np.linspace(0, np.log(self.T), self.dim)
                          ]).T.tolist()
        
        n_params = sum( [np.prod(p.shape) for p in self.parameters()] )
        print(f'{n_params/1e6:.1f}M params')
        
    def t_emb(self, t):
        return torch.Tensor( [self.t_sinusoid[i] for i in t] ).to(self.device)
        
    def forward(self, x, t):
        if type(t) == int:
            t = [t]*x.shape[0]
        self.device = next(self.parameters()).device
        x, t, xs = x.to(self.device), self.t_emb(t), []
        
        x = self.enc(x)
        for down, emb, layer in self.downs:
            x = layer( (down(x) + emb(t)[:,:,None,None] )/np.sqrt(2) )
            xs.append(x)
        for mid in self.middle:
            x = mid(x, t)
        for layer, emb, up in self.ups:
            x = (x + xs.pop() + emb(t)[:,:,None,None] ) / np.sqrt(3)
            x = layer(x)
            x = up(x)
        return self.dec(x)

class EfficientUNetUpsampler(EfficientUNet):
    def __init__(self, dim, image_size, small_image_size, mid_depth, n_downsample, T, base_dim=128, n_resblock=1):
        super().__init__(dim, image_size, mid_depth, n_downsample, T, base_dim, n_resblock)
        self.enc = nn.Conv2d(6, base_dim, kernel_size=3, padding=1)
        self.small_image_size = small_image_size
        n_params = sum( [np.prod(p.shape) for p in self.parameters()] )
        
    def forward(self, x, z, t):
        if type(t) == int:
            t = [t]*x.shape[0]
        self.device = next(self.parameters()).device
        x, t, xs = x.to(self.device), self.t_emb(t), []
        
        z = F.interpolate(z, scale_factor=self.image_size//self.small_image_size, mode='bicubic').to(self.device)
        x = self.enc(torch.cat([x,z], dim=1))
        for down, emb, layer in self.downs:
            x = layer( (down(x) + emb(t)[:,:,None,None] )/np.sqrt(2) )
            xs.append(x)
        for mid in self.middle:
            x = mid(x, t)
        for layer, emb, up in self.ups:
            x = (x + xs.pop() + emb(t)[:,:,None,None] ) / np.sqrt(3)
            x = layer(x)
            x = up(x)
        return self.dec(x)