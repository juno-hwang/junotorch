import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch.nn.utils import spectral_norm
from junotorch.vision import *

class UNeXt(nn.Module):
    def __init__(self, dim, image_size, mid_depth, n_downsample, T, base_dim=128, n_resblock=1, d_in=3):
        super().__init__()
        self.dim, self.T = dim, T
        self.image_size = image_size
        self.n_downsample =  n_downsample
        base_dim = base_dim
        dims =  [ min(base_dim*2**i, dim) for i in range(n_downsample+1) ]
        dims[-1] = dim
        
        self.enc = nn.Conv2d(d_in, base_dim, kernel_size=3, padding=1)
        self.downs = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1),
                nn.Sequential(nn.Linear(dim, dims[i+1]*2), nn.GELU(), nn.Linear(dims[i+1]*2, dims[i+1])),
                nn.Sequential( *[ResBlock(dims[i+1]) for n in range(n_resblock)], ConvNeXt(dims[i+1], kernel_size=9) ),
            ])for i in range(self.n_downsample)
        ])
        self.middle = nn.ModuleList([ AdaConvNeXt(dim, kernel_size=9) for i in range(mid_depth) ])
        self.ups = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential( *[ResBlock(dims[i+1]) for n in range(n_resblock)], ConvNeXt(dims[i+1], kernel_size=9) ),
                nn.Sequential(nn.Linear(dim, dims[i+1]*2), nn.GELU(), nn.Linear(dims[i+1]*2, dims[i+1])),
                nn.ConvTranspose2d(dims[i+1], dims[i], kernel_size=4, stride=2, padding=1)
            ])for i in range(self.n_downsample)[::-1]
        ])
        self.dec = nn.Conv2d(base_dim, 3, kernel_size=3, padding=1)
        
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
        
        x = F.gelu(self.enc(x))
        for down, emb, layer in self.downs:
            x = layer( (down(x) + emb(t)[:,:,None,None] )/np.sqrt(2) )
            xs.append(x)
        for mid in self.middle:
            x = mid(x, t)
        for layer, emb, up in self.ups:
            x = (x + xs.pop() + emb(t)[:,:,None,None] ) / np.sqrt(3)
            x = layer(x)
            x = up(x)
        return self.dec(F.gelu(x))

class UNeXtUpsampler(UNeXt):
    def __init__(self, dim, image_size, small_image_size, mid_depth, n_downsample, T, base_dim=128, n_resblock=1, d_in=6):
        super().__init__(dim, image_size, mid_depth, n_downsample, T, base_dim, n_resblock, d_in)
        self.small_image_size = small_image_size
        n_params = sum( [np.prod(p.shape) for p in self.parameters()] )
        
    def forward(self, x, z, t):
        if type(t) == int:
            t = [t]*x.shape[0]
        self.device = next(self.parameters()).device
        x, t, xs = x.to(self.device), self.t_emb(t), []
        
        z = F.interpolate(z, scale_factor=self.image_size//self.small_image_size, mode='bicubic').to(self.device)
        x = F.gelu(self.enc(torch.cat([x,z], dim=1)))
        for down, emb, layer in self.downs:
            x = layer( (down(x) + emb(t)[:,:,None,None] )/np.sqrt(2) )
            xs.append(x)
        for mid in self.middle:
            x = mid(x, t)
        for layer, emb, up in self.ups:
            x = (x + xs.pop() + emb(t)[:,:,None,None] ) / np.sqrt(3)
            x = layer(x)
            x = up(x)
        return self.dec(F.gelu(x))