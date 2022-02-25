import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

def drawImage(img, scale=2):
    img = img.cpu().detach().clamp(min=0, max=1)
    if len(img.shape)==4 :
        plt.figure(figsize=(img.shape[0]*scale, scale))
        for i in range(img.shape[0]):
            plt.subplot(1,img.shape[0], i+1); plt.imshow(img[i].permute(1,2,0), interpolation='nearest');
            plt.xticks([]); plt.yticks([]);
        plt.show()
    else:
        plt.imshow(img.permute(1,2,0)); plt.xticks([]); plt.yticks([]);

class Residual(nn.Module):
	def __init__(self, fn):
		super().__init__()
		self.fn = fn

	def forward(self, x):
		return self.fn(x) + x

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

def ConvNeXt(dim, kernel_size, dim_mult=2, groups=2):
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

class Tokenizer2d(nn.Module):
    def __init__(self, d_in, n_token, dim, depth, patch_size):
        super().__init__()
        self.n_token, self.d_in, self.patch_size = n_token, d_in, patch_size
        self.eps = 1e-6
        self.enc = nn.Sequential(
            nn.Conv2d(d_in, dim, kernel_size=patch_size, stride=patch_size),
            * [ConvNeXt(dim=dim, kernel_size=1) for i in range(depth) ],
            nn.GELU(), nn.Conv2d(dim, n_token, kernel_size=1)
        )
        self.dec = nn.Sequential(
            nn.Conv2d(n_token, dim, kernel_size=1), nn.GELU(),
            * [ConvNeXt(dim=dim, kernel_size=5) for i in range(depth) ],
            nn.GELU(),
            nn.ConvTranspose2d(dim, d_in, kernel_size=patch_size, stride=patch_size)
        )
        n_param = 0
        for p in self.parameters():
            n_param += np.prod(p.shape)
        print('# of params : ', n_param)
        
    def device(self):
        for p in self.parameters():
            return p.device
        
    def forward(self, X, training=False):
        prob = self.enc(X.to(self.device())).softmax(dim=1)
        idx = prob.argmax(dim=1, keepdim=True)
        onehot = torch.zeros_like(prob)
        onehot.scatter_(1, idx, 1)
        if not training:
            return onehot
        else:
            z = onehot + prob - prob.detach()
            loss = F.mse_loss(self.dec(z), X)# - ( onehot * (prob*(1-self.eps)+0.5*self.eps).log() ).mean()
            return z, loss
    
    def fit(self, X, n_iter, batch_size=16, draw_every=1000):
        self.train()
        opt = torch.optim.AdamW(self.parameters(), lr=0.0001)
        loader = DataLoader(X, batch_size=batch_size, shuffle=True)
        for n in range(n_iter):
            for i, x in enumerate(loader):
                opt.zero_grad()
                x = x.to(self.device())
                z, loss = self(x, training=True)
                loss.backward()
                opt.step()
                if draw_every != None:
                    if i%draw_every==0 and i:
                        print(i, loss.item())
                       	self.eval()
                        drawImage(x[:10])
                        drawImage(self.dec(self(x[:10])))
                        self.train()
        self.eval()