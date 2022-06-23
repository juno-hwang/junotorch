import math, time, glob, random, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm
from torch_ema import ExponentialMovingAverage
import junotorch.vision as v

class DDPM:
    def __init__(self, backbone, batch_size, s=1e-3, device='cuda', result_folder=None, pretrained_model=None):
        self.backbone = backbone
        self.T = self.backbone.T
        self.device = device
        self.backbone.to(device)
        self.result_folder = result_folder
        self.batch_size = batch_size
        self.step = 0
        try :
            os.mkdir(result_folder)
        except:
            pass
        
        f = torch.cos( (torch.arange(0, self.T+1)/self.T + s ) / (1+s) * (np.pi / 2) ) ** 2
        self.alpha_ = f / f[0]
        self.beta = (1 - self.alpha_[1:]/self.alpha_[:-1]).clamp(max=0.99)
        self.alpha = 1 - self.beta
        self.alpha_ = torch.Tensor(np.append(1, np.cumprod(self.alpha)))
        self.beta_ = (1-self.alpha_[:-1])/(1-self.alpha_[1:]) * self.beta
        self.ema = ExponentialMovingAverage(self.backbone.parameters(), decay=0.999)
        
        if len(glob.glob(self.result_folder+'/*.pt')) or pretrained_model is not None:
            if pretrained_model is not None:
                path = glob.glob(self.result_folder+'/*.pt')[0]
            else:
                path = pretrained_model
            pt = torch.load(path)
            self.ema.load_state_dict(pt)
            self.ema.copy_to(self.backbone.parameters())
            self.step = pt['num_updates']
        
    def extract(self, var, t):
        return torch.Tensor([ var[n] for n in t ]).to(self.device)[:,None,None,None]

    def q_xt(self, x0, t, return_noise=False):
        x0 = x0.to(self.device)
        if type(t) == int :
            t = np.array([t] * x0.shape[0])
        alpha, alpha_ = self.extract(self.alpha, t-1), self.extract(self.alpha_, t)
        noise = torch.randn_like(x0)
        x0 = alpha_.sqrt()*x0 + (1-alpha_).sqrt()*noise
        if return_noise:
            return x0, noise
        else:
            return x0
    
    @torch.no_grad()
    def p(self, x, t, mask=None, x_init=None):
        self.backbone.eval()
        x = x.to(self.device)
        if mask is not None:
            x = x*(1-mask) + x_init*mask
        if type(t) == int :
            t = np.array([t] * x.shape[0])
        alpha, alpha_, alpha_m1 = self.extract(self.alpha, t-1), self.extract(self.alpha_, t), self.extract(self.alpha_, t-1)
        beta, beta_ = self.extract(self.beta, t-1), self.extract(self.beta_, t-1)
        sigma = beta_ ** 0.5
        
        noise = self.backbone(x, t)
        x0 = (x - (1-alpha_).sqrt() * noise) / alpha_.sqrt()
        x0 = x0.clamp(min=-1, max=1)
        c_x0 = alpha_m1.sqrt() * beta / (1-alpha_)
        c_xt = alpha.sqrt()*(1-alpha_m1)/(1-alpha_)
        mu = c_x0 * x0 + c_xt * x
        return mu + sigma *torch.randn_like(x)
    
    @torch.no_grad()
    def restore(self, x, t):
        for i in range(t,0,-1):
            x = self.p(x, i)
        return x
    
    @torch.no_grad()
    def inpaint(self, x, t, mask):
        mask = torch.Tensor(mask).to(self.device)
        x = x.to(self.device)
        x_noised = self.q_xt(x, t)
        for i in range(t,0,-1):
            x_noised = self.p(x_noised, i, mask, x)
        return x_noised
    
    def generate(self, n, size=None):
        if size == None :
            size = self.backbone.image_size
        x = torch.randn(n, 3, size, size)
        return self.restore(x, self.T)
    
    def loss(self, x):
        self.backbone.train()
        t = np.random.randint(self.T, size=x.shape[0]) + 1
        x, z = self.q_xt(x.to(self.device), t, return_noise=True)
        return (self.backbone(x, t)-z).square().mean()
    
    def fit(self, path, lr=1e-4, grad_accum=1):
        dataset = ImageFolder(
            path,
            transforms.Compose([
                transforms.Resize(self.backbone.image_size),
                transforms.RandomCrop(self.backbone.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0]==1 else (x[:3] if x.shape[0]==4 else x) )
            ])
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)
        self.opt = AdamW(self.backbone.parameters(), lr=lr)
        history = []
        
        grad_accum_iter = 0
        while True:
            for data, _ in iter(dataloader):
                stt = time.time()
                loss = self.loss(data)/grad_accum
                self.opt.zero_grad()
                loss.backward()
                history.append(loss.item())
                grad_accum_iter += 1
                
                if grad_accum_iter == grad_accum :
                    self.opt.step()
                    self.ema.update()
                    self.step += 1
                    grad_accum_iter = 0
                    
                print(f'{self.step} step : loss {np.mean(history[-1000*grad_accum:]):.8f} /  {time.time()-stt:.3f}sec')
                if self.step % 1000 == 0 and self.step and grad_accum_iter == 0:
                    with self.ema.average_parameters():
                        xx, yy = np.meshgrid(np.arange(self.backbone.image_size), np.arange(self.backbone.image_size))
                        top_mask = xx > self.backbone.image_size/2
                        
                        image_list = [ self.restore(self.q_xt(data[:5],i), i).cpu()
                         for i in [0, int(self.T*0.7), self.T, self.T] ]
                        image_list = torch.cat(image_list, dim=0)/2 + 0.5
                        utils.save_image(image_list,
                             f'{self.result_folder}/sample_{self.step//1000:04d}_{np.mean(history[-1000*grad_accum:]):.6f}.png',
                             nrow = 5)
                        torch.save(self.ema.state_dict(), self.result_folder + '/model.pt')