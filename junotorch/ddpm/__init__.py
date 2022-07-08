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

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class DDPM:
    def __init__(self, backbone, batch_size, s=1e-3, device='cuda', result_folder='', pretrained_model=None, loss_type='l1'):
        self.backbone = backbone
        self.T = self.backbone.T
        self.device = device
        self.backbone.to(device)
        self.result_folder = result_folder
        self.batch_size = batch_size
        self.step = 0
        self.loss_type = loss_type
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
        self.ema = ExponentialMovingAverage(self.backbone.parameters(), decay=0.9999)
        
        self.path = None
        if len(glob.glob(self.result_folder+'/*.pt')) or pretrained_model is not None:
            if pretrained_model is None:
                self.path = glob.glob(self.result_folder+'/*.pt')[0]
            else:
                self.path = pretrained_model
            print('Pretrained model found :',self.path)
            pt = torch.load(self.path, map_location='cpu')['ema']
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
    def p(self, x, t):
        self.backbone.eval()
        if type(t) == int :
            t = np.array([t] * x.shape[0])
        x = x.to(self.device)
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
        for i in tqdm(range(t,0,-1)):
            x = self.p(x, i)
        return x
    
    def generate(self, n):
        x = torch.randn(n, 3, self.backbone.image_size, self.backbone.image_size)
        return self.restore(x, self.T)
    
    def loss(self, x):
        self.backbone.train()
        t = np.random.randint(self.T, size=x.shape[0]) + 1
        x, z = self.q_xt(x.to(self.device), t, return_noise=True)
        if self.loss_type == 'l1':
            return (self.backbone(x, t)-z).abs().mean()
        if self.loss_type == 'l2':
            return (self.backbone(x, t)-z).square().mean()
    
    def make_test_image(self, x):
        image_list = [ self.restore(self.q_xt(x,i), i).cpu() for i in [0, int(self.T*0.7), self.T, self.T] ]
        return torch.cat(image_list, dim=0)/2 + 0.5
    
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
        if self.path:
            self.opt.load_state_dict(torch.load(self.path, map_location='cpu')['opt'])
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
                    
                print(f'{self.step} step : loss {np.mean(history[-1000*grad_accum:]*grad_accum):.8f} /  {time.time()-stt:.3f}sec')
                if self.step % 1000 == 0 and self.step and grad_accum_iter == 0:
                    with self.ema.average_parameters():
                        surfix = '.png' if self.backbone.image_size < 256 else '.jpg'
                        utils.save_image(self.make_test_image(data[:5]),
                             f'{self.result_folder}/sample_{self.step//1000:04d}_{np.mean(history[-1000*grad_accum:])*grad_accum:.6f}{surfix}',
                             nrow = 5)
                        torch.save({'ema':self.ema.state_dict(), 'opt':self.opt.state_dict()}, self.result_folder + '/model.pt')
                        
class DDPMUpsampler(DDPM):
    def loss(self, x):
        self.backbone.train()
        t = np.random.randint(self.T, size=x.shape[0]) + 1
        z = F.avg_pool2d(x, kernel_size=self.backbone.image_size//self.backbone.small_image_size)
        if random.random() > 0.5 : 
            z = transforms.GaussianBlur(3, sigma=(0.4, 0.6))(z)
        z = z * 0.968 + torch.randn_like(z) * 0.25 # square sum is 1
        x, noise = self.q_xt(x.to(self.device), t, return_noise=True)
        if self.loss_type == 'l1':
            return (self.backbone(x, z, t)-noise).abs().mean()
        if self.loss_type == 'l2':
            return (self.backbone(x, z, t)-noise).square().mean()
    
    @torch.no_grad()
    def p(self, x, z, t):
        if type(t) == int :
            t = np.array([t] * x.shape[0])
        self.backbone.eval()
        x, z = x.to(self.device), z.to(self.device)
        alpha, alpha_, alpha_m1 = self.extract(self.alpha, t-1), self.extract(self.alpha_, t), self.extract(self.alpha_, t-1)
        beta, beta_ = self.extract(self.beta, t-1), self.extract(self.beta_, t-1)
        sigma = beta_ ** 0.5
        
        noise = self.backbone(x, z, t)
        x0 = (x - (1-alpha_).sqrt() * noise) / alpha_.sqrt()
        x0 = x0.clamp(min=-1, max=1)
        c_x0 = alpha_m1.sqrt() * beta / (1-alpha_)
        c_xt = alpha.sqrt()*(1-alpha_m1)/(1-alpha_)
        mu = c_x0 * x0 + c_xt * x
        return mu + sigma * torch.randn_like(x)
    
    @torch.no_grad()
    def restore(self, x, z, t):
        z = z * 0.968 + torch.randn_like(z) * 0.25 # square sum is 1
        for i in tqdm(range(t,0,-1)):
            x = self.p(x, z, i)
        return x
    
    def upscale(self, z, forget=0.9):
        x = F.interpolate(z, scale_factor=self.backbone.image_size//self.backbone.small_image_size, mode='bicubic')
        x = self.q_xt(x, int(self.T*forget)).to(self.device)
        return self.restore(x, z, int(self.T*forget))
    
    def make_test_image(self, x):
        image_list = [ x, self.upscale(F.avg_pool2d(x, kernel_size=self.backbone.image_size//self.backbone.small_image_size)).cpu() ]
        return torch.cat(image_list, dim=0)/2 + 0.5