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
    def __init__(self, backbone, batch_size, s=1e-3, device='cuda', result_folder='', pretrained_model=None, loss_type='l2'):
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
    def p(self, x, t, q=0.995):
        self.backbone.eval()
        if type(t) == int :
            t = np.array([t] * x.shape[0])
        x = x.to(self.device)
        alpha, alpha_, alpha_m1 = self.extract(self.alpha, t-1), self.extract(self.alpha_, t), self.extract(self.alpha_, t-1)
        beta, beta_ = self.extract(self.beta, t-1), self.extract(self.beta_, t-1)
        sigma = beta_ ** 0.5
        
        noise = self.backbone(x, t)
        x0 = (x - (1-alpha_).sqrt() * noise) / alpha_.sqrt()
        
        # static thresholding
        #x0 = x0.clamp(min=-1, max=1)
        
        #dynamic thresholding
        s = torch.quantile(x0.abs().view(x0.shape[0], x0.shape[1], -1), q, dim=-1)
        s = s.clamp(min=1.0)
        x0 = x0.clamp(min=-s[:,:,None,None], max=s[:,:,None,None]) / s[:,:,None,None]

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
        image_list = [ self.restore(self.q_xt(x,i), i).cpu() for i in [self.T, self.T] ]
        return torch.cat(image_list, dim=0)/2 + 0.5
    
    def fit(self, path, lr=1e-4, grad_accum=1, save_step=1000):
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
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.opt = AdamW(self.backbone.parameters(), lr=lr)
        if self.path:
            self.opt.load_state_dict(torch.load(self.path, map_location='cpu')['opt'])
        history = []
        
        grad_accum_iter = 0
        while True:
            for data, _ in iter(dataloader):
                stt = time.time()
                loss = self.loss(data)/grad_accum
                loss.backward()
                history.append(loss.item())
                grad_accum_iter += 1
                
                if grad_accum_iter == grad_accum :
                    self.opt.step()
                    self.ema.update()
                    self.step += 1
                    grad_accum_iter = 0
                    self.opt.zero_grad()
                    
                print(f'{self.step} step : loss {np.mean(history[-1000*grad_accum:]*grad_accum):.8f} /  {time.time()-stt:.3f}sec')
                if self.step % save_step == 0 and self.step and grad_accum_iter == 0:
                    with self.ema.average_parameters():
                        surfix = '.png' if self.backbone.image_size < 256 else '.jpg'
                        utils.save_image(self.make_test_image(data[:5]),
                             f'{self.result_folder}/sample_{self.step//save_step:04d}_{np.mean(history[-1000*grad_accum:])*grad_accum:.6f}{surfix}',
                             nrow = 5)
                        torch.save({'ema':self.ema.state_dict(), 'opt':self.opt.state_dict()}, self.result_folder + '/model.pt')
                        
class DDPMUpsampler(DDPM):
    def loss(self, x):
        self.backbone.train()
        t = np.random.randint(self.T, size=x.shape[0]) + 1
        z = F.avg_pool2d(x, kernel_size=self.backbone.image_size//self.backbone.small_image_size)
        if random.random() > 0.5 : 
            z = transforms.GaussianBlur(3, sigma=(0.4, 0.6))(z)
        z = z * 0.94868 + torch.randn_like(z) * 0.1 # square sum is 1
        x, noise = self.q_xt(x.to(self.device), t, return_noise=True)
        if self.loss_type == 'l1':
            return (self.backbone(x, z, t)-noise).abs().mean()
        if self.loss_type == 'l2':
            return (self.backbone(x, z, t)-noise).square().mean()
    
    @torch.no_grad()
    def p(self, x, z, t, q=0):
        if type(t) == int :
            t = np.array([t] * x.shape[0])
        self.backbone.eval()
        x, z = x.to(self.device), z.to(self.device)
        alpha, alpha_, alpha_m1 = self.extract(self.alpha, t-1), self.extract(self.alpha_, t), self.extract(self.alpha_, t-1)
        beta, beta_ = self.extract(self.beta, t-1), self.extract(self.beta_, t-1)
        sigma = beta_ ** 0.5
        
        noise = self.backbone(x, z, t)
        x0 = (x - (1-alpha_).sqrt() * noise) / alpha_.sqrt()

        # static thresholding
        #x0 = x0.clamp(min=-1, max=1)
        
        #dynamic thresholding
        s = torch.quantile(x0.abs().view(x0.shape[0], x0.shape[1], -1), q, dim=-1)
        s = s.clamp(min=1.0)
        x0 = x0.clamp(min=-s[:,:,None,None], max=s[:,:,None,None]) / s[:,:,None,None]

        c_x0 = alpha_m1.sqrt() * beta / (1-alpha_)
        c_xt = alpha.sqrt()*(1-alpha_m1)/(1-alpha_)
        mu = c_x0 * x0 + c_xt * x
        return mu + sigma * torch.randn_like(x)
    
    @torch.no_grad()
    def restore(self, x, z, t):
        z = z * 0.94868 + torch.randn_like(z) * 0.1 # square sum is 1
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

class MaskedDDPM(DDPM):
    def q_xt(self, x0, t, mask=None, return_noise=False):
        x0 = x0.to(self.device)
        if mask == None:
            mask = torch.ones(x0.shape[0], 1, x0.shape[2], x0.shape[3])
        mask = mask.to(self.device)
        if type(t) == int :
            t = np.array([t] * x0.shape[0])
        alpha, alpha_ = self.extract(self.alpha, t-1), self.extract(self.alpha_, t)
        noise = torch.randn_like(x0)
        xt = alpha_.sqrt()*x0 + (1-alpha_).sqrt()*noise
        if return_noise:
            return torch.cat([xt*mask+x0*(1-mask), mask], dim=1), noise
        else:
            return torch.cat([xt*mask+x0*(1-mask), mask], dim=1)
    
    def random_mask_(self):
        seed, size = random.random(), self.backbone.image_size
        mask = torch.ones(1, size, size)
        if seed <0.2 :
            return mask
        elif seed<0.3:
            mask[:,:random.randint(1, size-1),:] *= 0
            return mask
        elif seed<0.4:
            mask[:,random.randint(1, size-1):,:] *= 0
            return mask
        elif seed<0.5:
            mask[:,:, :random.randint(1, size-1)] *= 0
            return mask
        elif seed<0.6:
            mask[:,:, random.randint(1, size-1):] *= 0
            return mask
        else :
            a,b = random.randint(0,size), random.randint(0,size)
            wi, wf = min(a,b), max(a,b)
            a,b = random.randint(0,size), random.randint(0,size)
            hi, hf = min(a,b), max(a,b)
            mask[:,wi:wf, hi:hf] *= 0
            if random.random() < 0.2:
                return 1-mask
            else:
                return mask

    def random_mask(self):
        mask = self.random_mask_()
        while True:
            if random.random() < 0.5 :
                return mask.detach()
            else:
                mask = 1-(1-mask)*(1-mask)
        
    def loss(self, x):
        self.backbone.train()
        mask = torch.stack([self.random_mask() for i in range(x.shape[0])]).to(self.device)
        t = np.random.randint(self.T, size=x.shape[0]) + 1
        x, z = self.q_xt(x.to(self.device), t, mask=mask, return_noise=True)
        if self.loss_type == 'l1':
            return ((self.backbone(x, t)-z).abs()*mask).mean()
        if self.loss_type == 'l2':
            if self.backbone.image_size <= 64 :
                return ((self.backbone(x, t)-z).square()*mask).mean()
            else : 
                z_pred = self.backbone(x, t) 
                return ((z_pred-z).square()*mask).mean() + F.avg_pool2d((z_pred-z)*mask, kernel_size=4).square().mean()
        
    @torch.no_grad()
    def p(self, x, t, q=0.0):
        self.backbone.eval()
        if type(t) == int :
            t = np.array([t] * x.shape[0])
        x = x.to(self.device)
        mask = x[:,-1:,:,:]
        alpha, alpha_, alpha_m1 = self.extract(self.alpha, t-1), self.extract(self.alpha_, t), self.extract(self.alpha_, t-1)
        beta, beta_ = self.extract(self.beta, t-1), self.extract(self.beta_, t-1)
        sigma = beta_ ** 0.5
        
        noise = self.backbone(x, t)
        x0 = (x[:,:3] - (1-alpha_).sqrt() * noise) / alpha_.sqrt()
        # static thresholding
        #x0 = x0.clamp(min=-1, max=1)
        
        #dynamic thresholding
        s = torch.quantile(x0.abs().view(x0.shape[0], x0.shape[1], -1), q, dim=-1)
        s = s.clamp(min=1.0)
        x0 = x0.clamp(min=-s[:,:,None,None], max=s[:,:,None,None]) / s[:,:,None,None]

        c_x0 = alpha_m1.sqrt() * beta / (1-alpha_)
        c_xt = alpha.sqrt()*(1-alpha_m1)/(1-alpha_)
        mu = c_x0 * x0 + c_xt * x[:,:3]
        denoised = mu + sigma *torch.randn_like(x0)
        return torch.cat([denoised*mask+x[:,:3]*(1-mask), mask], dim=1)
        
    @torch.no_grad()
    def restore(self, x, t):
        for i in tqdm(range(t,0,-1)):
            x = self.p(x, i)
        return x[:,:3]
    
    def generate(self, n):
        x = torch.randn(n, 3, self.backbone.image_size, self.backbone.image_size)
        return self.inpaint(x, None)
    
    def inpaint(self, x, mask):
        return self.restore(self.q_xt(x, self.backbone.T, mask), self.backbone.T)
    
    def make_test_image(self, x):
        T = self.backbone.T
        mask_vertical = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3])
        mask_horizontal = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3])
        mask_extra = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3])
        mask_vertical[:,:,:x.shape[2]//2,:] *= 0
        mask_horizontal[:,:,:,:x.shape[3]//2] *= 0
        mask_extra [:,:,x.shape[2]//4:x.shape[2]//4*3,x.shape[3]//4:x.shape[3]//4*3] *= 0
        image_list = [
            self.inpaint(x, mask_vertical),
            self.inpaint(x, mask_horizontal),
            self.inpaint(x, mask_extra),
            self.inpaint(x, None)]
        return torch.cat(image_list, dim=0)/2 + 0.5

class MaskedDDPMv2(MaskedDDPM):
    def loss(self, x):
        self.backbone.train()
        eps = 0.01
        mask = torch.stack([self.random_mask() for i in range(x.shape[0])]).to(self.device)
        t = np.random.randint(self.T, size=x.shape[0]) + 1
        xt = self.q_xt(x.to(self.device), t, mask=mask)
        x_recon = self.backbone(xt, t)
        diff = (x_recon - x[:,:3].to(self.device)).square()*mask
        coef = self.extract((self.alpha_+eps) / (1-self.alpha_+eps), t)
        loss = (diff * coef).mean(dim=(1,2,3))
        loss_clip = loss.detach().clamp(min=1)
        return (loss / loss_clip).mean()

    @torch.no_grad()
    def p(self, x, t, q=0.0):
        self.backbone.eval()
        if type(t) == int :
            t = np.array([t] * x.shape[0])
        x = x.to(self.device)
        mask = x[:,-1:,:,:]
        alpha, alpha_, alpha_m1 = self.extract(self.alpha, t-1), self.extract(self.alpha_, t), self.extract(self.alpha_, t-1)
        beta, beta_ = self.extract(self.beta, t-1), self.extract(self.beta_, t-1)
        sigma = beta_ ** 0.5
        
        x0 = self.backbone(x, t)
        c_x0 = alpha_m1.sqrt() * beta / (1-alpha_)
        c_xt = alpha.sqrt()*(1-alpha_m1)/(1-alpha_)
        mu = c_x0 * x0 + c_xt * x[:,:3]
        denoised = mu + sigma *torch.randn_like(x0)
        return torch.cat([denoised*mask+x[:,:3]*(1-mask), mask], dim=1)
