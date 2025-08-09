import torch.nn as nn
import torch
import random
class calculate_multi_loss(nn.Module):    	
    def __init__(self, opt):
        super(calculate_multi_loss, self).__init__()
        self.opt = opt
        self.loss_type = opt.loss_type
        self.criterion_L1 = nn.L1Loss()
        self.criterion_MSE = nn.MSELoss() 
        
    def forward(self,  noise, denoise, stripe_1d):
        loss_dict = {}
        loss_total = 0       
        if 'MSE_S' in self.opt.loss_type:
            stripe_mean = torch.mean(stripe_1d,dim=(-2,-1),keepdim=True)
            stripe_zero = torch.zeros_like(stripe_mean)
            loss = self.criterion_MSE(stripe_mean,stripe_zero)*1e-2
            loss_dict['MSE_S'] = loss
            loss_total += loss    
        if 'TVx_Ip' in self.opt.loss_type:
            loss = 0
            n,c,h,w = denoise.shape
            for _ in range(self.opt.patch_num):
                w_offset = random.randint(0, max(0, w - self.opt.patch_size - 1))
                h_offset = random.randint(0, max(0, h - self.opt.patch_size - 1))
                loss += tv_loss_x(denoise[:,:, h_offset:h_offset + self.opt.patch_size,w_offset:w_offset + self.opt.patch_size])
            loss = loss*3e-3
            loss_dict['TVx_Ip'] = loss
            loss_total += loss                            
        if 'MSE_I_S' in self.opt.loss_type:
            n,c,h,w = denoise.shape
            stripe_2d = stripe_1d.repeat((1,1,h,1))
            loss = self.criterion_MSE(denoise+stripe_2d,noise)*1
            loss_dict['MSE_I_S'] = loss
            loss_total += loss                                                                                     
        loss_dict['total'] = loss_total                                         
        return loss_dict

def tv_loss_x(A):
    delta_w = A[:, :, :, 1:] - A[:, :, :, :-1]
    tv = delta_w.abs().mean((2, 3))
    loss = torch.mean(tv.sum(1) /A.shape[1])
    return loss