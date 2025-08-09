import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

class Self_BSR(nn.Module):
    def __init__(self, opt, refine=False):
        super().__init__()
        self.opt = opt
        self.refine = refine      
        in_ch=self.opt.input_ch
        out_ch=self.opt.input_ch
        base_ch=self.opt.base_ch
        num_module=self.opt.bnum
        self.head = nn.Sequential(
                                nn.Conv2d(in_ch, base_ch, kernel_size=1),
                                nn.ReLU(inplace=True),)
        self.maskconv_I = nn.Sequential(
                        CentralMaskedConv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1),            
                        nn.ReLU(inplace=True),
                        nn.Conv2d(base_ch, base_ch, kernel_size=1),                                
                        nn.ReLU(inplace=True),
                        nn.Conv2d(base_ch, base_ch, kernel_size=1),
                        nn.ReLU(inplace=True),
                        )
        self.branch_I = nn.Sequential( *[ResBlock(base_ch*4) for _ in range(num_module)],)
        self.branch_S = nn.Sequential( *[ResBlock(base_ch) for _ in range(num_module)],)  
        self.S2I = nn.Sequential( *[DCl(2, base_ch) for _ in range(2)],)              
        self.branchI_tail = nn.Sequential(
                        nn.Conv2d(base_ch, base_ch, kernel_size=1),
                        nn.ReLU(inplace=True),
                        )
        self.branchS_tail = nn.Sequential(
                        nn.Conv2d(base_ch, base_ch, kernel_size=1),
                        nn.ReLU(inplace=True),
                        )                   
        self.tail_I = nn.Sequential(
                                nn.Conv2d(base_ch,  base_ch,    kernel_size=1),
                                nn.ReLU(inplace=True),                                
                                nn.Conv2d(base_ch,  base_ch//2,    kernel_size=1),
                                nn.ReLU(inplace=True),                                
                                nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(base_ch//2,  out_ch, kernel_size=1),                                 
                                )
        self.tail_S = nn.Sequential(
                        nn.Conv2d(base_ch,  base_ch,    kernel_size=(3,1), stride=1, padding=(1,0)),
                        nn.ReLU(inplace=True),                                
                        nn.Conv2d(base_ch,  base_ch//2,    kernel_size=(3,1), stride=1, padding=(1,0)),
                        nn.ReLU(inplace=True),                                
                        nn.Conv2d(base_ch//2, base_ch//2, kernel_size=(3,1), stride=1, padding=(1,0)),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(base_ch//2,  out_ch, kernel_size=(3,1), stride=1, padding=(1,0)),                                 
                        )
        self.RWT = waveletDecomp(stride=1, c_channels=base_ch)
    def forward(self, x): 
        x_h = self.head(x) 
        # FRR     
        if self.refine:
            x_h = F.interpolate(x_h,scale_factor=(2,1.6),mode='bilinear')                      
        x_mask = self.maskconv_I(x_h)
        if self.refine:
            x_mask = F.interpolate(x_mask,scale_factor=(1/2,1/1.6),mode='bilinear')  

        x_mask_pd = pixel_shuffle_down_sampling_pd(x_mask, f=2, pad=2)
        x_pd_w,x_pd_w_HL = self.RWT.forward(x_mask_pd)

        I_pd_b_w = self.branch_I(x_pd_w)
        S_pd_b_w_HL = self.branch_S(x_pd_w_HL) 
        I_pd_b_ = self.RWT.inverse(I_pd_b_w)

        S2I = self.S2I(S_pd_b_w_HL)
        I_pd_b_en = I_pd_b_ + S2I

        I_pd_b = self.branchI_tail(I_pd_b_en) 
        S_pd_b = self.branchS_tail(S_pd_b_w_HL)  

        I_b = pixel_shuffle_up_sampling_pd(I_pd_b, f=2, pad=2)
        S_b = pixel_shuffle_up_sampling_pd(S_pd_b, f=2, pad=2)

        stripe = self.tail_S(S_b)
        stripe_1d = torch.sum(stripe,dim=-2,keepdim=True)

        img_clean = self.tail_I(I_b)
        
        return img_clean,stripe_1d

class ResBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        ly = []
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)

class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()
        ly = []
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)

class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH//2, kH//2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class waveletDecomp(nn.Module):
    def __init__(self, stride=2, c_channels=1):
        super(waveletDecomp, self).__init__()

        self.stride = stride
        self.c_channels = c_channels

        wavelet = pywt.Wavelet('haar')
        dec_hi = torch.tensor(wavelet.dec_hi[::-1])
        dec_lo = torch.tensor(wavelet.dec_lo[::-1])

        self.filters_dec = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                                        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                                        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                                        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0).cuda()
        self.filters_dec = self.filters_dec.unsqueeze(1)
        self.filters_dec = self.filters_dec.repeat((self.c_channels,1,1,1))
        self.psize = int(self.filters_dec.size(3) / 2)

        rec_hi = torch.tensor(wavelet.rec_hi[::-1])
        rec_lo = torch.tensor(wavelet.rec_lo[::-1])
        self.filters_rec = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                                        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                                        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                                        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0).cuda()
        self.filters_rec = self.filters_rec.unsqueeze(0)
        self.filters_rec = self.filters_rec.repeat((self.c_channels,1,1,1))  
        self.filters_rec_transposed = torch.flip(self.filters_rec.permute(1, 0, 2, 3), [2, 3])

    def forward(self, x):
        if self.stride == 1:
            x = F.pad(x, (self.psize - 1, self.psize, self.psize - 1, self.psize), mode='replicate')
        coeff = F.conv2d(x, self.filters_dec, stride=self.stride, bias=None, padding=0, groups=self.c_channels)
        
        out = coeff / 2
        HL = out[:, 2::4, :, :].contiguous()
        return out,HL

    def inverse(self, x):
        if self.stride == 1:
            x = F.pad(x, (self.psize, self.psize - 1, self.psize, self.psize - 1), mode='replicate')

        if self.stride == 1:       
            coeff = F.conv2d(x, self.filters_rec, stride=self.stride, bias=None, padding=0, groups=self.c_channels)
        else:
            coeff = F.conv_transpose2d(x, self.filters_rec_transposed, stride=self.stride,
                                                       bias=None, padding=0)
        out = coeff

        return (out * self.stride ** 2) / 2


def pixel_shuffle_down_sampling_pd(x: torch.Tensor, f: int, pad: int = 0, pad_value: float = 0.):
    b, c, w, h = x.shape
    unshuffled = F.pixel_unshuffle(x, f)
    if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), 'reflect')
    unshuffled = unshuffled.view(b, c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 2, 3, 1, 4, 5).contiguous()
    unshuffled = unshuffled.view(-1, c, w // f + 2 * pad, h // f + 2 * pad).contiguous()
    return unshuffled
        
def pixel_shuffle_up_sampling_pd(x: torch.Tensor, f: int, pad: int = 0):
    b, c, w, h = x.shape
    b = b // (f * f)
    before_shuffle = x.view(b, f, f, c, w, h)
    before_shuffle = before_shuffle.permute(0, 3, 1, 2, 4, 5).contiguous()
    before_shuffle = before_shuffle.view(b, c*f*f, w, h)
    if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
    return F.pixel_shuffle(before_shuffle, f)