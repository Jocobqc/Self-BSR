import os
import torch
from torch.utils.data import DataLoader
from dataset_loader.dataset import dataset_loader
from utils.utils import *
from utils.metrics import *
from options import get_train_options,get_train_dir,print_options
from model.Self_BSR import Self_BSR

if __name__ == '__main__':
    opt, parser = get_train_options()
    get_train_dir(opt)   
    print_options(opt, parser)
    model_option = "best_psnr"
    # model_option = "best_ssim"
    # model_option = "2000"    
    model_path = os.path.join( opt.checkpoints_dir,"epoch_"+model_option+".pth")
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    model = Self_BSR(opt,opt.frr).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    if opt.frr:
        save_path_imgs  = os.path.join( opt.experimrnt_dir,"test_results_frr","epoch_"+model_option+"_result",'imgs')
        save_path_denoise  = os.path.join( opt.experimrnt_dir,"test_results_frr","epoch_"+model_option+"_result",'denoise') 
        save_path_stripe  = os.path.join( opt.experimrnt_dir,"test_results_frr","epoch_"+model_option+"_result",'stripe') 
    else:    
        save_path_imgs  = os.path.join( opt.experimrnt_dir,"test_results","epoch_"+model_option+"_result",'imgs')
        save_path_denoise  = os.path.join( opt.experimrnt_dir,"test_results","epoch_"+model_option+"_result",'denoise') 
        save_path_stripe  = os.path.join( opt.experimrnt_dir,"test_results","epoch_"+model_option+"_result",'stripe')        
    if not os.path.exists(save_path_imgs):
        os.makedirs(save_path_imgs)   
    if not os.path.exists(save_path_denoise):
        os.makedirs(save_path_denoise)  
    if not os.path.exists(save_path_stripe):
        os.makedirs(save_path_stripe)                                                
    test_dataset = dataset_loader(opt, status='test')
    test_loader = DataLoader(dataset=test_dataset, num_workers=1, batch_size=1, shuffle=False)        
    psnr_scene_total = 0
    ssim_scene_total = 0
    for iteration, (clean_image,noise_image,img_name) in enumerate(test_loader):
        save_path_imgs_ = os.path.join(save_path_imgs,img_name[0])
        save_path_denoise_ = os.path.join(save_path_denoise,img_name[0])
        save_path_stripe_ = os.path.join(save_path_stripe,img_name[0])        
        print(save_path_denoise_)
        clean_image = clean_image.to(device)
        noise_image = noise_image.to(device)
        n,c,h,w = clean_image.shape
        with torch.no_grad():
            denosie_img,stripe_1d = model(noise_image)
            stripe_2d = stripe_1d.repeat((1,1,h,1))
            stripe_2d += 0.5
        if iteration<5:
            mutil_imgs = torch.zeros([n,c,h,4*w+15],dtype=clean_image.dtype).to(device) 
            mutil_imgs[:,:,:,0:w] = noise_image
            mutil_imgs[:,:,:,w+5:2*w+5] = stripe_2d
            mutil_imgs[:,:,:,2*w+10:3*w+10] = denosie_img   
            mutil_imgs[:,:,:,3*w+15:4*w+15] = clean_image  
            img_save(mutil_imgs,save_path_imgs_) 
            img_save(denosie_img,save_path_denoise_) 
            img_save(stripe_2d,save_path_stripe_)               
    