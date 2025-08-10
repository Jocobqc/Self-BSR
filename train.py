import time
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader.dataset import dataset_loader
from options import get_train_options,get_train_dir,print_options
from utils.utils import *
from utils.metrics import *
from utils.visualizer import Visualizer
from loss.calculate_loss import calculate_multi_loss
from model.Self_BSR import Self_BSR

if __name__ == '__main__':
    opt, parser = get_train_options() 
    get_train_dir(opt)   
    print_options(opt, parser)
    visualizer = Visualizer(opt) 
    setup_seed(opt)
    train_dataset = dataset_loader(opt, status='train')
    val_dataset = dataset_loader(opt, status='val')
    train_loader = DataLoader(dataset=train_dataset, num_workers=16, batch_size=opt.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(dataset=val_dataset, num_workers=16, batch_size=1, shuffle=False, drop_last=False)
    
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    model = Self_BSR(opt,False).to(device)
    print('model parameters: [%.2f] M'%(sum(param.numel() for param in model.parameters())/1e6))
    init_weights(model, init_type='normal', init_gain=0.02)
    optimizer = optim.Adam(model.parameters(),lr=opt.lr,weight_decay=0)
    scheduler = get_scheduler(opt,optimizer) 
    cal_loss = calculate_multi_loss(opt)
    total_iters = 0
    best_psnr_epoch = 0
    best_ssim_epoch = 0
    psnr_best = 0
    ssim_best = 0    
    for epoch in range(1,opt.epochs+1):
 
        epoch_start_time = time.time() 
        model.train()
        total_loss_epoch = 0
        epoch_iter = 0
        for iteration, (noise) in enumerate(train_loader):
            optimizer.zero_grad() 
            noise = noise.to(device)
            denoise,stripe_1d = model(noise)            
            loss_dict = cal_loss(noise, denoise, stripe_1d)                                         
            loss_total = loss_dict['total']             
            loss_total.backward()
            optimizer.step()
            total_loss_epoch += loss_total.item()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size              
            if total_iters % opt.print_freq == 0:
                visualizer.print_current_losses(epoch, epoch_iter, loss_dict)

        scheduler.step()  
        visualizer.plot_current_losses(loss_dict)       
        visualizer.print_logs('End of epoch %d / %d \t lr %.8f \t Train Time: %d sec' % (epoch, opt.epochs, optimizer.param_groups[0]['lr'], time.time() - epoch_start_time),do_print=True)

        psnr_avg = 0
        ssim_avg = 0
        if epoch % opt.val_freq == 0:
            model.eval()
            with torch.no_grad():
                psnr_total = 0
                ssim_total = 0                
                img_num = 0
                val_images = []
                save_img_num = 2
                for iteration, (clean_image,noise_image) in enumerate(val_loader):
                    compare_num += 1
                    clean_image = clean_image.to(device)
                    noise_image = noise_image.to(device)
                    img_denoise,stripe_1d = model(noise_image)
                    n,c,h,w = img_denoise.shape
                    stripe_2d = stripe_1d.repeat((1,1,h,1))
                    stripe_2d += 0.5
                    psnr, ssim = compute_psnr_ssim(clean_image, img_denoise, 1.)
                    psnr_total += psnr
                    ssim_total += ssim 
                    if iteration < save_img_num:                 
                        val_images.extend([noise_image.squeeze(0).data.cpu(), stripe_2d.squeeze(0).data.cpu(), img_denoise.squeeze(0).data.cpu(), clean_image.squeeze(0).data.cpu()])          
                save_mutil_imgs(val_images,opt.training_results_dir,epoch,4,save_img_num) 
                               
                psnr_ave = psnr_total / img_num
                ssim_ave = ssim_total / img_num
                if psnr_ave > psnr_best:
                    psnr_best = psnr_ave
                    best_psnr_epoch = epoch
                    save_checkpoint(model, epoch, opt.checkpoints_dir, save_type="best_psnr") 
                if ssim_ave > ssim_best:
                    ssim_best = ssim_ave
                    best_ssim_epoch = epoch  
                    save_checkpoint(model, epoch, opt.checkpoints_dir, save_type="best_ssim")  
                metrics_msg = '[valid | epoch%d] PSNR: %.4f SSIM: %.4f --- Best_PSNR_epoch %d  Best_PSNR %.4f  Best_SSIM_epoch %d Best_SSIM: %.4f  T+V Time: %d sec'  % \
                        (epoch, psnr_ave, ssim_ave, best_psnr_epoch, psnr_best, best_ssim_epoch, ssim_best,time.time() - epoch_start_time)
                visualizer.print_logs(metrics_msg, do_print=True)
              
        if epoch > (opt.epochs//4-1) and epoch % opt.weight_save_freq == 0:          
            save_checkpoint(model, epoch, opt.checkpoints_dir)
        save_checkpoint(model, 'latest', opt.checkpoints_dir)

