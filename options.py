import argparse
import os

def parse_options(parser):
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--lr_policy',  default="linear", type=str,help='[linear | cosine]')
    parser.add_argument('--seed', type=int, default=123)  
    parser.add_argument('--name', type=str, default='CVC09', help='')
    parser.add_argument('--dataset_path', type=str, default="your path", help='')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--bnum', type=int, default=9, help='the number of res blocks') 
    parser.add_argument('--base_ch', type=int, default=64, help='channels of feature')                  
    parser.add_argument('--input_ch', type=int, default=1, help='3 for RGB and 1 for grayscale')
    parser.add_argument('--output_ch', type=int, default=1, help='3 for RGB and 1 for grayscale')    
    parser.add_argument('--loss_type', type=list, default=['MSE_S','MSE_I_S','TVx_Ip'], help='loss type')  
    parser.add_argument('--loss_weight', type=list, default=['e2','1','3e3'], help='loss weight')    
    parser.add_argument('--patch_size', type=int, default=48, help='Randomly extract `patch_num` patches of size `patch_size` from the reconstructed image I for TV loss.') 
    parser.add_argument('--patch_num', type=int, default=6, help='Randomly extract `patch_num` patches of size `patch_size` from the reconstructed image I for TV loss.')        
    parser.add_argument('--batch_size', type=int, default=8, help="training batch size")
    parser.add_argument('--data_preprocess', type=str, default='crop_Hflip_Vflip',help='training data augmentation')
    parser.add_argument('--crop_size', default=128, type=int, help='training images crop size')
    parser.add_argument('--epochs', type=int, default=2000, help='total number of epochs')
    parser.add_argument('--print_freq', type=int, default=400, help='iter frequency of showing training results on console')
    parser.add_argument('--val_freq', type=int, default=1, help='epoch frequency of saving images at the end of val')
    parser.add_argument('--weight_save_freq', type=int, default=50, help='epoch frequency of saving images at the end of val')
    return parser

def get_train_options():
    parser = argparse.ArgumentParser()
    parser = parse_options(parser)
    opt = parser.parse_args()
    opt.name = opt.name+'-n'+str(opt.bnum)+'c'+str(opt.base_ch)
    for i in range(len( opt.loss_type)):
        opt.name = opt.name+'_'+opt.loss_type[i]+'-'+opt.loss_weight[i]
    opt.name = opt.name+'_'+'b'+str(opt.batch_size)+'c'+str(opt.crop_size)+'e'+str(opt.epochs)  
    return opt, parser

def get_train_dir(opt):
    experimrnt_dir = os.path.join('experiment', opt.name)
    if not os.path.exists(experimrnt_dir):
        os.makedirs(experimrnt_dir)
    opt.experimrnt_dir = experimrnt_dir

    checkpoints_dir = os.path.join('experiment', opt.name,"checkpoints")
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    opt.checkpoints_dir = checkpoints_dir

    training_results_dir = os.path.join('experiment', opt.name,"training_results")
    if not os.path.exists(training_results_dir):
        os.makedirs(training_results_dir)
    opt.training_results_dir = training_results_dir        


def print_options(opt, parser):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    file_name = os.path.join(opt.experimrnt_dir, 'train_opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

