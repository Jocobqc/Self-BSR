import os
import time
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, opt):
        self.opt = opt 
        self.name = opt.name
        self.saved = False
        self.current_epoch = 0
        self.experimrnt_dir = opt.experimrnt_dir
        self.checkpoints_dir = opt.checkpoints_dir
        self.loss_list = {}
        self.loss_list['total'] = []
        for loss in opt.loss_type:
            self.loss_list[loss] = []
        self.log_name = os.path.join(self.experimrnt_dir, 'log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def plot_current_losses(self, losses):
        plt.figure()
        self.loss_list['total'].append(losses['total'].item()) 
        steps = range(1, len(self.loss_list['total']) + 1, 1)
        plt.plot(steps, self.loss_list['total'], label='total')            
        for loss in self.opt.loss_type:
            self.loss_list[loss].append(losses[loss].item())                      
            plt.plot(steps, self.loss_list[loss], label=loss)  
        plt.title("Loss Curves")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.experimrnt_dir, 'loss.png'))
        plt.close()            

    def print_current_losses(self, epoch, iters, losses):
        message = '(epoch: %d, iters: %d) ' % (epoch, iters)

        for k, v in losses.items():
            message += '%s: %.8f ' % (k, v)    
        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def print_logs(self, message, do_print=False):
        if do_print:
            print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
