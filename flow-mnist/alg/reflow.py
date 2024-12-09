


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# safe import
import os
import sys

import torch.optim.optimizer
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the model directory
model_dir = os.path.join(current_dir, '..', 'model')
# Add the model directory to sys.path
if model_dir not in sys.path:
    sys.path.append(model_dir)
from model import *

from data  import *
from script.helpers import *
import hydra
from omegaconf import DictConfig
import time

class ReFlow(nn.Module):
    def __init__(self, device, data_shape, model, train_cfg):
        super(ReFlow, self).__init__()
        
        self.device=device
        
        self.data_shape=tuple(data_shape)
        
        self.model = model
        
        self.n_epochs= train_cfg.n_epochs
        
        self.eval_interval=train_cfg.eval_interval
        
        self.num_steps = train_cfg.n_steps
        
        self.batch_size = train_cfg.batch_size
        
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=train_cfg.lr)
        
        self.scheduler = CosineAnnealingWarmupLR(
                    self.optimizer,
                    warmup_epochs=train_cfg.warmup_epochs,
                    max_epochs=train_cfg.max_epochs,
                    min_lr=train_cfg.min_lr,
                    warmup_start_lr=train_cfg.warmup_start_lr
                    )
    def generate_target(self, x1, cls):
        '''
        inputs:
            x1. tensor. real data
            cls. class label
        outputs:
            (xt, t, cls): tuple, the inputs to the model. containing...
                xt: corrupted data. torch.  torch.Size([N,C,H,W])
                t:  corruption ratio        torch.Size([N])
                cls. class label            torch.Size([N])
            v:  tensor. target velocity, from x0 to x1 (v=x1-x0). the desired output of the model. torch.Size([N, C, H, W])
        '''
        # random time, or mixture ratio between (0,1). different for each sample, but he same for each channel. 
        t=torch.randn(self.batch_size,device=self.device)
        t_broadcast=torch.ones_like(x1, device=self.device) * t.view(self.batch_size, 1, 1, 1)
        # generate random noise
        x0=torch.randn(x1.shape, dtype=torch.float32, device=self.device)
        # generate corrupted data
        xt= t_broadcast* x1 + (1-t_broadcast)* x0
        # generate target
        v=x1-x0

        # print(f"xt.shape={xt.shape}")
        # print(f"t.shape={t.shape}")
        # print(f"cls.shape={cls.shape}")
        
        return (xt, t, cls), v
        
    @torch.no_grad()
    def sample(self, cls, num_steps:int, record_intermediate=False):
        '''
        inputs:
            cls: label
            num_step: number of denoising steps in a single generation. 
            record_intermediate: whether to return predictions at each step
        outputs:
            if `record_intermediate` is False, xt. tensor of shape `self.data_shape`
            if `record_intermediate` is True,  xt_list. tensor of shape `[num_steps,self.data_shape]`
        '''
        
        if record_intermediate:
            x_hat_list=torch.zeros((num_steps,)+self.data_shape)  # [num_steps,self.data_shape]
        
        x_hat=torch.randn((self.batch_size,)+self.data_shape, device=self.device)    # [batchsize, C, H, W]
        
        dt = (1/num_steps)* torch.ones_like(x_hat).to(self.device)
        
        steps = torch.linspace(0,1,num_steps).repeat(self.batch_size, 1).to(self.device)                       # [batchsize, num_steps]
    
        for i in range(num_steps):
            t = steps[:,i]
            vt=self.model(x_hat,t,cls)
            x_hat+= vt* dt
            
            if record_intermediate:
                x_hat_list[i] = x_hat
        # print(f"x_hat.shape={x_hat.shape}")
        return x_hat_list if record_intermediate else x_hat

    def loss(self, xt, t, cls, v):
        
        v_hat = self.model(xt, t, cls)
        
        loss = F.mse_loss(input=v_hat, target=v)
        
        return loss 
        
        
    def run(self, train_loader: DataLoader, test_loader: DataLoader):
        train_losses = []
        eval_losses = []
        bc_losses = []
        log_dir = os.path.join(BASE_DIR,'log', current_time())
        image_dir = os.path.join(log_dir,'visualize')
        ckpt_dir = os.path.join(log_dir,'ckpt')
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        for epoch in tqdm(range(1, self.n_epochs+1, 1)):
            start_time = time.time()  # Start time for the epoch

            self.model.train()
            epoch_train_loss = []
            
            for (x, cls) in train_loader:
                (xt, t, cls), v = self.generate_target(x, cls)

                self.optimizer.zero_grad()

                loss = self.loss(xt, t, cls, v)
                epoch_train_loss.append(loss.item())
                
                loss.backward()
                self.optimizer.step()

            # Calculate average training loss for this epoch
            train_loss_mean = np.mean(epoch_train_loss)
            train_losses.append(train_loss_mean)
            train_epochs = np.arange(start=1, stop=epoch+1, step=1)
            
            if epoch % self.eval_interval == 0:
                # evaluate model
                err_list = []
                
                eval_loss_list=[]
                self.model.eval()
                with torch.no_grad():
                    for (x, cls) in test_loader:
                        x_hat = self.sample(cls=cls, num_steps=self.num_steps, record_intermediate=False)
                        
                        mse_err = F.mse_loss(x_hat, x)
                        err_list.append(mse_err.item())

                        eval_targets, eval_v = self.generate_target(x, cls)
                        eval_loss_list.append(self.loss(*eval_targets, eval_v))
                        
                        
                eval_loss_mean = np.mean(eval_loss_list)
                eval_losses.append(eval_loss_mean)
                
                bc_loss_mean = np.mean(err_list)
                bc_losses.append(bc_loss_mean)
                
                eval_epochs = np.arange(start=self.eval_interval, stop=epoch+1, step=self.eval_interval)
                
                # Calculate stats for logging
                epoch_duration = time.time() - start_time  # Duration of the epoch
                remaining_epochs = self.n_epochs - epoch
                estimated_remaining_time = remaining_epochs * epoch_duration

                # Print logs
                
                print(f'Epoch [{epoch}/{self.n_epochs}], '
                    f'Train Loss: {train_loss_mean:.4f}, '
                    f'Eval Loss: {eval_loss_mean:.4f}, '
                    f'Behavior Cloning: {bc_loss_mean:.4f}, '
                    f'Time: {epoch_duration:.2f}s, '
                    'Estimated Remaining Time:'+format_time_seconds(estimated_remaining_time))

                # plot the loss, eval curves. 
                # print(f"plot!")
                # print(f"train_losses={train_losses}")
                # print(f"eval_losses={eval_losses}")
                # print(f"bc_losses={bc_losses}")
                # print(f"fig_dir={os.path.join(BASE_DIR,'visualize','loss.png')}")
                plt.figure(figsize=(10, 5))
                plt.plot(train_epochs, train_losses, label='Train Loss', color='red')
                plt.plot(eval_epochs, eval_losses, label='Eval Loss', color='black')
                plt.plot(eval_epochs,bc_losses, label='BC Loss', color='blue')
                plt.title('Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig(os.path.join(image_dir, 'loss.png'))
                plt.close()  # Close the figure to free up memory
                    
                # Save model after evaluation
                torch.save(self.model.state_dict(), os.path.join(ckpt_dir, f'model_{epoch}.pth'))

            self.scheduler.step()  # Update the learning rate scheduler
            
        print_summary(image_dir=image_dir, ckpt_dir=ckpt_dir)
