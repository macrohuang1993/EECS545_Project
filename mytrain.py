import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import myDnCNN
from dataset import prepare_data, Dataset
from utils import *
from torch.optim.lr_scheduler import MultiStepLR

def main(use_cuda=True,BS=128,lr=1e-3,num_epoch=1,sig=25,save_folder_path='mylogs/test',dict_save_name='model dict',noise2noise=False):
    device = torch.device("cuda" if use_cuda else "cpu")
    writer = SummaryWriter(save_folder_path)

    train_dataset=Dataset(train=True)           
    val_dataset=Dataset(train=False)
    train_loader = DataLoader(dataset=train_dataset, num_workers=4, batch_size=BS, shuffle=True)
    net = myDnCNN(channels=1, num_of_layers=17)
    net.to(device)
    net.apply(weights_init_kaiming)
    criterion=nn.MSELoss(reduction='sum')

    optimizer=optim.Adam(net.parameters(),lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[30], gamma=0.1)

    step=0
    for epoch in range(num_epoch):
        scheduler.step()
        for idx, data in enumerate(train_loader, 0):
            # training step
            net.train()
            img_train = data
            noise =torch.randn_like(img_train)*sig/255               
            img_train,noise=img_train.to(device),noise.to(device) #transfer tensors to GPU if avaiable
            imgn_train = img_train + noise
            imgout_train = net(imgn_train)
            if noise2noise:
                noise_on_target=torch.randn_like(img_train)*sig/255
                noise_on_target=noise_on_target.to(device)
                loss = criterion(imgout_train, img_train+noise_on_target) / (2*imgn_train.shape[0])
            else:
                loss = criterion(imgout_train, img_train) / (2*imgn_train.shape[0])
            loss.backward()
            optimizer.step()
            net.zero_grad()
            
            # results
            net.eval()
            imgout_train = torch.clamp(imgout_train, 0., 1.)
            psnr_train = batch_PSNR(imgout_train, img_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, idx+1, len(train_loader), loss.item(), psnr_train))
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.data[0], step)
                writer.add_scalar('PSNR on training data', psnr_train, step)

            step += 1

        # validate
        psnr_val = 0
        for k in range(len(val_dataset)):
            img_val = torch.unsqueeze(val_dataset[k], 0)
            noise =torch.randn_like(img_val)*sig/255
            img_val,noise=img_val.to(device),noise.to(device)
            imgn_val = img_val + noise
            with torch.no_grad():
                out_val=torch.clamp(net(imgn_val), 0., 1.)
                psnr_val += batch_PSNR(out_val, img_val, 1.)
        psnr_val /= len(val_dataset)
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        # log the images
        Img = utils.make_grid(img_train.detach(), nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.detach(), nrow=8, normalize=True, scale_each=True)
        Imgout_train = utils.make_grid(imgout_train.detach(), nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('Denoised image', Imgout_train, epoch)    
        torch.save(net.state_dict(), os.path.join(save_folder_path, dict_save_name+'.pth'))