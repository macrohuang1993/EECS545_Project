import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import myDnCNN
from utils import *
import matplotlib.pyplot as plt
import torchvision.utils as utils


def main(model_dict_path='mylogs/test/model dict.pth',use_cuda=True,sig=25,test_set='Set68',output=False):
    device = torch.device("cuda" if use_cuda else "cpu")
    # Build model
    print('Loading model ...\n')
    net = myDnCNN(channels=1, num_of_layers=17)
    net = net.to(device)
    net.load_state_dict(torch.load(model_dict_path))
    net.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', test_set, '*.png'))
    files_source.sort()
    # process data
    psnr_test = 0
    INoisy_All=np.zeros([68,1,481,481])
    Iclean_All=np.zeros([68,1,481,481])
    Idenoised=np.zeros([68,1,481,481])
    idx=0
    for f in files_source:
        # image
        Img = cv2.imread(f)
        Img = (np.float32(Img[:,:,0]))/255
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)
        
        # noise
        noise=torch.randn_like(ISource)*sig/255
        ISource,noise=ISource.to(device),noise.to(device)
        # noisy image
        INoisy = ISource + noise
        with torch.no_grad(): # this can save much memory
            Iout = torch.clamp(net(INoisy), 0., 1.)
        psnr = batch_PSNR(Iout, ISource, 1.)
        print("File %s has PSNR %.3f" % (f, psnr))
        psnr_test += psnr
        if output:
            H,W=INoisy.cpu().numpy().squeeze().shape
            INoisy_All[idx,0,:H,:W]=INoisy.cpu().numpy()
            Iclean_All[idx,0,:H,:W]=ISource.cpu().numpy()
            Idenoised[idx,0,:H,:W]=Iout.cpu().numpy()
        idx+=1
    psnr_test /= len(files_source)
    print("\nMean PSNR on test data %f" % psnr_test)
    if output:
        return{'PSNR':psnr_test,'INoisy_All':INoisy_All,'Iclean_All':Iclean_All,'Idenoised':Idenoised}
        

