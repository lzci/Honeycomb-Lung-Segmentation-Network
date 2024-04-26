import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import pdb
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
import os
from BagData import train_dataloader,test_dataloader
from models.FCN import VGGNet,FCNs,FCN8s,FCN16s,FCN32s
from models.segnet import SegNet
from models.md10 import md10
from models.r2unet import R2U_Net
from models.attention_unet import AttU_Net
from models.UNet__ import NestedUNet
from models.R2AttU_Net import R2AttU_Net
from models.Res_UNet import res_unet
from models.resnest_unet import resnest_unet
from models.resnestunet import resnestunet
from models.resnestunetattention import resnestunetattention
from models.HS_Unet import HS_Unet
from inference import IoU,meanIoU,Dice,meanDice,PA,MPA,remove_small_regions

if __name__ == "__main__":


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    epo_num = 200
    learning_rate= 0.001
    momentum= 0.9

    
    
    # model = FCNs(pretrained_net=VGGNet(requires_grad=True), n_class=2)
    # model = SegNet()
    model = md10()
    

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    
    criterion = nn.BCELoss().to(device)                 
    # criterion = MixedLoss().to(device)                 
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)       #学习率调整（LambdaLR;StepLR;MultiStepLR;ExponentialLR）四种调整形式
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    all_train_iter_loss = []
    all_test_iter_loss = []

    
    with open("results/unet_predict_dataexoansionmod.txt", "w") as f:
        for epo in range(epo_num):
            # if epoch % 50 == 0 and epoch != 0:
            #     for group in optimizer.param_groups:
            #         group['lr'] *= 0.5
            train_loss = 0
            start = time.perf_counter()
            model.train()
            for index, (input, y) in enumerate(train_dataloader):
                input,y = input.to(device),y.to(device)  
                input,y = torch.autograd.Variable(input),torch.autograd.Variable(y)

                optimizer.zero_grad()
                output = model(input)
                output = torch.sigmoid(output)
                loss = criterion(output, y)
                loss.backward()
                iter_loss = loss.item()
                all_train_iter_loss.append(iter_loss)
                train_loss += iter_loss
                optimizer.step()

                output_np = output.cpu().data.numpy().copy()
                output_np = np.argmin(output_np, axis=1)
                y_np = y.cpu().data.numpy().copy()
                y_np = np.argmin(y_np, axis=1)
                print('epoch {}, {}/{}, loss is {}'.format(epo + 1, index + 1, len(train_dataloader), iter_loss))
            print('train epoch loss = %f' % (train_loss / len(train_dataloader)))
            
            
            
            test_loss = 0
            total_ious = []
            total_mean_ious = []
            total_dices = []
            total_pre = []
            model.eval()
            with torch.no_grad():
                for index, (input, y) in enumerate(test_dataloader):
                    input,y = input.to(device),y.to(device) 
                    input,y = torch.autograd.Variable(input),torch.autograd.Variable(y)
                    
                    output = model(input)
                    output = torch.sigmoid(output)
                    loss = criterion(output, y)
                    iter_loss = loss.item()
                    all_test_iter_loss.append(iter_loss)
                    test_loss += iter_loss

                    output_np = output.cpu().data.numpy().copy()
                    output_np = np.argmin(output_np, axis=1)
                    y_np = y.cpu().data.numpy().copy()
                    y_np = np.argmin(y_np, axis=1)
                    # y=remove_small_regions(y, 0.02 * np.prod((160,160)))
                    total_ious.append(IoU(y_np, output_np))
                    total_mean_ious.append(meanIoU(y_np,output_np))
                    total_dices.append(Dice(y_np,output_np))
                    total_pre.append(Pre(y_np,output_np))
                   
                    # cv2.imwrite("predict/image_{}.jpg".format(index), np.squeeze(y_np[0, ...] * 255))
                    # cv2.imwrite("predict_msk/binary_{}.jpg".format(index), np.squeeze(output_np[0, ...] * 255))
            all_test_iter_loss.append(test_loss/len(test_dataloader))
            # draw_loss_plot(all_train_iter_loss,all_test_iter_loss)
            ious = np.nanmean(total_ious, axis=0)
            mean_ious = np.nanmean(total_mean_ious,axis=0)
            dices = np.nanmean(total_dices, axis=0)
            pres = np.nanmean(total_pre, axis=0)
           
            # pixel_accs = np.array(pixel_accs).mean()
            end = time.perf_counter()
            print('IoU = %.5f , meanIoU = %.5f , Dice = %.5f , pre = %.5f , test loss = %f' % (ious,mean_ious,dices,pre(test_loss/len(test_dataloader))))
            # torch.save(model, 'checkpoints/R2AttU_Net/R2AttU_Net_{}.pth'.format(epo+1))
            f.write('epoch = %d , train loss = %.5f , test loss = %.5f , IoU = %.5f , meanIoU = %.5f , Dice = %.5f , pre = %.5f , time = %.2f ' % 
                ((epo+1),(train_loss / len(train_dataloader)),(test_loss / len(test_dataloader)),ious,mean_ious,dices,pre,(end-start)))
            f.write("\n")
            f.flush()