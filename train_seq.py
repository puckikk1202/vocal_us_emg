import os
import numpy as np
import torch
import torch.nn as nn
from vit_pytorch.vivit import ViT
from torch.utils.data import DataLoader, random_split, Dataset  
from data.kpdataset import SeqKeypointDataset
from model.resnet import ResNet50, ResNet101, ResNet152
import cv2
import json

import wandb
from torch.utils.tensorboard import SummaryWriter



wandb.login()
wandb.init(project='us_annotation')
# writer = SummaryWriter()

# 86.0 520.0
# -100.0, 504.0
global_max = 504.0
global_min = -100.0

def draw_img(img, keypoint, gt_keypoint):
    keypoint = ((keypoint + 1) / 2 ) * (global_max - global_min) + global_min
    gt_keypoint = ((gt_keypoint + 1) / 2 ) * (global_max - global_min) + global_min
    points = [(keypoint[i], keypoint[i+1]) for i in range(0, len(keypoint), 2)]
    gt_points = [(gt_keypoint[i], gt_keypoint[i+1]) for i in range(0, len(gt_keypoint), 2)]

    img = (img + 1) / 2 * 255
    img = cv2.resize(img, (600, 600))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    for x, y in gt_points:
        cv2.circle(img, (int(x), int(y)), radius=5, color=(12, 50, 255), thickness=-1)


    for x, y in points:     
        cv2.circle(img, (int(x), int(y)), radius=5, color=(128, 255, 128), thickness=-1)
    
    return img

with open('us_annotation_Cady_normal.json', 'r') as f:
    normalized_data = json.load(f)

seq_len = 15
dataset = SeqKeypointDataset(normalized_data, seq_len=seq_len)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# model = ResNet101(num_classes=10, channels=1).cuda()
# optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
# criterion = nn.MSELoss()

model= ViT(
    image_size = 512,          # image size
    frames = seq_len,               # number of frames
    image_patch_size = 64,     # image patch size
    frame_patch_size = 3,      # frame patch size
    num_classes = 10,
    channels = 1,              # number of image channels
    dim = 1024,
    spatial_depth = 6,         # depth of the spatial transformer
    temporal_depth = 6,        # depth of the temporal transformer
    heads = 8,
    mlp_dim = 1024,
).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.MSELoss()
wandb.watch(model, log_freq=100)

for epoch in range(300):
    for i, (kp, img) in enumerate(train_loader):
        model.train()
        kp = kp.cuda()
        img = img.cuda()

        output = model(img)
        loss = criterion(output, kp)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
            # print(torch.squeeze(img[i][:,seq_len//2,:,:]).shape, output.shape, kp.shape)
            result_imgs = [draw_img(torch.squeeze(img[i][:,seq_len//2,:,:]).cpu().detach().numpy(), output[i].cpu().detach().numpy(), kp[i].cpu().detach().numpy()) for i in range(16)]
            wandb.log({"train_loss": loss})
            wandb.log({"train_image": [wandb.Image(img) for img in result_imgs]})
            result_imgs_tensor = torch.stack([torch.tensor(img/255, dtype=torch.float32) for img in result_imgs]).permute(0, 3, 1, 2)
            assert result_imgs_tensor.ndim == 4 and result_imgs_tensor.shape[1] == 3, "Image tensors must be (N, 3, H, W)"
            # writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)
            # writer.add_images('Images/train', result_imgs_tensor, epoch * len(train_loader) + i)

    del kp, img, output, loss
    # print(f'Epoch {epoch}, Loss: {loss.item()}')

    for i ,(kp, img) in enumerate(test_loader):
        model.eval()
        kp = kp.cuda()
        img = img.cuda()

        output = model(img)
        loss = criterion(output, kp)

        if i % 10 == 0:
            print(f'Test Loss: {loss.item()}')
            result_imgs = [draw_img(torch.squeeze(img[i][:,seq_len//2,:,:]).cpu().detach().numpy(), output[i].cpu().detach().numpy(), kp[i].cpu().detach().numpy()) for i in range(img.shape[0])]
            wandb.log({"test_loss": loss})
            wandb.log({"test_image": [wandb.Image(img) for img in result_imgs]})
            result_imgs_tensor = torch.stack([torch.tensor(img/255, dtype=torch.float32) for img in result_imgs]).permute(0, 3, 1, 2)
            assert result_imgs_tensor.ndim == 4 and result_imgs_tensor.shape[1] == 3, "Image tensors must be (N, 3, H, W)"
            # writer.add_scalar('Loss/test', loss.item(), epoch * len(test_loader) + i)
            # writer.add_images('Images/test', result_imgs_tensor, epoch * len(test_loader) + i)

    del kp, img, output, loss

wandb.finish()
torch.save(model.state_dict(), f'../output_models/Cady_model_seq_{epoch}.pth')



