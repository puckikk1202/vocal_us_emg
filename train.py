import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset  
from model.resnet import ResNet50, ResNet101, ResNet152
import cv2
import json

import wandb
from torch.utils.tensorboard import SummaryWriter



wandb.login()
wandb.init(project='us_annotation')
writer = SummaryWriter()

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
    
class KeypointDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (dict): Dictionary with normalized data.
        """
        self.keys = list(data.keys())
        self.data = data

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        # print(key)
        # img = cv2.imread(f'./frames/{key.split("_")[0]}_cond3_{key.split("_")[1]}/frame_{int(key.split("_")[2]):04d}.jpg', cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(f'./frames/{key.split("_")[0]}_{key.split("_")[1]}/frame_{int(key.split("_")[2]):04d}.jpg', cv2.IMREAD_GRAYSCALE)
        # print(img.shape)
        height_extension = 100
        new_height = img.shape[0] + height_extension
        new_image = np.zeros((new_height, img.shape[1]), dtype=np.float32)
        new_image[:img.shape[0], :] = img
        new_image = cv2.resize(new_image, (512, 512))
        new_image = (new_image / 255.0) * 2 - 1
        new_image = np.expand_dims(new_image, axis=0)
        # print(np.min(new_image), np.max(new_image))
        kp = self.data[key]

        return torch.tensor(kp, dtype=torch.float32), torch.tensor(new_image, dtype=torch.float32)


dataset = KeypointDataset(normalized_data)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


model = ResNet101(num_classes=10, channels=1).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
criterion = nn.MSELoss()

wandb.watch(model, log_freq=100)

for epoch in range(100):
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
            result_imgs = [draw_img(img[i][0].cpu().detach().numpy(), output[i].cpu().detach().numpy(), kp[i].cpu().detach().numpy()) for i in range(16)]
            wandb.log({"train_loss": loss})
            wandb.log({"train_image": [wandb.Image(img) for img in result_imgs]})
            result_imgs_tensor = torch.stack([torch.tensor(img/255, dtype=torch.float32) for img in result_imgs]).permute(0, 3, 1, 2)
            assert result_imgs_tensor.ndim == 4 and result_imgs_tensor.shape[1] == 3, "Image tensors must be (N, 3, H, W)"
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)
            writer.add_images('Images/train', result_imgs_tensor, epoch * len(train_loader) + i)

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
            result_imgs = [draw_img(img[i][0].cpu().detach().numpy(), output[i].cpu().detach().numpy(), kp[i].cpu().detach().numpy()) for i in range(img.shape[0])]
            wandb.log({"test_loss": loss})
            wandb.log({"test_image": [wandb.Image(img) for img in result_imgs]})
            result_imgs_tensor = torch.stack([torch.tensor(img/255, dtype=torch.float32) for img in result_imgs]).permute(0, 3, 1, 2)
            assert result_imgs_tensor.ndim == 4 and result_imgs_tensor.shape[1] == 3, "Image tensors must be (N, 3, H, W)"
            writer.add_scalar('Loss/test', loss.item(), epoch * len(test_loader) + i)
            writer.add_images('Images/test', result_imgs_tensor, epoch * len(test_loader) + i)

    del kp, img, output, loss
    # print(f'Test Loss: {loss.item()}')
wandb.finish()
torch.save(model.state_dict(), f'./Cady_model_{epoch}.pth')



