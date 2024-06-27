import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset 
import cv2
import numpy as np
import os


class SeqKeypointDataset(Dataset):
    def __init__(self, data, seq_len=5):
        """
        Args:
            data (dict): Dictionary with normalized data.
        """
        self.keys = list(data.keys())
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        print(key)
        # print(key)
        # img = cv2.imread(f'./frames/{key.split("_")[0]}_cond3_{key.split("_")[1]}/frame_{int(key.split("_")[2]):04d}.jpg', cv2.IMREAD_GRAYSCALE)
        frames = []
        p = 1
        for i in range(self.seq_len):
            fi = i-self.seq_len // 2
            img_path = f'../frames/{key.split("_")[0]}_{key.split("_")[1]}/frame_{int(key.split("_")[2]) + fi:04d}.jpg'
            # print(img_path)
            if not os.path.exists(img_path):
                fi = fi - p
                p += 1
                img_path = f'../frames/{key.split("_")[0]}_{key.split("_")[1]}/frame_{int(key.split("_")[2]) + fi:04d}.jpg'
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            height_extension = 100
            new_height = img.shape[0] + height_extension
            new_image = np.zeros((new_height, img.shape[1]), dtype=np.float32)
            new_image[:img.shape[0], :] = img
            new_image = cv2.resize(new_image, (512, 512))
            new_image = (new_image / 255.0) * 2 - 1
            new_image = np.expand_dims(new_image, axis=0)
            new_image = torch.tensor(new_image, dtype=torch.float32)
            frames.append(new_image)
            # print(np.min(new_image), np.max(new_image))
            kp = self.data[key]
        frames = torch.stack(frames)
        frames = frames.permute(1, 0, 2, 3)

        return torch.tensor(kp, dtype=torch.float32), frames
        



        