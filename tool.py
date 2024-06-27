import numpy as np
import cv2
import torch

def draw_img(img, keypoint, gt_keypoint=None, global_max=504.0, global_min=-100.0):
    keypoint = ((keypoint + 1) / 2 ) * (global_max - global_min) + global_min
    points = [(keypoint[i], keypoint[i+1]) for i in range(0, len(keypoint), 2)]

    if gt_keypoint is not None:
        gt_keypoint = ((gt_keypoint + 1) / 2 ) * (global_max - global_min) + global_min
        gt_points = [(gt_keypoint[i], gt_keypoint[i+1]) for i in range(0, len(gt_keypoint), 2)]

    img = (img + 1) / 2 * 255
    img = cv2.resize(img, (600, 600))
    # print(img.shape)
    img = np.float32(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if gt_keypoint is not None:
        for x, y in gt_points:
            cv2.circle(img, (int(x), int(y)), radius=5, color=(12, 50, 255), thickness=-1)


    for x, y in points:     
        cv2.circle(img, (int(x), int(y)), radius=5, color=(128, 255, 128), thickness=-1)
    
    return img