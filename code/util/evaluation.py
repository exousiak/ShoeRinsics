import torch, torch.nn as nn
import numpy as np
from util.misc import valid_tensor
from scipy import ndimage
import os
import matplotlib.pyplot as plt
from datetime import datetime
import torch.nn.functional as F

'''Calculate Intersection over Union (IoU) between print_pred and print_GT. IoU is only calculated in area specified by mask.'''
def iou(print_pred, print_GT, mask):
    intersection = (~print_pred * ~print_GT)
    union = (~print_pred + ~print_GT)
    intersection_sum = torch.sum(intersection[mask])
    union_sum = torch.sum(union[mask])
    return 100.0 * intersection_sum / union_sum

import torch
import torch.nn.functional as F

def find_edges(mask, dilation_iterations):
    # 초기 경계선 검출
    edge_mask = torch.zeros_like(mask).bool()
    for y_shift in [-1, 0, 1]:
        for x_shift in [-1, 0, 1]:
            if y_shift == 0 and x_shift == 0:
                continue
            shifted_mask = torch.roll(mask, shifts=(y_shift, x_shift), dims=(-2, -1))
            edge_mask |= (mask & ~shifted_mask)

    # 경계선을 더 두껍게 만들기 위해 추가적인 팽창 과정 적용
    for i in range(dilation_iterations):
        dilated_edge_mask = edge_mask.clone()
        # 팽창을 균일하게 적용하기 위해, 각 방향으로 동일한 횟수만큼 팽창을 시도합니다.
        for y_shift in [-1, 0, 1]:
            for x_shift in [-1, 0, 1]:
                if y_shift == 0 and x_shift == 0:
                    continue
                shifted_edge_mask = torch.roll(dilated_edge_mask, shifts=(y_shift, x_shift), dims=(1, 2))
                edge_mask |= shifted_edge_mask
    return edge_mask

"""Given a depth map or an RGB image of a shoe, get an estimate for the print left on the ground"""
def get_print(t, mask, print_GT):
    if t.shape[0] > 1:
        prints = torch.ones(t.shape, dtype=torch.bool).to(t.device)

        for i in range(t.shape[0]):
            ind_prints = get_print(t[i:i+1, ...], mask[i:i+1, ...], print_GT[i:i+1, ...] if print_GT is not None else None)
            prints[i, ...] = ind_prints

        return prints

    if t.shape[1] == 3:
        tensor = 1 - t.mean(dim=1).unsqueeze(1).clamp(0, 1)
        kernel_size = 15
        thresh = 1
    else:
        tensor = t
        kernel_size = 45
        thresh = 1
    tensor = tensor - torch.min(tensor)

    kernel = (torch.ones((1, 1, kernel_size, kernel_size)) / (kernel_size * kernel_size)).to(t.device)
    depth_blur = nn.functional.conv2d(mask * tensor, kernel, padding=kernel_size//2)
    mask_blur = nn.functional.conv2d(mask.float(), kernel, padding=kernel_size//2)
    blur = depth_blur / mask_blur
    outside_shoe = ~mask.bool()
    blur[outside_shoe] = 0

    if t.shape[1] == 3 or not valid_tensor(print_GT):
        print_ = tensor < (blur * thresh)
        print_[outside_shoe] = 0
        print_ = ~print_
        depth_vals = torch.sort(tensor[mask])[0]
        high_depth_val = depth_vals[int(depth_vals.shape[0] * .95)]
        print_[tensor > high_depth_val] = True
        low_depth_val = depth_vals[int(depth_vals.shape[0] * .05)]
        print_[tensor < low_depth_val] = False
    else:
        thresh_min = 0.1
        thresh_max = 2
        int_over_uni = -1
        ious = [None] * 190
        for i, thresh in enumerate(np.arange(thresh_min, thresh_max, 0.01)):

            print_ = tensor < (blur * thresh)
            print_[outside_shoe] = 0
            print_ = ~print_
            new_int_over_uni = iou(print_, print_GT, mask)
            ious[i] = new_int_over_uni
            if new_int_over_uni > int_over_uni:
                best_thresh = thresh
                int_over_uni = new_int_over_uni

        print_ = tensor < (blur * best_thresh)
        print_[outside_shoe] = 0
        print_ = ~print_
        best_iou = iou(print_, print_GT, mask)

        # specify very high values as non contact surface
        depth_vals = tensor[mask]
        high_depth_val = torch.sort(depth_vals)[0][int(depth_vals.shape[0] * .95)]
        best_upper_depth_thresh = None
        for i, thresh in enumerate(np.arange(0.1, 1, 0.01)):
            new_iou = iou(print_ + (tensor > (high_depth_val * thresh)), print_GT, mask)
            if new_iou > best_iou:
                best_upper_depth_thresh = thresh
                best_iou = new_iou
        if best_upper_depth_thresh:
            print_[tensor > high_depth_val * best_upper_depth_thresh] = True

        # specify very low values as contact surface
        low_depth_val = torch.sort(depth_vals)[0][int(depth_vals.shape[0] * .05)]
        best_lower_depth_thresh = None
        for i, thresh in enumerate(np.arange(1, 30, 0.1)):
            new_iou = iou(print_ * (tensor > (low_depth_val * thresh)), print_GT, mask)
            if new_iou > best_iou:
                best_lower_depth_thresh = thresh
                best_iou = new_iou
        if best_lower_depth_thresh:
            print_[tensor < low_depth_val * best_lower_depth_thresh] = False
    
    # 경계선 검출 
    edges = find_edges(mask, dilation_iterations=3) # 팽창 파라미터 선정 -> 3
    print_ = print_.clone()
    print_[edges] = True
            
    # 족적 이미지 흑백 전환
    return ~print_ 

'''
    # 족적 이미지 흑백 전환 
    print_ = ~print_

    # edges = detect_edges(mask)
    # print_[edges] = False
    
    return print_
'''