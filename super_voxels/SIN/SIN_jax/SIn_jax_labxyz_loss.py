#based on https://github.com/yuanqqq/SIN

from flax import linen as nn
import numpy as np
from typing import Any, Callable, Optional, Tuple, Type, List
from jax import lax, random, numpy as jnp
import einops
import torchio
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import jax
import tensorflow as tf
import torch 
import einops
import torchio as tio
import optax
from flax.training import train_state  
from torch.utils.data import DataLoader
import jax.profiler
import ml_collections
from ml_collections import config_dict

import numpy as np



def lab_xyz_loss(prob, label):
    _, color_c, _, _ = label.shape
    #v
    kernel = np.array([[[0, -1, 0], [0, 1, 0], [0, 0, 0]],
              [[0, 0, 0], [0, 1, 0], [0, -1, 0]]])
    #h
    kernel = [[[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, -1], [0, 0, 0]]]
    
    kernel = torch.tensor(kernel).float().cuda().repeat(color_c, 1, 1).unsqueeze(1)
    # cat_feat = img
    label = F.pad(label, (1, 1, 0, 0), mode='replicate')
    cat_feat = F.conv2d(label, kernel, stride=(2, 1), padding=(0, 0), groups=color_c)
    cat_feat = cat_feat*cat_feat

    cat_feat = F.relu(-F.relu(cat_feat)+1)
    b, c, h, w = cat_feat.shape
    _, gt_id = cat_feat.permute(0, 2, 3, 1).reshape(-1, 2).max(1, keepdim=False)

    cross_loss = nn.CrossEntropyLoss(reduction='none')
    color_loss = cross_loss(prob[:, :, 1:-1:2, :].permute(0, 2, 3, 1).reshape(-1, 2), gt_id)
    color_loss = color_loss.view(b, h, w)


    gt = cat_feat[:, 0, :, :] - cat_feat[:, 1, :, :]
    weight = gt * gt
    color_loss = weight*color_loss
    color_loss = torch.sum(torch.sum(color_loss, dim=-1), dim=-1)



    return color_loss



def compute_labxy_loss(prob0_v, prob0_h, prob1_v, prob1_h, prob2_v, prob2_h, prob3_v, prob3_h, label):
    p0v = prob0_v.clone()
    p0h = prob0_h.clone()
    p1v = prob1_v.clone()
    p1h = prob1_h.clone()
    p2v = prob2_v.clone()
    p2h = prob2_h.clone()
    p3v = prob3_v.clone()
    p3h = prob3_h.clone()
    b, c, h, w = label.shape
    # weight = [0.25, 0.5, 1., 2., 4., 8., 16., 32.]
    weight = [1, 2.5, 2, 5, 4, 10, 8, 20]
    # weight = [1, 1, 1, 1, 1, 1, 1, 1]
    weight = torch.tensor(weight).cuda().reshape(8, ).float()

    # img = rgb2Lab_torch(img, torch.tensor([0.411, 0.432, 0.45]).unsqueeze(-1).unsqueeze(-1))
    # img = gaussian_kernel(img)
    # img_lr_grad = compute_lr_grad(img)
    # img_tb_grad = compute_tb_grad(img)
    # gt_h_17, gt_h_9, gt_h_5, gt_h_3, gt_v_17, gt_v_9, gt_v_5, gt_v_3 = compute_gt(img)
    # img = (0.2989 * img[:, 0, :, :] + 0.5870 * img[:, 1, :, :] + 0.1140 * img[:, 2, :, :]).unsqueeze(1)

    lab_loss = torch.zeros(8, b).cuda()

    # todo: complete labxy_loss
    label_0v = label
    lab_loss[0] = labxy_v_loss(p0v, label_0v)
    label_0h = label_0v[:, :, 0::2, :]
    lab_loss[1] = labxy_h_loss(p0h, label_0h)

    label_1v = label_0h[:, :, :, 0::2]
    lab_loss[2] = labxy_v_loss(p1v, label_1v)
    label_1h = label_1v[:, :, 0::2, :]
    lab_loss[3] = labxy_h_loss(p1h, label_1h)

    label_2v = label_1h[:, :, :, 0::2]
    lab_loss[4] = labxy_v_loss(p2v, label_2v)
    label_2h = label_2v[:, :, 0::2, :]
    lab_loss[5] = labxy_h_loss(p2h, label_2h)

    label_3v = label_2h[:, :, :, 0::2]
    lab_loss[6] = labxy_v_loss(p3v, label_3v)
    label_3h = label_3v[:, :, 0::2, :]
    lab_loss[7] = labxy_h_loss(p3h, label_3h)

    lab_loss = torch.sum(lab_loss, dim=-1) / b

    lab_loss = torch.sum(lab_loss * weight, dim=0)

    lab_loss = 0.005 * lab_loss

    return lab_loss




# kernel_v = jnp.array([[[0, -1, 0], [0, 1, 0], [0, 0, 0]],
#             [[0, 0, 0], [0, 1, 0], [0, -1, 0]]])
# kernel_v=einops.rearrange(kernel_v,'c x y-> 1 c x y')
# image_rgb= jnp.ones()

# kernel_h = np.array([[[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
#         [[0, 0, 0], [0, 1, -1], [0, 0, 0]]])



# jax.lax.conv(lhs, rhs, window_strides, padding)

# kernel_h.shape#(2, 3, 3)
# kernel_v.shape#(2, 3, 3)


# np.array(kernel_h[0,:,:])
# np.rot90(np.array(kernel_v[0,:,:]))


# np.array(kernel_h[1,:,:])
# np.rot90(np.array(kernel_v[1,:,:]))

# kernel_h[1,:,:]
# kernel_v[1,:,:]


# kernel_h[:,0,:]
# kernel_v[:,0,:]

# kernel_h[:,1,:]
# kernel_v[:,1,:]

# kernel_h[:,2,:]
# kernel_v[:,2,:]



# kernel_h[:,:,0]
# kernel_v[:,:,0]

# kernel_h[:,:,1]
# kernel_v[:,:,1]

# kernel_h[:,:,2]
# kernel_v[:,:,2]


# kernel_h[:,0,0]#0 0
# kernel_v[:,0,0]#0 0 

# kernel_h[0,:,0]# 0, -1,  0
# kernel_v[0,:,0]#0, 0, 0

# kernel_h[0,0,1]#0
# kernel_v[0,0,1]#-1