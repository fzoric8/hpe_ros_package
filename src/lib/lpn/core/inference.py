# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Updated by Zhe Zhang (zhangzhe@smail.nju.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch
import torch.nn as nn
import time

from lpn.utils.transforms import transform_preds


def get_predicitons(batch_heatmaps, scaling=False, img_shape=(640, 480)): 

    # Batch size
    batch_size = batch_heatmaps.shape[0]
    # Number of joints
    num_joints = batch_heatmaps.shape[1]
    # Heatmap shape
    hm_w, hm_h = batch_heatmaps.shape[-2] ,batch_heatmaps.shape[-1]
    
    if scaling: 
        img_w, img_h = img_shape[0], img_shape[1]
    else:
        img_w, img_h = hm_w, hm_h

    preds = np.zeros((16, 2))
    for i in range(num_joints): 
        _ = np.unravel_index(batch_heatmaps[0, i, :, :].argmax(), (hm_w, hm_h))
        # Rotate it 
        preds[i][0] = _[1] * img_w/hm_w
        preds[i][1] = _[0] * img_h/hm_h

    return preds


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = np.array(
                        [
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals


class SoftArgmax2D(nn.Module):
    def __init__(self, height=64, width=48, beta=100):
        super(SoftArgmax2D, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.beta = beta
        # Note that meshgrid in pytorch behaves differently with numpy.
        self.WY, self.WX = torch.meshgrid(torch.arange(height, dtype=torch.float),
                                          torch.arange(width, dtype=torch.float))

    def forward(self, x):
        b, c, h, w = x.shape
        device = x.device

        probs = self.softmax(x.view(b, c, -1) * self.beta).to(device)
        probs = probs.view(b, c, h, w)
        self.WX = self.WX.to(device)
        self.WY = self.WY.to(device)

        px = torch.sum(probs * self.WX, dim=(2, 3)).to(device)
        py = torch.sum(probs * self.WY, dim=(2, 3)).to(device)
        preds = torch.stack((px, py), dim=-1)

        # I don't use maxvals and I don't know what they do!
        # Copy to CPU? Bottleneck!
        #preds = preds.cpu().detach().numpy()

        #idx = np.round(preds).astype(np.int32)
        #maxvals = np.zeros(shape=(b, c, 1))
        #for bi in range(b):
        #    for ci in range(c):
        #        maxvals[bi, ci, 0] = x[bi, ci, idx[bi, ci, 1], idx[bi, ci, 0]]

        return preds


def get_final_preds_using_softargmax(config, batch_heatmaps, center, scale):
    soft_argmax = SoftArgmax2D(config.MODEL.HEATMAP_SIZE[1], config.MODEL.HEATMAP_SIZE[0], beta=160)
    start_time0 = time.time()
    coords = soft_argmax(batch_heatmaps)
    print("Soft Argmax duration is: {}".format(time.time() - start_time0))

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    #batch_heatmaps = batch_heatmaps.cpu().detach().numpy()
    start_time = time.time()
    coords = coords.cpu().detach().numpy()
    print("Coords CPU conversion is: {}".format(time.time() - start_time))

    # post-processing --> Not sure what this does
    if False:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = np.array(
                        [
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25


    #preds = coords.copy()

    # Transform back
    #for i in range(coords.shape[0]):
    #    preds[i] = transform_preds(
    #        coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
    #    )

    return coords
