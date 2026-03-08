#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#

import torch
import torch.nn.functional as F


def compute_auto_masks(reprojection_loss, identity_reprojection_loss):
    """ 
    This function computes auto mask using reprojection loss and identity reprojection loss.
    """
    if identity_reprojection_loss is None:
        reprojection_loss_mask = torch.ones_like(reprojection_loss)
    else:
        losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
        idxs = torch.argmin(losses, dim=1, keepdim=True)
        reprojection_loss_mask = (idxs == 0).float()
    return reprojection_loss_mask


def compute_masked_loss(loss, mask, eps=0):    
    """
    This function masks losses while avoiding zero division.
    """    

    return (loss * (mask+eps)).sum() / ((mask+eps).sum()+ 1e-8)


def compute_edg_smooth_loss(rgb, disp_map):
    """
    This function calculates edge-aware smoothness.
    """
    grad_rgb_x = (rgb[:, :, :, :-1] - rgb[:, :, :, 1:]).abs().mean(1, True)
    grad_rgb_y = (rgb[:, :, :-1, :] - rgb[:, :, 1:, :]).abs().mean(1, True)

    grad_disp_x = (disp_map[:, :, :, :-1] - disp_map[:, :, :, 1:]).abs()
    grad_disp_y = (disp_map[:, :, :-1, :] - disp_map[:, :, 1:, :]).abs()

    grad_disp_x *= (-1.0 * grad_rgb_x).exp()
    grad_disp_y *= (-1.0 * grad_rgb_y).exp()
    return grad_disp_x.mean() + grad_disp_y.mean()


def compute_ssim_loss(pred, target):
    """
    This function calculates SSIM loss between predicted image and target image.
    """
    ref_pad = torch.nn.ReflectionPad2d(1)
    pred = ref_pad(pred)
    target = ref_pad(target)

    mu_pred = F.avg_pool2d(pred, kernel_size = 3, stride = 1)
    mu_target = F.avg_pool2d(target, kernel_size = 3, stride = 1)

    musq_pred = mu_pred.pow(2)
    musq_target = mu_target.pow(2)
    mu_pred_target = mu_pred*mu_target

    sigma_pred = F.avg_pool2d(pred.pow(2), kernel_size = 3, stride = 1)-musq_pred
    sigma_target = F.avg_pool2d(target.pow(2), kernel_size = 3, stride = 1)-musq_target
    sigma_pred_target = F.avg_pool2d(pred*target, kernel_size = 3, stride = 1)-mu_pred_target

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu_pred_target + C1)*(2*sigma_pred_target + C2)) \
                    /((musq_pred + musq_target + C1)*(sigma_pred + sigma_target + C2)+1e-8)    
    return torch.clamp((1-ssim_map)/2, 0, 1)


def compute_photometric_loss(pred=None, target=None):
    """
    This function calculates photometric reconstruction loss (0.85*SSIM + 0.15*L1)
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)
    ssim_loss = compute_ssim_loss(pred, target).mean(1, True)
    rep_loss = 0.85 * ssim_loss + 0.15 * l1_loss
    return rep_loss
