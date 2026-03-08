
#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#

from collections import defaultdict
import PIL.Image as Image
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import shutil
import numpy as np
import pytorch_lightning as pl
from einops import rearrange, reduce
from torch import Tensor
from torch_scatter import scatter_add, scatter_max
from lpips import LPIPS
from jaxtyping import Float, UInt8
from pytorch_lightning.utilities import rank_zero_only
from skimage.metrics import structural_similarity
from kornia.losses import SSIMLoss
from math import log2, log
import sys
from gsplat.rendering import rasterization
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.checkpoint import checkpoint

from models.vggt.models.vggt import VGGT
from models.vggt.heads.dpt_head import DPTHead
from models.vggt.heads.gs_dpt_head import VGGT_DPT_GS_Head
from models.gaussian_util import render, focal2fov, getProjectionMatrix, depth2pc, pc2depth, rotate_sh, quat_multiply
from models.loss_util import compute_photometric_loss, compute_masked_loss, compute_edg_smooth_loss
from utils.visual_util import predictions_to_glb
from models.geometry_util import Projection

from models.gaussian_util import render, focal2fov, getProjectionMatrix,  depth2pc, pc2depth, rotate_sh, quat_multiply

from models.loss_util import compute_photometric_loss, compute_masked_loss, compute_edg_smooth_loss
from utils.visual_util import predictions_to_glb
from models.geometry_util import Projection


class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=8, alpha=32, dropout=0.0):
        super().__init__()
        self.linear = linear_layer
        self.lora_down = nn.Linear(linear_layer.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, linear_layer.out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)
        
        self.scaling = alpha / rank
        
        
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x):
        orig_out = self.linear(x)
        lora_out = self.lora_up(self.dropout(self.lora_down(x)))
        return orig_out + lora_out * self.scaling
    
    def merge_weights(self):
        
        merged_weight = self.linear.weight.data + self.scaling * (
            self.lora_up.weight @ self.lora_down.weight
        )
        return merged_weight

def apply_lora(model, layer_names=None, rank=8, alpha=32, dropout=0.0):

    if layer_names is None:
        layer_names = []
        for name, _ in model.named_modules():
            if isinstance(_, nn.Linear):
                layer_names.append(name)
    
    
    for name, module in model.named_children():
        
        full_name = f"{name}"
        
        
        should_replace = any(key in full_name for key in layer_names)
        
        if should_replace and isinstance(module, nn.Linear):
            
            setattr(model, name, LoRALinear(module, rank, alpha, dropout))
        elif len(list(module.children())) > 0:
            
            apply_lora(module, layer_names, rank, alpha, dropout)
    
    return model

def verify_frozen_parameters(model):
    trainable_params = 0
    all_params = 0
   
    for name, param in model.named_parameters():
        param_size = param.numel()
        all_params += param_size
        if "lora_" not in name and param.requires_grad:
           
            param.requires_grad = False  
            trainable_params +=param_size


def extract_lora_state_dict(model):
    
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            prefix = name + "."
            lora_state_dict[prefix + "lora_down.weight"] = module.lora_down.weight
            lora_state_dict[prefix + "lora_up.weight"] = module.lora_up.weight
    return lora_state_dict

def load_lora_state_dict(model, state_dict):
    
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            prefix = name + "."
            module.lora_down.weight = nn.Parameter(state_dict[prefix + "lora_down.weight"])
            module.lora_up.weight = nn.Parameter(state_dict[prefix + "lora_up.weight"])
    return model


class ReconDriveModel(torch.nn.Module):
    def __init__(self, sh_degree, min_depth, max_depth, vggt_checkpoint="./checkpoints/vggt.pt"):
        super(ReconDriveModel, self).__init__()
        self.img_size = 518
        self.patch_size = 14
        self.embed_dim = 1024
        self.sh_degree = sh_degree
        self.min_depth = min_depth
        self.max_depth = max_depth

        vggt_model = VGGT()
        # VGGT_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        # vggt_model.load_state_dict(torch.hub.load_state_dict_from_url(VGGT_URL))

        vggt_model.load_state_dict(torch.load(vggt_checkpoint))

        
        # self.aggregator = vggt_model.aggregator
        self.aggregator = apply_lora(vggt_model.aggregator,layer_names=['qkv','proj','fc1','fc2'], dropout=0.05)
        verify_frozen_parameters(self.aggregator)
        # self.depth_head = DPTHead(dim_in=2 * self.embed_dim, output_dim=2, activation="sigmoid", conf_activation="expp1")
        self.depth_head = vggt_model.depth_head
        
        
        # verify_frozen_parameters(self.depth_head)
        del vggt_model

        self.d_sh = (self.sh_degree + 1) ** 2

        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

        self.raw_gs_dim =  1 + 3 + 4 + 3*self.d_sh # opacity + scale + rot + d_sh

        self.gs_head = VGGT_DPT_GS_Head(
            dim_in=2 * self.embed_dim,
            img_dim_in=3,
            output_dim=self.raw_gs_dim,
            activation="linear",
            conf_activation="expp1",
        )

        self.track_head = None


    def freeze_parameters_except_heads(self):
        """
        Freeze all model parameters except those in depth_head and gs_head.
        This allows depth estimation and 3D Gaussian Splatting heads to be trained while keeping other components frozen.
        """
        # Count parameters before freezing
        total_params = 0
        trainable_params_before = 0
        trainable_params_after = 0

        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params_before += param.numel()

        # Freeze all parameters first
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze depth_head parameters
        for name, param in self.depth_head.named_parameters():
            param.requires_grad = True
            trainable_params_after += param.numel()

        # Unfreeze gs_head (3DGS) parameters
        for name, param in self.gs_head.named_parameters():
            param.requires_grad = True
            trainable_params_after += param.numel()

        print(f"Parameter freezing summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters before: {trainable_params_before:,}")
        print(f"  Trainable parameters after (depth_head + gs_head): {trainable_params_after:,}")
        print(f"  Trainable percentage: {trainable_params_after/total_params*100:.2f}%")
        print(f"  Only depth_head and gs_head (3DGS) parameters are trainable.")

        # Verify that only specified heads parameters are trainable
        self.verify_frozen_heads_parameters()

    def verify_frozen_heads_parameters(self):
        """
        Verify that only depth_head and gs_head parameters are trainable.
        """
        depth_head_trainable = 0
        gs_head_trainable = 0
        other_trainable = 0
        other_trainable_names = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                if name.startswith('depth_head'):
                    depth_head_trainable += param.numel()
                elif name.startswith('gs_head'):
                    gs_head_trainable += param.numel()
                else:
                    other_trainable += param.numel()
                    other_trainable_names.append(name)

        if other_trainable == 0:
            print(f"✓ Verification passed: Only depth_head and gs_head are trainable")
            print(f"  - depth_head: {depth_head_trainable:,} parameters")
            print(f"  - gs_head: {gs_head_trainable:,} parameters")
        else:
            print(f"✗ Verification failed: {other_trainable:,} non-target parameters are trainable")
            for name in other_trainable_names[:5]:  # Show first 5
                print(f"    - {name}")
            if len(other_trainable_names) > 5:
                print(f"    ... and {len(other_trainable_names) - 5} more")

    def unfreeze_all_parameters(self):
        """
        Unfreeze all model parameters (useful for full model training).
        """
        for param in self.parameters():
            param.requires_grad = True
        print("All parameters have been unfrozen.")

    def forward(self, images):
        '''
        images: Batch_size, view_num, 3, H, W
        e2c_extr: Batch_size, view_num, 4, 4
        K: Batch_size, view_num, 4, 4
        '''

        # with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            aggregated_tokens_list, patch_start_idx = self.aggregator(images.to(torch.bfloat16))

        with torch.amp.autocast("cuda", enabled=False):

            depth_maps, depth_conf = self.depth_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )

            depth_maps = torch.nn.functional.sigmoid(torch.log(depth_maps))

            min_depth = self.min_depth
            max_depth = self.max_depth
            depth_range = max_depth-min_depth
            depth_maps = min_depth + depth_range * depth_maps

            raw_gaussian = self.gs_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
            )  # batch_size, view_num,  H , W, D

            rot_maps, scale_maps, opacity_maps, sh_maps = raw_gaussian.split((4, 3, 1, 3 * self.d_sh), dim=-1)
            
            rot_maps = rot_maps / (rot_maps.norm(dim=-1, keepdim=True) + 1e-8)
            scale_maps = nn.functional.softplus(scale_maps,beta=1) * 0.01
            opacity_maps = nn.functional.sigmoid(opacity_maps)
            
            sh_maps = rearrange(sh_maps, "b n h w (i c) -> b n h w i c",i=3)
            sh_maps = sh_maps * self.sh_mask

            # Disabled track_head - create zero forward_flow to maintain compatibility
            if self.track_head is not None:
                forward_flow = self.track_head(
                                aggregated_tokens_list,
                                images,
                                patch_start_idx=patch_start_idx,
                                motion_tokens=None) # b, s, c, h, w
            else:
                # Create zero flow tensor with the same shape
                b, v, h, w, _ = depth_maps.shape
                forward_flow = torch.zeros(b, v, h, w, 3, dtype=depth_maps.dtype, device=depth_maps.device)
        return depth_maps, rot_maps, scale_maps, opacity_maps, sh_maps, forward_flow
    
    def forward_renderer(self, gs_params, data_dict, render_motion_seg=True, radius_clip=0.0):
        b, t, v, h, w, _ = gs_params["means"].shape
        tgt_h, tgt_w = data_dict["height"], data_dict["width"]
        tgt_t, tgt_v = data_dict["target_camtoworlds"].shape[1:3]
        means = rearrange(gs_params["means"], "b t v h w c -> b (t v h w) c")
        scales = rearrange(gs_params["scales"], "b t v h w c -> b (t v h w) c")
        quats = rearrange(gs_params["quats"], "b t v h w c -> b (t v h w) c")
        opacities = rearrange(gs_params["opacities"], "b t v h w -> b (t v h w)")
        colors = rearrange(gs_params["colors"], "b t v h w c -> b (t v h w) c")
        forward_v = rearrange(gs_params["forward_flow"], "b t v h w c -> b (t v h w) c")

        means_batched = means.repeat_interleave(tgt_t, dim=0)
        scales_batched = scales.repeat_interleave(tgt_t, dim=0)
        quats_batched = quats.repeat_interleave(tgt_t, dim=0)
        opacities_batched = opacities.repeat_interleave(tgt_t, dim=0)
        color_batched = colors.repeat_interleave(tgt_t, dim=0)
        forward_v_batched = forward_v.repeat_interleave(tgt_t, dim=0)

        ctx_time = data_dict["context_time"] * data_dict["timespan"]
        tgt_time = data_dict["target_time"] * data_dict["timespan"]
        if tgt_time.ndim == 3:
            tdiff_forward = tgt_time.unsqueeze(2) - ctx_time.unsqueeze(1)
            tdiff_forward = tdiff_forward.view(b * tgt_t, t * v, 1)
            tdiff_forward_batched = tdiff_forward.repeat_interleave(h * w, dim=1)
        else:
            tdiff_forward = tgt_time.unsqueeze(-1) - ctx_time.unsqueeze(-2)
            tdiff_forward = tdiff_forward.view(b * tgt_t, t, 1)
            tdiff_forward_batched = tdiff_forward.repeat_interleave(v * h * w, dim=1)
        forward_translation = forward_v_batched * tdiff_forward_batched
        means_batched = means_batched + forward_translation

        if not self.training:  # mask out some noisy flow
            forward_v[forward_v.norm(dim=-1) < 1.0] = 0.0
            forward_v_batched = forward_v.repeat_interleave(tgt_t, dim=0)

        if not self.training and self.num_motion_tokens > 0 and render_motion_seg:
            # render the motion segmentation map
            motion_weights = rearrange(gs_params["motion_weights"], "b t v h w k -> b (t v h w) k")
            weights_batched = motion_weights.repeat_interleave(tgt_t, dim=0)
            colors_batched = torch.cat([color_batched, forward_v_batched, weights_batched], dim=-1)
        else:
            colors_batched = torch.cat([color_batched, forward_v_batched], dim=-1)

        camtoworlds_batched = data_dict["target_camtoworlds"].view(b * tgt_t, -1, 4, 4)
        viewmats_batched = torch.linalg.inv(camtoworlds_batched.float())
        Ks_batched = data_dict["target_intrinsics"].view(b * tgt_t, -1, 3, 3)

        motion_seg = None
        if self.use_latest_gsplat:
            means_batched = means_batched.float()
            quats_batched = quats_batched.float()
            scales_batched = scales_batched.float()
            opacities_batched = opacities_batched.float()
            colors_batched = colors_batched.float()
            viewmats_batched = viewmats_batched.float()
            Ks_batched = Ks_batched.float()

            if not self.training:
                rendered_colors, rendered_alphas, rendered_flow, motion_seg = [], [], [], []
                rendered_depths = []
                with torch.autocast("cuda", enabled=False):
                    for bid in range(means_batched.size(0)):
                        renderings, alpha, _ = rasterization(
                            means=means_batched[bid],
                            quats=quats_batched[bid],
                            scales=scales_batched[bid],
                            opacities=opacities_batched[bid],
                            colors=colors_batched[bid],
                            viewmats=viewmats_batched[bid],
                            Ks=Ks_batched[bid],
                            width=data_dict["width"],
                            height=data_dict["height"],
                            render_mode="RGB+ED",
                            near_plane=self.near,
                            far_plane=self.far,
                            packed=False,
                            radius_clip=radius_clip,
                        )
                        color, forward_flow, weights, depth = renderings.split(
                            [self.gs_dim, 3, self.num_motion_tokens, 1], dim=-1
                        )
                        rendered_colors.append(color)
                        rendered_alphas.append(alpha)
                        rendered_flow.append(forward_flow)
                        motion_seg.append(weights)
                        rendered_depths.append(depth)
                color = torch.stack(rendered_colors, dim=0)
                rendered_alpha = torch.stack(rendered_alphas, dim=0)
                forward_flow = torch.stack(rendered_flow, dim=0)
                depth = torch.stack(rendered_depths, dim=0)
                motion_seg = torch.stack(motion_seg, dim=0)
                if motion_seg.numel() > 0:
                    motion_seg = motion_seg.reshape(b, tgt_t, v, h, w, -1).argmax(dim=-1)
                else:
                    motion_seg = None
            else:
                rendered_colors, rendered_alphas, rendered_flow, rendered_depths = [], [], [], []
                with torch.autocast("cuda", enabled=False):
                    for bid in range(means_batched.size(0)):
                        renderings, alpha, _ = rasterization(
                            means=means_batched[bid],
                            quats=quats_batched[bid],
                            scales=scales_batched[bid],
                            opacities=opacities_batched[bid],
                            colors=colors_batched[bid],
                            viewmats=viewmats_batched[bid],
                            Ks=Ks_batched[bid],
                            width=data_dict["width"],
                            height=data_dict["height"],
                            render_mode="RGB+ED",
                            near_plane=self.near,
                            far_plane=self.far,
                            packed=False,
                            radius_clip=radius_clip,
                        )
                        color, forward_flow, depth = renderings.split([self.gs_dim, 3, 1], dim=-1)
                        rendered_colors.append(color)
                        rendered_alphas.append(alpha)
                        rendered_flow.append(forward_flow)
                        rendered_depths.append(depth)
                color = torch.stack(rendered_colors, dim=0)
                rendered_alpha = torch.stack(rendered_alphas, dim=0)
                forward_flow = torch.stack(rendered_flow, dim=0)
                depth = torch.stack(rendered_depths, dim=0)

        else:
            if not self.training:
                with torch.autocast("cuda", enabled=False):
                    rendered_color, rendered_alpha, _ = rasterization(
                        means=means_batched.float(),
                        quats=quats_batched.float(),
                        scales=scales_batched.float(),
                        opacities=opacities_batched.float(),
                        colors=(
                            colors_batched[..., : -self.num_motion_tokens].float()
                            if self.num_motion_tokens > 0 and render_motion_seg
                            else colors_batched.float()
                        ),
                        viewmats=viewmats_batched,
                        Ks=Ks_batched,
                        width=tgt_w,
                        height=tgt_h,
                        render_mode="RGB+ED",
                        near_plane=self.near,
                        far_plane=self.far,
                        packed=False,
                        radius_clip=radius_clip,
                    )
                    color, forward_flow, depth = rendered_color.split([self.gs_dim, 3, 1], dim=-1)
                    if self.num_motion_tokens > 0 and render_motion_seg:
                        chunksize = 32
                        assignment_map = []
                        rendered_colors = colors_batched[..., -self.num_motion_tokens :]
                        for i in range(0, self.num_motion_tokens, chunksize):
                            weights, _, _ = rasterization(
                                means=means_batched.float(),
                                quats=quats_batched.float(),
                                scales=scales_batched.float(),
                                opacities=opacities_batched.float(),
                                colors=rendered_colors[..., i : i + chunksize],
                                viewmats=viewmats_batched,
                                Ks=Ks_batched,
                                width=tgt_w,
                                height=tgt_h,
                                render_mode="RGB+ED",
                                near_plane=self.near,
                                far_plane=self.far,
                                packed=False,
                                radius_clip=radius_clip,
                            )
                            weights = weights.split([weights.size(-1) - 1, 1], dim=-1)[0]
                            assignment_map.append(weights)
                        motion_seg = torch.cat(assignment_map, dim=-1)
                        motion_seg = motion_seg.reshape(b, tgt_t, tgt_v, tgt_h, tgt_w, -1).argmax(
                            dim=-1
                        )
            else:
                with torch.autocast("cuda", enabled=False):
                    rendered_color, rendered_alpha, _ = rasterization(
                        means=means_batched.float(),
                        quats=quats_batched.float(),
                        scales=scales_batched.float(),
                        opacities=opacities_batched.float(),
                        colors=colors_batched.float(),
                        viewmats=viewmats_batched,
                        Ks=Ks_batched,
                        width=tgt_w,
                        height=tgt_h,
                        render_mode="RGB+ED",
                        near_plane=self.near,
                        far_plane=self.far,
                        packed=False,
                        radius_clip=radius_clip,
                    )
                color, forward_flow, depth = rendered_color.split([self.gs_dim, 3, 1], dim=-1)
        output_dict = {
            "rendered_image": color.view(b, tgt_t, tgt_v, tgt_h, tgt_w, -1),
            "rendered_depth": depth.view(b, tgt_t, tgt_v, tgt_h, tgt_w),
            "rendered_alpha": rendered_alpha.view(b, tgt_t, tgt_v, tgt_h, tgt_w),
            "rendered_flow": forward_flow.view(b, tgt_t, tgt_v, tgt_h, tgt_w, -1),
            "means_batched": means_batched,
        }
        if motion_seg is not None:
            output_dict["rendered_motion_seg"] = motion_seg.squeeze(-1)
        return output_dict

    def voxelizaton_with_fusion(self, img_feat, pts3d, voxel_size, conf=None):
        # img_feat: V, C, H, W
        # pts3d: V, H* W, 3
        V, C, H, W = img_feat.shape
        pts3d_flatten = pts3d.flatten(0, 2)

        voxel_indices = (pts3d_flatten / voxel_size).round().int()  # [B*V*N, 3]
        unique_voxels, inverse_indices, counts = torch.unique(
            voxel_indices, dim=0, return_inverse=True, return_counts=True
        )

        # Flatten confidence scores and features
        conf_flat = conf.flatten()  # [B*V*N]
        anchor_feats_flat = img_feat.permute(0, 2, 3, 1).flatten(0, 2)  # [B*V*N, ...]

        # Compute softmax weights per voxel
        conf_voxel_max, _ = scatter_max(conf_flat, inverse_indices, dim=0)
        conf_exp = torch.exp(conf_flat - conf_voxel_max[inverse_indices])
        voxel_weights = scatter_add(
            conf_exp, inverse_indices, dim=0
        )  # [num_unique_voxels]
        weights = (conf_exp / (voxel_weights[inverse_indices] + 1e-6)).unsqueeze(
            -1
        )  # [B*V*N, 1]

        # Compute weighted average of positions and features
        weighted_pts = pts3d_flatten * weights
        weighted_feats = anchor_feats_flat.squeeze(1) * weights

        # Aggregate per voxel
        voxel_pts = scatter_add(
            weighted_pts, inverse_indices, dim=0
        )  # [num_unique_voxels, 3]
        voxel_feats = scatter_add(
            weighted_feats, inverse_indices, dim=0
        )  # [num_unique_voxels, feat_dim]

        return voxel_pts, voxel_feats


class ReconDrive_LITModelModule(pl.LightningModule):
    def __init__(self, cfg, save_dir='.', logger=None):
        super().__init__()
        self.read_config(cfg)

        # Set default values for ego transformation configuration if not in config
        if not hasattr(self, 'translate_3dgs'):
            self.translate_3dgs = False
        self.save_visualizations = False # True

        self.save_dir = save_dir
        vggt_checkpoint = getattr(self, 'vggt_checkpoint', './checkpoints/vggt.pt')
        self.model = ReconDriveModel(sh_degree=self.sh_degree,min_depth=self.min_depth, max_depth=self.max_depth, vggt_checkpoint=vggt_checkpoint) 
        self.lpips = LPIPS(net="vgg")
        self.ssim_fn = SSIMLoss(window_size=11,reduction='none')
        self.l1_fn = torch.nn.L1Loss(reduction='none')
        self.lpips.eval()
        self.project = Projection(self.batch_size, self.height, self.width)
        self.flow_reg_coeff = 0.005
        self.init_novel_view_mode()
        self.save_hyperparameters('cfg','save_dir')
        
        # Initialize SAM2 for vehicle segmentation
        self.sam2_predictor = None
        self.sam2_initialized = False
        # Delay initialization to avoid issues during model creation
        print("SAM2 will initialize on first use")

        self.camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                            'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
        
        # Performance optimization flags
        self.compute_alternative_flow = cfg.get('model_cfg', {}).get('compute_alternative_flow', False)  # Mode 2 flow for comparison
    
    def load_pretrained_checkpoint(self, checkpoint_path, strict=False, verbose=True):
       
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if strict:
            
            self.load_state_dict(checkpoint['state_dict'])
            if verbose:
                print(f"strict mode loading checkpoint: {checkpoint_path}")
        else:
            
            current_state_dict = self.state_dict()
            pretrained_state_dict = checkpoint['state_dict']

            matched_params = {}
            unmatched_params = []

            for name, param in pretrained_state_dict.items():
                if name in current_state_dict:
                    if current_state_dict[name].shape == param.shape:
                        matched_params[name] = param
                    else:
                        unmatched_params.append(f"{name}: shape mismatch {param.shape} vs {current_state_dict[name].shape}")
                else:
                    unmatched_params.append(f"{name}: not found in current model")

            
            self.load_state_dict(matched_params, strict=False)

        
    
    def save_training_step_images(self, batch_idx, batch_splating_data, batch_recontrast_data=None):
        """Save rendered images, GT images, and flow heatmaps during training/inference"""
        import torchvision.utils as vutils
        
        # Create directory if not exists
        os.makedirs(self.save_images_dir, exist_ok=True)
        
        # Save images for each frame and camera
        for frame_id in self.all_render_frame_ids:
            for cam_id in range(self.num_cams):
                # Get camera name
                cam_name = self.camera_names[cam_id] if cam_id < len(self.camera_names) else f'CAM_{cam_id}'

                # Get rendered and GT images
                if ('gaussian_color', frame_id, cam_id) in batch_splating_data:
                    rendered_imgs = batch_splating_data[('gaussian_color', frame_id, cam_id)]
                    gt_imgs = batch_splating_data[('groudtruth', frame_id, cam_id)]
                    
                    # Save first batch item only
                    if len(rendered_imgs) > 0:
                        # Save rendered image
                        rendered_path = os.path.join(
                            self.save_images_dir, 
                            f'step_{self.saved_steps_count}_batch_{batch_idx}_frame_{frame_id}_{cam_name}_rendered.png'
                        )
                        vutils.save_image(rendered_imgs[0], rendered_path)
                        
                        # Save GT image
                        gt_path = os.path.join(
                            self.save_images_dir, 
                            f'step_{self.saved_steps_count}_batch_{batch_idx}_frame_{frame_id}_{cam_name}_gt.png'
                        )
                        vutils.save_image(gt_imgs[0], gt_path)
                        
                        # Save flow heatmap if available
                        if batch_recontrast_data is not None and 'forward_flow' in batch_recontrast_data:
                            # Flow is organized as: all cameras for frame 0, then all cameras for frame 1, etc.
                            # So the index is: frame_id * num_cams + cam_id
                            flow_cam_idx = frame_id * self.num_cams + cam_id
                            cam_pixels = self.height * self.width
                            start_idx = flow_cam_idx * cam_pixels
                            end_idx = start_idx + cam_pixels
                            if start_idx < batch_recontrast_data['forward_flow'][0].shape[0]:
                                # Save Mode 1 flow (default)
                                flow_path = os.path.join(
                                    self.save_images_dir,
                                    f'step_{self.saved_steps_count}_batch_{batch_idx}_frame_{frame_id}_{cam_name}_flow_mode1.png'
                                )
                                flow_data = batch_recontrast_data['forward_flow'][0]  # First batch item
                                camera_flow = flow_data[start_idx:end_idx]
                                
                                self.save_flow_heatmap(
                                    camera_flow, 
                                    gt_imgs[0], 
                                    flow_path,
                                    batch_idx=batch_idx,
                                    frame_id=frame_id,
                                    cam_id=cam_id
                                )
        
        self.saved_steps_count += 1
        
        # Check if should terminate
        if self.saved_steps_count >= self.max_save_steps:
            print(f"[INFO] Reached maximum save steps ({self.max_save_steps}). Terminating training...")
            # Force exit
            import sys
            sys.exit(0)
    
    def init_novel_view_mode(self):
        self.recontrast_frame_ids = 0
        self.render_frame_ids = [0]
        self.render_cam_mode = 'origin'
        self.render_width = self.width
        self.render_height = self.height
        self.render_scale = 1.0
        self.render_shift_T = torch.eye(4,dtype=torch.float32).unsqueeze(0)
        self.render_shift_x = 0.0
        self.render_shift_y = 0.0

    def read_config(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)

        # Calculate time_delta from context_span (assumes 12Hz sampling rate)
        self.time_delta = getattr(self, 'context_span', 6) / 12.0
        # Set default for use_vehicle_flow if not in config
        if not hasattr(self, 'use_vehicle_flow'):
            self.use_vehicle_flow = True
    
    def detect_valid_frames(self, inputs):
        """Dynamically detect valid frames based on actual data shape"""
        # Get the number of frames from the data tensor shape
        if ('color_aug', 0) in inputs:
            print(inputs[('color_aug', 0)].shape)
            b, s, c, h, w = inputs[('color_aug', 0)].shape
            total_frames = s // self.num_cams
            return list(range(total_frames))
        else:
            # Fallback to default if data structure is different
            return [0, 1, 2, 3, 4, 5, 6]

    def set_normal_params(self, data_dict):
        inputs = data_dict['all_dict']
        
        # Detect valid frames with actual data
        valid_frames = self.detect_valid_frames(inputs)
        
        # Update render frame IDs based on actual valid frames
        self.all_render_frame_ids = valid_frames
        self.context_span = len(valid_frames) - 1
        
        # Context frames are always first and last valid frames
        if len(valid_frames) >= 2:
            self.all_context_frame_ids = [valid_frames[0], valid_frames[-1]]
        else:
            # Edge case: if only one valid frame
            self.all_context_frame_ids = [valid_frames[0], valid_frames[0]]
        
        outputs = {}

    # TODO: Hardcode setting here
    def prob_sample_rendered_ids(self):
        prob_all_render_frame_ids = [0.7, 0.3, 0.2, 0.1, 0.1, 0.05, 0]

        # For multi-GPU training: use different random values for each GPU
        # Get the global rank to ensure different GPUs get different samples
        if hasattr(self, 'global_rank') and self.global_rank is not None:
            # Create a temporary random state based on global_rank and current step
            # This ensures different GPUs get different samples while maintaining reproducibility
            rng = np.random.RandomState()
            # Use a combination of training step (or epoch) and rank for seed
            # This way each GPU gets different samples, but same GPU gets same sequence
            current_step = self.global_step if hasattr(self, 'global_step') else 0
            temp_seed = hash((current_step, self.global_rank)) % (2**32)
            rng.seed(temp_seed)
            render_prob = rng.rand(7)
        else:
            # Single GPU or inference mode - use global random state
            render_prob = np.random.rand(7)

        all_render_frame_ids_mask = render_prob < prob_all_render_frame_ids

        selected_ids = np.nonzero(all_render_frame_ids_mask)[0].tolist()
        # Ensure at least one sample ID is selected
        if len(selected_ids) == 0:
            # If no IDs were selected, select based on weighted probabilities
            valid_probs = prob_all_render_frame_ids[:6]  # Exclude last one with 0 probability
            probabilities = np.array(valid_probs)
            probabilities = probabilities / probabilities.sum()  # Normalize to sum to 1
            if hasattr(self, 'global_rank') and self.global_rank is not None:
                selected_ids = [rng.choice(6, p=probabilities)]
            else:
                selected_ids = [np.random.choice(6, p=probabilities)]

        # Limit maximum number of selected frames to 4 to avoid CUDA OOM
        if len(selected_ids) > 4:
            if hasattr(self, 'global_rank') and self.global_rank is not None:
                selected_ids = sorted(rng.choice(selected_ids, size=4, replace=False).tolist())
            else:
                selected_ids = sorted(np.random.choice(selected_ids, size=4, replace=False).tolist())

        # Add rank info to debug message for multi-GPU
        if hasattr(self, 'global_rank') and self.global_rank is not None:
            print(f"[GPU {self.global_rank}] Sampling rendered ids: {selected_ids}")
        else:
            print(f"Sampling rendered ids: {selected_ids}")

        self.all_render_frame_ids = selected_ids
        self.context_span = 6
         

    def training_step(self, batch_input, batch_idx):
        self.stage = stage = 'train'

        self.prob_sample_rendered_ids()
        self._log_weights_and_grads(batch_input)

        batch_recontrast_data = self.get_recontrast_data(batch_input, batch_idx)

        loss_norm = self.compute_norm_loss(batch_recontrast_data)

        batch_render_data = self.get_render_data(batch_input)
        batch_splating_data  = self.render_splating_imgs(batch_recontrast_data,batch_render_data)

        loss_gaussian = self.compute_gaussian_loss(batch_splating_data)
        loss_depth = torch.tensor(0.0, device=self.device)

        batch_render_project_data = self.render_project_imgs(batch_input, batch_recontrast_data)
        loss_project = self.compute_project_loss(batch_render_project_data)


        self.log(f'{stage}/gs', loss_gaussian.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/proj', loss_project.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/norm', loss_norm.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        # Exclude projection loss from total loss
        loss_all = loss_gaussian + loss_depth + loss_project + loss_norm
        psnr, ssim, lpips = self.compute_reconstruction_metrics(batch_splating_data,stage)

        del batch_input, batch_recontrast_data, batch_render_data, batch_render_project_data, batch_splating_data, psnr, ssim, lpips

        return loss_all

    def predict_step(self, batch_input, batch_idx):
        self.stage = 'predict'
        self.set_normal_params(batch_input)
        self.init_novel_view_mode()

        # Save all valid frames
        all_frames = self.all_render_frame_ids.copy()

        # Get recontrast data (shared across both modes)
        batch_recontrast_data = self.get_recontrast_data(batch_input)

        # === Mode 1: Scene Reconstruction (frame 0) ===
        self.all_render_frame_ids = [0]
        batch_render_data_recon = self.get_render_data(batch_input)
        batch_splating_data_recon = self.render_splating_imgs(batch_recontrast_data, batch_render_data_recon)

        # === Mode 2: Novel View Synthesis (middle frames) ===
        if len(all_frames) > 2:
            self.all_render_frame_ids = all_frames[1:-1]
            batch_render_data_novel = self.get_render_data(batch_input)
            batch_splating_data_novel = self.render_splating_imgs(batch_recontrast_data, batch_render_data_novel)

        # Combine both modes' data for return
        # Merge reconstruction and novel view data
        batch_render_data = batch_render_data_recon.copy()
        batch_splating_data = batch_splating_data_recon.copy()

        if len(all_frames) > 2:
            # Add novel view data to the combined dictionaries
            batch_render_data.update(batch_render_data_novel)
            batch_splating_data.update(batch_splating_data_novel)

        # Restore all_render_frame_ids to include all frames
        self.all_render_frame_ids = all_frames

        return batch_recontrast_data, batch_render_data, batch_splating_data

    def validation_step(self, batch_input, batch_idx):
        self.stage = stage = 'val'
        # Haibao: hardcode
        context_span = 6
        self.all_render_frame_ids = range(0, context_span)


        self.set_normal_params(batch_input)
        self.init_novel_view_mode()
        batch_recontrast_data = self.get_recontrast_data(batch_input)

        batch_render_data = self.get_render_data(batch_input)

        loss_norm = self.compute_norm_loss(batch_recontrast_data)
        # loss_flow_reg = self.compute_flow_reg_loss(batch_recontrast_data)
        
        # Comment out projection loss computation as it's not useful
        batch_render_project_data = self.render_project_imgs(batch_input,batch_recontrast_data)
        loss_project = self.compute_project_loss(batch_render_project_data)

        batch_splating_data  = self.render_splating_imgs(batch_recontrast_data,batch_render_data)

        loss_depth = self.compute_depth_loss(batch_splating_data)
        loss_gaussian = self.compute_gaussian_loss(batch_splating_data)

        self.log(f'{stage}/gs', loss_gaussian.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/proj', loss_project.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/depth', loss_depth.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}/norm', loss_norm.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        # Exclude projection loss from total loss
        loss_all = loss_gaussian + loss_depth + loss_norm + loss_project
        psnr, ssim, lpips = self.compute_reconstruction_metrics(batch_splating_data,stage)

        del batch_input,batch_recontrast_data, batch_render_data, batch_render_project_data, batch_splating_data, psnr, ssim, lpips

        return loss_all

    def test_step(self, batch_input, batch_idx):
        self.stage = stage = 'test'
        # Haibao: hardcode
        context_span = 6
        self.all_render_frame_ids = range(0, context_span)

        self.set_normal_params(batch_input)
        self.init_novel_view_mode()
        batch_recontrast_data = self.get_recontrast_data(batch_input)

        batch_render_data = self.get_render_data(batch_input)

        loss_norm = self.compute_norm_loss(batch_recontrast_data)

        # Comment out projection loss computation as it's not useful
        batch_render_project_data = self.render_project_imgs(batch_input,batch_recontrast_data)
        loss_project = self.compute_project_loss(batch_render_project_data)

        batch_splating_data  = self.render_splating_imgs(batch_recontrast_data,batch_render_data)

        loss_depth = self.compute_depth_loss(batch_recontrast_data)
        loss_gaussian = self.compute_gaussian_loss(batch_splating_data)

        self.log(f'{stage}/gs', loss_gaussian.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/proj', loss_project.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/depth', loss_depth.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}/norm', loss_norm.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        loss_all = loss_gaussian + loss_depth + loss_norm + loss_project
        psnr, ssim, lpips = self.compute_reconstruction_metrics(batch_splating_data,stage)

        del batch_input,batch_recontrast_data, batch_render_data, batch_render_project_data, batch_splating_data, psnr, ssim, lpips

        return loss_all

    def _log_weights_and_grads(self, inputs):
        current_step = self.global_step
        max_weight = -np.inf
        max_grad = -np.inf
        max_weight_name = ""
        max_grad_name = ""
        nan_params = []
        inf_params = []

        for name ,val in inputs.items():
            if isinstance(val,torch.Tensor):
                if torch.isnan(val).any():
                    nan_params.append(name)        
                if torch.isinf(val).any():
                    inf_params.append(name)     

       
        if len(nan_params)>0 or len(inf_params)>0:
            print('nan_prams: ',nan_params)
            print('inf_prams: ',inf_params)
            
            sys.exit(-1)
        

        
        for optimizer in self.trainer.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                for j, param in enumerate(param_group['params']):
                    state = optimizer.state[param]
                    if 'exp_avg' in state and torch.isnan(state['exp_avg']).any():
                        nan_params.append(f'exp_avg:param_group={i}, param={j}')
                    if 'exp_avg_sq' in state and torch.isnan(state['exp_avg_sq']).any():
                        nan_params.append(f'exp_avg_sq:param_group={i}, param={j}')

                    if 'exp_avg' in state and torch.isinf(state['exp_avg']).any():
                        inf_params.append(f'exp_avg:param_group={i}, param={j}')
                    if 'exp_avg_sq' in state and torch.isinf(state['exp_avg_sq']).any():
                        inf_params.append(f'exp_avg_sq:param_group={i}, param={j}')

        
        if len(nan_params)>0 or len(inf_params)>0:
            print('nan_prams: ',nan_params)
            print('inf_prams: ',inf_params)
            
            sys.exit(-1)

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
                
           
            if torch.isnan(param.data).any() or torch.isnan(param.grad).any():
                nan_params.append(name)
            if torch.isinf(param.data).any() or torch.isinf(param.grad).any():
                inf_params.append(name)
            
            
            param_max = param.data.abs().max().item()
            if param_max > max_weight:
                max_weight = param_max
                max_weight_name = name
                
            
            grad_max = param.grad.data.abs().max().item()
            if grad_max > max_grad:
                max_grad = grad_max
                max_grad_name = name
        
        if len(nan_params)>0 or len(inf_params)>0:
            print('nan_prams: ',nan_params)
            print('inf_prams: ',inf_params)
            
            sys.exit(-1)
    
    def on_before_optimizer_step(self, optimizer):
        
        valid_gradients = True
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaN gradient: {name}")
                    valid_gradients = False
                if torch.isinf(param.grad).any():
                    print(f"Inf gradient: {name}")
                    valid_gradients = False
        
        if not valid_gradients:
            print("skipping ...")
           
            optimizer.zero_grad()
            return False
        return True
                        
    def configure_optimizers(self):
        # Collect all trainable parameters (simplified, no parameter groups)
        trainable_params = []
        trainable_param_names = []

        for name, parameters in self.model.named_parameters():
            if parameters.requires_grad:
                trainable_params.append(parameters)
                trainable_param_names.append(name)
                print(f'Training parameter: {name}')

        if self.auto_scale_lr:
            num_devices = self.trainer.num_devices
            scale_devices = max(1, log2(num_devices))  
            base_lr = self.learning_rate * scale_devices
        else:
            base_lr = self.learning_rate

        print(f"\nOptimizer configuration (simplified):")
        print(f"  Total trainable parameters: {len(trainable_params)}")
        print(f"  Learning rate: {base_lr}")
        print(f"  Using single learning rate for all parameters")
        
        # Verify we're training depth_head and gs_head
        depth_head_count = sum(1 for name in trainable_param_names if 'depth_head' in name)
        gs_head_count = sum(1 for name in trainable_param_names if 'gs_head' in name)
        other_count = len(trainable_param_names) - depth_head_count - gs_head_count
        print(f"  - depth_head parameters: {depth_head_count}")
        print(f"  - gs_head parameters: {gs_head_count}")
        if other_count > 0:
            print(f"  - WARNING: {other_count} other parameters are also trainable")

        if not trainable_params:
            print("ERROR: No trainable parameters found!")
            # Create dummy parameter to avoid crash
            trainable_params = [torch.nn.Parameter(torch.zeros(1))]

        # Create simple optimizer without parameter groups
        optimizer = optim.AdamW(trainable_params, lr=base_lr, betas=(0.9,0.98), eps=1e-7, weight_decay=self.weight_decay)

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=self.scheduler_step_size,gamma=self.scheduler_gamma)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=self.lr_restart_epoch,  
                    T_mult=self.lr_restart_mult,
                    eta_min=base_lr*self.lr_min_factor*0.1  
                )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
    
    def init_sam2(self):
        """Initialize SAM2 model for vehicle segmentation"""
        try:
            checkpoint = getattr(self, 'sam2_checkpoint', None)
            model_cfg = getattr(self, 'sam2_model_cfg', "configs/sam2.1/sam2.1_hiera_s.yaml")
            sam2_dir = getattr(self, 'sam2_dir', None)

            # Auto-detect SAM2 directory if not specified
            if sam2_dir is None:
                import sam2
                sam2_dir = os.path.dirname(os.path.dirname(sam2.__file__))

            # Construct full checkpoint path if relative
            if checkpoint and not os.path.isabs(checkpoint):
                checkpoint = os.path.join(sam2_dir, checkpoint)

            # Check if files exist
            if checkpoint and os.path.exists(checkpoint):
                # Temporarily disable deterministic algorithms for SAM2 initialization
                prev_det = torch.are_deterministic_algorithms_enabled()
                if prev_det:
                    torch.use_deterministic_algorithms(False)

                # Change to SAM2 directory for config loading
                original_dir = os.getcwd()
                os.chdir(sam2_dir)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                sam2_model = build_sam2(model_cfg, checkpoint)
                sam2_model = sam2_model.to(device)

                # Freeze SAM2 model parameters to prevent training
                sam2_model.eval()
                for param in sam2_model.parameters():
                    param.requires_grad = False

                self.sam2_predictor = SAM2ImagePredictor(sam2_model)

                # Change back to original directory
                os.chdir(original_dir)

                # Restore deterministic setting
                if prev_det:
                    torch.use_deterministic_algorithms(True, warn_only=True)

                print("SAM2 initialized and frozen (not trainable)")
            else:
                self.sam2_predictor = None
        except Exception:
            self.sam2_predictor = None
    
    def segment_vehicles_with_sam2(self, image, bbox_2d_list=None):
        """
        Segment vehicles in the image using SAM2 or simple box masks
        Args:
            image: numpy array [H, W, 3] or tensor
            bbox_2d_list: list of 2D bounding boxes for vehicles (optional)
        Returns:
            vehicle_masks: list of boolean masks for each vehicle
        """
        # Initialize SAM2 on first use if not already initialized
        if not self.sam2_initialized and self.sam2_predictor is None:
            try:
                # Temporarily disable deterministic algorithms for initialization
                prev_det = torch.are_deterministic_algorithms_enabled()
                if prev_det:
                    torch.use_deterministic_algorithms(False)
                
                self.init_sam2()
                self.sam2_initialized = True
                
                # Restore deterministic setting
                if prev_det:
                    torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                self.sam2_predictor = None
                self.sam2_initialized = True  # Mark as attempted
        
        # If SAM2 is not available, use simple box-based masks
        if self.sam2_predictor is None:
            if not bbox_2d_list or len(bbox_2d_list) == 0:
                return []
            
            # Get image dimensions
            if torch.is_tensor(image):
                if image.shape[0] == 3:  # CHW
                    h, w = image.shape[1], image.shape[2]
                else:  # HWC
                    h, w = image.shape[0], image.shape[1]
            else:
                h, w = image.shape[:2]
            
            # Create simple box masks
            vehicle_masks = []
            for bbox in bbox_2d_list:
                # Convert bbox to list of coordinates - handle various formats
                try:
                    if torch.is_tensor(bbox):
                        bbox_vals = bbox.cpu().tolist()
                    elif isinstance(bbox, np.ndarray):
                        bbox_vals = bbox.tolist()
                    elif isinstance(bbox, (list, tuple)):
                        bbox_vals = list(bbox)
                    else:
                        continue  # Skip unknown bbox type
                    
                    # Ensure we have exactly 4 values
                    if len(bbox_vals) != 4:
                        continue  # Skip invalid bbox
                except Exception:
                    continue  # Skip on error
                
                # Create mask from bounding box
                mask = np.zeros((h, w), dtype=bool)
                x1, y1, x2, y2 = [int(coord) for coord in bbox_vals]
                
                # Check if bbox is at least partially within image bounds
                # A vehicle is partially visible if any part of its bbox overlaps with image
                if x2 > 0 and x1 < w and y2 > 0 and y1 < h:
                    # Clip to image bounds for the visible portion
                    x1_clipped = max(0, x1)
                    x2_clipped = min(w, x2)
                    y1_clipped = max(0, y1)
                    y2_clipped = min(h, y2)
                    
                    # Only create mask if there's a valid visible area
                    if x2_clipped > x1_clipped and y2_clipped > y1_clipped:
                        mask[y1_clipped:y2_clipped, x1_clipped:x2_clipped] = True
                        vehicle_masks.append(mask)
                    else:
                        # Vehicle has no visible pixels, but we still need to track it for velocity
                        vehicle_masks.append(mask)  # Empty mask
                else:
                    # Vehicle is completely outside image bounds
                    vehicle_masks.append(mask)  # Empty mask
            
            return vehicle_masks
        
        # Convert tensor to numpy if needed
        if torch.is_tensor(image):
            image_np = image.cpu().numpy()
            if image_np.shape[0] == 3:  # CHW to HWC
                image_np = np.transpose(image_np, (1, 2, 0))
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image
        
        h, w = image_np.shape[:2]
        vehicle_masks = []
        
        # Temporarily disable deterministic mode for all SAM2 operations
        prev_deterministic = torch.are_deterministic_algorithms_enabled()
        if prev_deterministic:
            torch.use_deterministic_algorithms(False, warn_only=True)  # Use warn_only to avoid crash
        
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                self.sam2_predictor.set_image(image_np)
            
            if bbox_2d_list and len(bbox_2d_list) > 0:
                # if len(vehicle_masks) == 0 and len(bbox_2d_list) > 0:
                #     for i, bbox in enumerate(bbox_2d_list[:3]):
                #         if isinstance(bbox, (list, tuple, np.ndarray)):
                #             print(f"  BBox {i}: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                
                # Use provided bounding boxes as prompts
                for bbox in bbox_2d_list:
                    # Convert bbox to SAM2 format (xyxy)
                    if torch.is_tensor(bbox):
                        bbox_np = bbox.cpu().numpy()
                    elif isinstance(bbox, (list, tuple)):
                        bbox_np = np.array(bbox, dtype=np.float32)
                    elif isinstance(bbox, np.ndarray):
                        bbox_np = bbox.astype(np.float32)
                    else:
                        continue
                    
                    box_prompt = bbox_np.reshape(1, 4)
                    
                    masks, scores, _ = self.sam2_predictor.predict(
                        box=box_prompt,
                        multimask_output=False
                    )
                    
                    if scores[0] > 0.5:  # Lowered threshold for more detections
                        # Convert mask to boolean
                        mask_bool = masks[0].astype(bool)
                        vehicle_masks.append(mask_bool)
                    else:
                        # Even if score is low, add an empty mask to maintain alignment with bboxes
                        vehicle_masks.append(np.zeros((h, w), dtype=bool))
            # No automatic detection - only use provided bboxes
        finally:
            # Restore deterministic setting
            if prev_deterministic:
                torch.use_deterministic_algorithms(True, warn_only=True)  # Use warn_only to avoid crash
        
        return vehicle_masks

    
    def compute_velocity_flow(self, vehicle_masks, vehicle_velocities, image_shape):
        """
        Assign 3D velocity to vehicle pixels
        Args:
            vehicle_masks: list of boolean masks for each vehicle
            vehicle_velocities: list of 3D velocities [vx, vy, vz] in ego frame (m/s)
            image_shape: (H, W) of the image
        Returns:
            flow_3d: 3D velocity map [H, W, 3] in ego frame (m/s)
        """
        h, w = image_shape
        flow_3d = np.zeros((h, w, 3), dtype=np.float32)

        # Assign velocity to each vehicle mask
        # Note: Velocity is in m/s, will be scaled by delta_t during rendering
        for mask, velocity in zip(vehicle_masks, vehicle_velocities):
            if mask is not None and velocity is not None:
                mask = mask.astype(bool) if mask.dtype != bool else mask
                vel_array = np.array(velocity, dtype=np.float32)
                # Store velocity directly (not displacement)
                flow_3d[mask] = vel_array

        return flow_3d
    
    def get_recontrast_data(self, data_dict, batch_idx=0):
        """
        This function computes recontrast data for each viewpoint.
        """
        inputs = data_dict['context_frames']
        outputs = {}

        image_list = []
        c2e_extr_list = []

        for frame_cam_id in range(inputs[('color_aug', 0)].shape[1]):
            c2e_extr = inputs['c2e_extr'][:, frame_cam_id, ...]
            image_list.append(inputs[(f'color_aug', 0)][:,frame_cam_id,...])
            c2e_extr_list.append(c2e_extr)
        image_list = torch.stack(image_list,dim=1)

        # 6 -> 18
        # [4, 6, 280, 518, 1], [4, 6, 280, 518, 4], [4, 6, 280, 518, 3], [4, 6, 280, 518, 1], [4, 6, 280, 518, 3, 25], [4, 6, 280, 518, 3]
        depth_maps, rot_maps, scale_maps, opacity_maps, sh_maps, forward_flow = self.model(image_list)
        del image_list
        batch_size = depth_maps.shape[0]
        frame_camrea = depth_maps.shape[1]

        c2e_extr_list = torch.stack(c2e_extr_list, dim=1) # b, s, 4, 4

        bfc_depth_maps = rearrange(depth_maps.squeeze(-1), 'b c h w -> (b c) h w ')
        bfc_K = rearrange(inputs['K'], 'b c i j -> (b c) i j ')
        bfc_c2e = rearrange(c2e_extr_list, 'b c i j -> (b c) i j ')
        bfc_sh = rearrange(sh_maps, 'b c h w p d -> (b c) h w p d') # height weight points d_sh

        # bfc_xyz = self._unproject_depth_map_to_points_map(bfc_depth_maps, bfc_K, bfc_c2e)
        bf_e2c = torch.linalg.inv(bfc_c2e)
        bfc_xyz = depth2pc(bfc_depth_maps, bf_e2c, bfc_K)

        c2w_rotations = rearrange(bfc_c2e[:, :3, :3], "b i j -> b () () () i j")

        bfc_sh = rotate_sh(bfc_sh, c2w_rotations)

        # Transform rot_maps from camera frame to ego frame
        # rot_maps shape: [batch, num_cams, h, w, 4]
        # bfc_c2e shape: [(batch*num_cams), 4, 4]
        bfc_rot_maps = rearrange(rot_maps, 'b c h w d -> (b c) (h w) d', d=4)

        outputs['pred_depths'] = rearrange(bfc_depth_maps, '(b c) h w -> b (c h w)', b=batch_size, c=frame_camrea)#.contiguous()

        outputs['xyz'] = rearrange(bfc_xyz, '(b c) p k -> b (c p) k', b=batch_size, c=frame_camrea)#.contiguous()
        outputs['rot_maps'] = rearrange(bfc_rot_maps, '(b c) p d -> b (c p) d', b=batch_size, c=frame_camrea, d=4)#.contiguous()

        outputs['scale_maps'] = rearrange(scale_maps, 'b c h w d -> b (c h w) d', d=3)#.contiguous()
        outputs['opacity_maps'] = rearrange(opacity_maps, 'b c h w d -> b (c h w) d')#.contiguous()
        outputs['sh_maps'] = rearrange(bfc_sh, '(b c) h w p d -> b (c h w) d p', b=batch_size, c=frame_camrea)#.contiguous()

        # Generate vehicle-based 3D velocity flow
        if self.use_vehicle_flow:
            new_forward_flow = []

            # ALWAYS compute vehicle masks using SAM2 for correct flow application
            # The masks are essential for applying velocity to the correct pixels
            compute_vehicle_masks = True  # Always compute for proper flow

            all_vehicle_masks = []

            for b in range(batch_size):
                batch_flows = []
                batch_masks = []  # Always collect masks for unified format

                for c in range(frame_camrea):
                    # Determine frame index and camera index
                    frame_idx = c // self.num_cams  # 0 for frame 0, 1 for frame context_span
                    cam_idx = c % self.num_cams

                    # Get image for segmentation
                    color_tensor = inputs.get(('color_aug', 0), torch.zeros(batch_size, frame_camrea, 3, self.height, self.width))
                    seg_img = color_tensor[b, c]

                    # Initialize outputs
                    vehicle_masks = []
                    vehicle_velocities = []

                    # Determine which frame-specific annotations to use
                    if frame_idx == 0:
                        anno_key = 'vehicle_annotations_frame_0'
                    else:
                        anno_key = f'vehicle_annotations_frame_{self.context_span}'

                    # Fallback to combined annotations if frame-specific not available
                    if anno_key not in inputs and 'vehicle_annotations' in inputs:
                        anno_key = 'vehicle_annotations'
                        lookup_idx = c
                    else:
                        lookup_idx = cam_idx

                    # Process vehicle annotations if available
                    if anno_key in inputs and b < len(inputs[anno_key]):
                        try:
                            batch_data = inputs[anno_key][b]

                            if lookup_idx < len(batch_data):
                                vehicle_data = batch_data[lookup_idx]

                                # Extract bounding boxes and velocities
                                bbox_2d_list = []
                                raw_velocities = []
                                vehicle_depths = []
                                vehicle_intrinsics = []

                                if isinstance(vehicle_data, list):
                                    for vehicle in vehicle_data:
                                        if isinstance(vehicle, dict) and 'bbox_2d' in vehicle:
                                            bbox_2d_list.append(vehicle['bbox_2d'])
                                            vel = vehicle.get('velocity', [0, 0, 0])
                                            # Ensure velocity is 3D
                                            if isinstance(vel, (list, tuple)) and len(vel) > 3:
                                                vel = vel[:3]
                                            raw_velocities.append(vel)
                                            vehicle_depths.append(vehicle.get('depth', 10.0))
                                            vehicle_intrinsics.append(vehicle.get('camera_intrinsic', None))

                                # Create masks using SAM2
                                vehicle_masks = self.segment_vehicles_with_sam2(seg_img, bbox_2d_list if bbox_2d_list else None)
                                vehicle_velocities = raw_velocities
                        except Exception:
                            pass  # Skip if annotations not accessible

                    # Ensure we have masks and velocities aligned
                    if len(vehicle_velocities) < len(vehicle_masks):
                        vehicle_velocities += [[0.0, 0.0, 0.0]] * (len(vehicle_masks) - len(vehicle_velocities))

                    # Compute 3D velocity flow
                    flow = self.compute_velocity_flow(
                        vehicle_masks, 
                        vehicle_velocities,
                        (self.height, self.width)
                    )

                    batch_flows.append(torch.from_numpy(flow).to(depth_maps.device))

                    # Store combined mask for inference (always needed)
                    combined_mask = np.zeros((self.height, self.width), dtype=bool)
                    for mask in vehicle_masks:
                        if mask is not None:
                            combined_mask |= mask
                    batch_masks.append(torch.from_numpy(combined_mask).to(depth_maps.device))

                new_forward_flow.append(torch.stack(batch_flows))
                if len(batch_masks) > 0:
                    all_vehicle_masks.append(torch.stack(batch_masks))
                else:
                    # Create empty masks if no masks were collected
                    empty_masks = torch.zeros(frame_camrea, self.height, self.width, dtype=torch.bool, device=depth_maps.device)
                    all_vehicle_masks.append(empty_masks)

            # Reshape to final format [b, (c*h*w), 3]
            outputs['forward_flow'] = rearrange(torch.stack(new_forward_flow), 'b c h w d -> b (c h w) d')
            # Store vehicle masks as tensor for unified format [b, c, h, w]
            if len(all_vehicle_masks) > 0:
                outputs['vehicle_masks'] = torch.stack(all_vehicle_masks)
            else:
                outputs['vehicle_masks'] = None

        else:
            # Use original flow from model
            outputs['forward_flow'] = rearrange(forward_flow, 'b c h w d -> b (c h w) d')#.contiguous()
            outputs['vehicle_masks'] = None  # No vehicle masks when not using vehicle flow

        # Perform ICP refinement early if we have multiple frames (needed for ego pose and velocity refinement)
        ego_T_ego_key = ('ego_T_ego', 0, self.context_span)
        if frame_camrea > self.num_cams and ego_T_ego_key in inputs:
            # Get ego_T_ego transformations
            ego_T_ego_0toN_initial = inputs[ego_T_ego_key]

            # Store the transformation being used (no refinement)
            outputs['ego_T_ego_original'] = ego_T_ego_0toN_initial
        
        if frame_camrea > self.num_cams:
            num_frames = frame_camrea // self.num_cams
            if num_frames != 2:
                raise NotImplementedError(f"Context frames should have exactly 2 frames (frame 0 and frame {self.context_span}), but got {num_frames} frames")

            xyz_transformed = outputs['xyz'].clone()
            if self.translate_3dgs:
                rot_maps_transformed = outputs['rot_maps'].clone()
                sh_maps_transformed = outputs['sh_maps'].clone()
            mid_point = xyz_transformed.shape[1] // 2


            # Check the original input to see if we have per-camera transformations
            ego_T_ego_key = ('ego_T_ego', 0, self.context_span)
            ego_T_ego_0toN_input = inputs.get(ego_T_ego_key, None)
            if ego_T_ego_0toN_input is not None:
                # Check if original input has per-camera transformations
                if ego_T_ego_0toN_input.dim() == 4:
                    # [batch_size, num_cameras, 4, 4] - Use camera-specific transformations
                    batch_size = xyz_transformed.shape[0]
                    points_per_camera = self.height * self.width

                    # Check if we have refined per-camera transformations
                    use_refined = 'ego_T_ego_refined' in outputs
                    if use_refined:
                        refined_transforms = outputs['ego_T_ego_refined']

                    # Transform each camera's frame N points separately
                    for cam_id in range(self.num_cams):
                        cam_start = mid_point + cam_id * points_per_camera
                        cam_end = mid_point + (cam_id + 1) * points_per_camera

                        # Get camera-specific transformation: ego0 → egoN
                        # Use refined transformation if available
                        if use_refined:
                            ego_T_ego_0toN_cam = refined_transforms[:, cam_id]
                        else:
                            ego_T_ego_0toN_cam = ego_T_ego_0toN_input[:, cam_id]  # [batch_size, 4, 4]

                        # Invert to get: egoN → ego0
                        ego_T_ego_Nto0_cam = torch.linalg.inv(ego_T_ego_0toN_cam)

                        # Transform this camera's frame N points to frame 0 ego coordinates
                        xyz_transformed[:, cam_start:cam_end, :] = self.transform_points(
                            xyz_transformed[:, cam_start:cam_end, :],
                            ego_T_ego_Nto0_cam
                        )
                    if self.translate_3dgs:
                        if ego_T_ego_Nto0_cam.dim() == 2:
                            R_Nto0 = ego_T_ego_Nto0_cam[:3, :3]  # [3, 3]
                        else:
                            R_Nto0 = ego_T_ego_Nto0_cam[:, :3, :3]  # [B, 3, 3]

                        try:
                            from pytorch3d.transforms import matrix_to_quaternion
                            q_Nto0 = matrix_to_quaternion(torch.linalg.inv(R_Nto0)) # [3,] or [B, 4]
                        except ImportError:
                            def matrix_to_quaternion_manual(R):
                                # R: [..., 3, 3]
                                tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
                                qw = torch.sqrt((tr + 1.0).clamp(min=0)) / 2.0
                                qx = (R[..., 2, 1] - R[..., 1, 2]) / (4 * qw + 1e-8)
                                qy = (R[..., 0, 2] - R[..., 2, 0]) / (4 * qw + 1e-8)
                                qz = (R[..., 1, 0] - R[..., 0, 1]) / (4 * qw + 1e-8)
                                return torch.stack([qw, qx, qy, qz], dim=-1)
                            q_Nto0 = matrix_to_quaternion_manual(R_Nto0) 

                        if rot_maps_transformed[:, cam_start:cam_end, :].dim() == 2:
                            # [N, 4]
                            if q_Nto0.dim() == 1:
                                q_Nto0 = q_Nto0.unsqueeze(0)  # [1, 4]
                            q_Nto0 = q_Nto0.expand(rot_maps_transformed[:, cam_start:cam_end, :].shape[0], -1)  # [N, 4]
                        else:
                            # [B, N, 4]
                            if q_Nto0.dim() == 1:
                                q_Nto0 = q_Nto0.unsqueeze(0).unsqueeze(0)  # [1, 1, 4]
                            elif q_Nto0.dim() == 2:
                                q_Nto0 = q_Nto0.unsqueeze(1)  # [B, 1, 4]
                            q_Nto0 = q_Nto0.expand(-1, rot_maps_transformed[:, cam_start:cam_end, :].shape[1], -1)  # [B, N, 4]

                        def quat_multiply(q1, q2):
                            w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
                            w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
                            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
                            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
                            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
                            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
                            return torch.stack([w, x, y, z], dim=-1)

                        rot_maps_slice = rot_maps_transformed[:, cam_start:cam_end, :].clone()
                        rot_maps_transformed_slice = quat_multiply(q_Nto0, rot_maps_slice)
                        rot_maps_transformed = torch.cat([
                            rot_maps_transformed[:, :cam_start, :],
                            rot_maps_transformed_slice,
                            rot_maps_transformed[:, cam_end:, :]
                        ], dim=1)

                        outputs['rot_maps_transformed'] = rot_maps_transformed
                        outputs['sh_maps_transformed'] = sh_maps_transformed
                    
                    outputs['xyz_transformed'] = xyz_transformed
                else:
                    # [batch_size, 4, 4] or [4, 4] - Use unified transformation for all cameras
                    # Prefer refined transformation if available
                    if 'ego_T_ego_refined' in outputs:
                        refined = outputs['ego_T_ego_refined']
                        # If refined is per-camera [batch_size, num_cameras, 4, 4], use camera 0
                        if refined.dim() == 4:
                            ego_T_ego_0toN = refined[:, 0]  # Use camera 0's refined transformation
                        else:
                            ego_T_ego_0toN = refined
                    else:
                        ego_T_ego_0toN = ego_T_ego_0toN_input

                    # Ensure batch dimension exists
                    if ego_T_ego_0toN.dim() == 2:
                        ego_T_ego_0toN = ego_T_ego_0toN.unsqueeze(0)
                    elif ego_T_ego_0toN.dim() == 3 and ego_T_ego_0toN.shape[0] == 1:
                        # Already has batch dimension of 1, expand to match batch size
                        batch_size = xyz_transformed.shape[0]
                        if batch_size > 1:
                            ego_T_ego_0toN = ego_T_ego_0toN.expand(batch_size, -1, -1)

                    ego_T_ego_Nto0 = torch.linalg.inv(ego_T_ego_0toN)
                    xyz_transformed[:, mid_point:, :] = self.transform_points(
                        xyz_transformed[:, mid_point:, :],
                        ego_T_ego_Nto0
                    )
                    outputs['xyz_transformed'] = xyz_transformed
            else:
                outputs['xyz_transformed'] = outputs['xyz']
                outputs['rot_maps_transformed'] = outputs['rot_maps']
                outputs['sh_maps_transformed'] = outputs['sh_maps']
        else:
            outputs['xyz_transformed'] = outputs['xyz']
            outputs['rot_maps_transformed'] = outputs['rot_maps']
            outputs['sh_maps_transformed'] = outputs['sh_maps']

        del bfc_K, bfc_c2e, bfc_depth_maps, bfc_xyz, rot_maps, scale_maps, opacity_maps, bfc_sh,
        return outputs

    def get_render_data(self, data_dict):
        inputs = data_dict['all_dict']
        outputs = {}
        b, s, c, h, w = inputs[('color_aug', 0)].shape

        outputs['input_all'] = inputs.copy()
        device = inputs[('color_aug', 0)].device
        outputs['input_all'][('ego_T_ego', 0, 0)] = torch.eye(4, device=device).unsqueeze(0).repeat(b, 1, 1)

        # Ensure timestamp is in input_all for timestamp-based interpolation
        if 'timestamp' in inputs:
            outputs['input_all']['timestamp'] = inputs['timestamp']

        # Copy any existing ego_T_ego transformations from inputs to input_all
        ego_keys_in_inputs = [k for k in inputs.keys() if isinstance(k, tuple) and len(k) == 3 and k[0] == 'ego_T_ego']
        for key in ego_keys_in_inputs:
            if key not in outputs['input_all']:
                ego_T_ego = inputs[key]
                # Extract camera 0's ego_T_ego if multi-camera format
                if ego_T_ego.dim() == 4:
                    ego_T_ego = ego_T_ego[:, 0]
                outputs['input_all'][key] = ego_T_ego

        # Compute missing ego transformations using camera transformation chain
        # Use all_dict which contains ego poses for all 7 frames (0-6)
        # Compute or load ego_T_ego transformations
        # Training path: use precomputed values from dataloader (computed in __getitem__)
        # Inference path: compute here (get_scene_sample doesn't precompute)
        for frame_id in range(1, self.context_span + 1):
            key = ('ego_T_ego', 0, frame_id)
            if key in outputs['input_all']:
                continue

            # Get precomputed ego_T_ego from dataloader (computed in __getitem__)
            if key in data_dict['all_dict']:
                # Dataset returns torch tensor [1, 4, 4] or [1, num_cameras, 4, 4]
                ego_T_ego = data_dict['all_dict'][key]
                # Extract camera 0's ego_T_ego if multi-camera format
                if ego_T_ego.dim() == 4:
                    ego_T_ego = ego_T_ego[:, 0]  # [batch_size, 4, 4]
                ego_T_ego = ego_T_ego.to(inputs['c2e_extr'].device)
                outputs['input_all'][key] = ego_T_ego
            else:
                raise KeyError(f"ego_T_ego key {key} not found in data_dict['all_dict']. "
                             f"This should be precomputed in the dataset.")

        for frame_id in self.all_render_frame_ids:
            for cam_id in range(self.num_cams):
                if self.render_cam_mode == 'origin':
                    # Load ground truth images and extrinsics from all_dict which contains all frames (0-6)
                    # all_dict layout: [frame0_cams(0-5), frame1_cams(6-11), ..., frame6_cams(36-41)]
                    all_dict_idx = frame_id * self.num_cams + cam_id
                    gt_img = data_dict['all_dict'][(f'color_aug', 0)][:, all_dict_idx, ...]

                    # CRITICAL: Camera is fixed on ego vehicle, so c2e_extr is the SAME for all frames!
                    # Always use frame 0's camera extrinsics (from context_frames)
                    # Since Gaussians are in ego_0 coordinates and camera is always in the same position
                    # relative to ego, we use frame 0's c2e_extr for all frames
                    c2e_extr = inputs['c2e_extr'][:, cam_id, ...]  # Frame 0's camera extrinsics
                    e2c_extr = torch.linalg.inv(c2e_extr)
                    K = inputs['K'][:, cam_id]
                    
                    # Extract gt_depth if available
                    if 'gt_depth' in inputs:
                        gt_depth = inputs['gt_depth'][:, frame_id*self.num_cams+cam_id, ...]
                        # Ensure gt_depth has shape [bs, c, h, w] - add channel dimension if missing
                        if gt_depth.dim() == 3:  # [bs, h, w] -> [bs, 1, h, w]
                            gt_depth = gt_depth.unsqueeze(1)
                        elif gt_depth.dim() == 4 and gt_depth.shape[-1] == 1:  # [bs, h, w, 1] -> [bs, 1, h, w]
                            gt_depth = gt_depth.squeeze(-1).unsqueeze(1)
                        outputs[('gt_depths', frame_id, cam_id)] = gt_depth
                
                elif self.render_cam_mode=='shift':
                    e2c_extr = torch.linalg.inv(inputs['c2e_extr'][:, frame_id*self.num_cams+cam_id, ...])
                    cam_T_cam = self.render_shift_T.to(e2c_extr.device).repeat(len(e2c_extr),1,1)
                    e2c_extr = torch.matmul(cam_T_cam, e2c_extr)
                    K = inputs['K'][:, frame_id*self.num_cams+cam_id]

                    gt_img = inputs[(f'color_aug', 0)][:, frame_id*self.num_cams+cam_id, ...]

                    outputs[('cam_T_cam',frame_id, cam_id)] = cam_T_cam
                    outputs[('gt_mask',frame_id, cam_id)] = inputs['mask'][:,frame_id*self.num_cams+cam_id]
                elif self.render_cam_mode=='scale':
                    if frame_id==0:
                        e2c_extr = torch.linalg.inv(inputs['c2e_extr'][:, frame_id*self.num_cams+cam_id, ...])
                    else:
                        cam_T_cam_key = ('cam_T_cam', 0, frame_id)
                        if cam_T_cam_key in inputs:
                            cam_T_cam = inputs[cam_T_cam_key][:, cam_id, ...]
                            e2c_extr = torch.matmul(cam_T_cam, torch.linalg.inv(inputs['c2e_extr'][:, cam_id, ...]))
                        else:
                            e2c_extr = torch.linalg.inv(inputs['c2e_extr'][:, frame_id*self.num_cams+cam_id, ...])
                    K = inputs['K'][:, frame_id*self.num_cams+cam_id].clone()
                    K[:,:2] = K[:,:2] * self.render_scale

                    gt_img = inputs[(f'color_aug', 0)][:, frame_id*self.num_cams+cam_id, ...]

                    gt_img = F.interpolate(gt_img, size=(self.render_height,self.render_width),mode = 'bilinear', align_corners=False)

                outputs[('groudtruth',frame_id, cam_id)] = gt_img
                outputs[('e2c_extr',frame_id, cam_id)] = e2c_extr
                # outputs[('c2e_extr',frame_id, cam_id)] = inputs['c2e_extr'][:, cam_id, ...]

                outputs[('K',frame_id, cam_id)] = K

        # Add inputs to outputs so render_splating_imgs can access cam_T_cam
        outputs['inputs'] = inputs
        return outputs

    def transform_points(self, points, transform_matrix):
        """
        Transform 3D points using a 4x4 transformation matrix
        Args:
            points: [..., N, 3] tensor of 3D points
            transform_matrix: [..., 4, 4] transformation matrix
        Returns:
            transformed_points: [..., N, 3] transformed 3D points
        """
        ones = torch.ones([*points.shape[:-1], 1], device=points.device)
        points_homo = torch.cat([points, ones], dim=-1)  # [..., N, 4]

        # Add dimension to transform_matrix for proper broadcasting with points
        # points_homo: [batch, N, 4] -> unsqueeze(-2) -> [batch, N, 1, 4]
        # transform_matrix: [batch, 4, 4] -> unsqueeze(-3) -> [batch, 1, 4, 4]
        # This allows broadcasting: [batch, N, 1, 4] @ [batch, 1, 4, 4] -> [batch, N, 1, 4]
        transformed = torch.matmul(points_homo.unsqueeze(-2), transform_matrix.unsqueeze(-3).transpose(-2, -1)).squeeze(-2)

        return transformed[..., :3]

    def skew_symmetric(self, w):
        """
        Convert 3D vector to skew-symmetric matrix.

        Args:
            w: [batch_size, 3] vectors

        Returns:
            w_hat: [batch_size, 3, 3] skew-symmetric matrices
        """
        bs = w.shape[0]
        device = w.device
        w_hat = torch.zeros(bs, 3, 3, device=device, dtype=w.dtype)
        w_hat[:, 0, 1] = -w[:, 2]
        w_hat[:, 0, 2] = w[:, 1]
        w_hat[:, 1, 0] = w[:, 2]
        w_hat[:, 1, 2] = -w[:, 0]
        w_hat[:, 2, 0] = -w[:, 1]
        w_hat[:, 2, 1] = w[:, 0]
        return w_hat
    
    def matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion (w, x, y, z)"""
        bs = R.shape[0]
        q = torch.zeros(bs, 4, device=R.device)
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        
        # Case 1: trace > 0
        mask1 = trace > 0
        if mask1.any():
            s = torch.sqrt(trace[mask1] + 1.0) * 2
            q[mask1, 0] = 0.25 * s
            q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
            q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
            q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s
        
        # Case 2: R[0,0] is largest
        mask2 = (~mask1) & ((R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2]))
        if mask2.any():
            s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
            q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
            q[mask2, 1] = 0.25 * s
            q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
            q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s
        
        # Case 3: R[1,1] is largest
        mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
        if mask3.any():
            s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
            q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
            q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
            q[mask3, 2] = 0.25 * s
            q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s
        
        # Case 4: R[2,2] is largest
        mask4 = (~mask1) & (~mask2) & (~mask3)
        if mask4.any():
            s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
            q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
            q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
            q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
            q[mask4, 3] = 0.25 * s
        
        return q / torch.norm(q, dim=1, keepdim=True)

    def render_splating_imgs(self, recontrast_data, render_data):
        bs = len(recontrast_data['xyz'])
        outputs = {}

        for frame_id in self.all_render_frame_ids:
            if 'xyz_transformed' in recontrast_data:
                xyz = recontrast_data['xyz_transformed']
            else:
                xyz = recontrast_data['xyz']
            flow = recontrast_data['forward_flow']
            mid_point = xyz.shape[1] // 2

            xyz_t = xyz.clone()
            if self.use_vehicle_flow:
                context_span_delta = self.context_span / 12.0
                delta_t_flow = (frame_id / self.context_span) * context_span_delta
                xyz_t[:, :mid_point] += flow[:, :mid_point] * delta_t_flow
                xyz_t[:, mid_point:] -= flow[:, mid_point:] * (context_span_delta - delta_t_flow)

            # Get input data and compute transformation for frame_id > 0
            # GT mode: Use cam_T_cam with e2c_0 -> cam_T_cam @ e2c_0
            # Interp mode: Use ego_T_ego with e2c_N -> e2c_N @ ego_T_ego
            input_all = render_data.get('input_all', {})
            if frame_id > 0:
                # Use per-frame GT cam_T_cam transformations: [batch, num_cameras, 4, 4]
                cam_T_cam_key = ('cam_T_cam', 0, frame_id)
                cam_T_cam_0to_delta = input_all[cam_T_cam_key]
            else:
                cam_T_cam_0to_delta = None

            # Project xyz to depth maps (xyz in ego_0 frame, pc2depth handles visibility)
            projected_depths = {}
            gt_depths = {}
            for cam_id in range(self.num_cams):
                K = render_data[('K', frame_id, cam_id)]  # [batch, 4, 4]

                if cam_T_cam_0to_delta is not None:
                    transform_cam = cam_T_cam_0to_delta[:, cam_id, :, :]  # [batch, 4, 4]

                    # GT mode: cam_T_cam @ inv(c2e_0)
                    # Note: frame 0 might not be in render_data during training (sampled frames)
                    c2e_extr_0 = input_all['c2e_extr'][:, cam_id, ...]  # [batch, 4, 4]
                    e2c_extr_0 = torch.linalg.inv(c2e_extr_0)
                    e2c_extr = torch.matmul(transform_cam, e2c_extr_0)
                else:
                    # Frame 0: no transformation, compute e2c from c2e in input_all
                    c2e_extr_0 = input_all['c2e_extr'][:, cam_id, ...]  # [batch, 4, 4]
                    e2c_extr = torch.linalg.inv(c2e_extr_0)

                projected_depth = pc2depth(
                    xyz, e2c_extr, K,
                    self.render_height, self.render_width
                )

                projected_depth = projected_depth.unsqueeze(1)
                projected_depths[('projected_depths', frame_id, cam_id)] = projected_depth

                if ('gt_depths', frame_id, cam_id) in render_data:
                    gt_depths[('gt_depths', frame_id, cam_id)] = render_data[('gt_depths', frame_id, cam_id)]

            # Store depths in outputs
            outputs.update(projected_depths)
            outputs.update(gt_depths)

            for i in range(bs):
                cam_point_num = self.render_height * self.render_width * self.num_cams
                xyz_i = xyz_t[i]
                if self.translate_3dgs:
                    rot_i = recontrast_data['rot_maps_transformed'][i]
                    sh_i = recontrast_data['sh_maps_transformed'][i]
                else:
                    rot_i = recontrast_data['rot_maps'][i]
                    sh_i = recontrast_data['sh_maps'][i]
                scale_i = recontrast_data['scale_maps'][i]
                opacity_i = recontrast_data['opacity_maps'][i]

                # Frame 0: render each camera separately with only its own 3DGS
                if frame_id == 0:
                    # Calculate points per camera (Gaussians are organized sequentially by camera)
                    points_per_cam = mid_point // self.num_cams

                    for cam_id in range(self.num_cams):
                        # Extract camera-specific Gaussians from both frames
                        # First frame indices
                        start_idx_frame0 = cam_id * points_per_cam
                        end_idx_frame0 = (cam_id + 1) * points_per_cam
                        # Second frame indices
                        start_idx_frame1 = mid_point + cam_id * points_per_cam
                        end_idx_frame1 = mid_point + (cam_id + 1) * points_per_cam

                        # Concatenate Gaussians from both frames for this camera
                        xyz_cam = torch.cat([xyz_i[start_idx_frame0:end_idx_frame0],
                                            xyz_i[start_idx_frame1:end_idx_frame1]], dim=0)
                        rot_cam = torch.cat([rot_i[start_idx_frame0:end_idx_frame0],
                                            rot_i[start_idx_frame1:end_idx_frame1]], dim=0)
                        sh_cam = torch.cat([sh_i[start_idx_frame0:end_idx_frame0],
                                           sh_i[start_idx_frame1:end_idx_frame1]], dim=0)
                        scale_cam = torch.cat([scale_i[start_idx_frame0:end_idx_frame0],
                                              scale_i[start_idx_frame1:end_idx_frame1]], dim=0)
                        opacity_cam = torch.cat([opacity_i[start_idx_frame0:end_idx_frame0],
                                                opacity_i[start_idx_frame1:end_idx_frame1]], dim=0)

                        # Get transformation for this camera
                        c2e_data = input_all['c2e_extr'][i, cam_id, ...]
                        e2c_data = torch.linalg.inv(c2e_data)
                        e2c_extr_cam = e2c_data.unsqueeze(0)  # [1, 4, 4]

                        k_data = render_data[('K', frame_id, cam_id)][i, :3, :3]
                        K_cam = k_data.unsqueeze(0)  # [1, 3, 3]

                        # Render this camera
                        render_colors_cam, render_alphas_cam, meta_cam = rasterization(
                            xyz_cam,  # [2*points_per_cam, 3] - both frames
                            rot_cam,  # [2*points_per_cam, 4]
                            scale_cam,  # [2*points_per_cam, 3]
                            opacity_cam.squeeze(-1),  # [2*points_per_cam]
                            sh_cam,  # [2*points_per_cam, K, 3]
                            e2c_extr_cam,  # [1, 4, 4]
                            K_cam,  # [1, 3, 3]
                            self.render_width,
                            self.render_height,
                            sh_degree=self.sh_degree,
                            render_mode="RGB",
                        )

                        render_rgb_cam = render_colors_cam[..., :3].permute(0, 3, 1, 2)  # [1, 3, H, W]

                        if ('gaussian_color', frame_id, cam_id) not in outputs:
                            outputs[('gaussian_color', frame_id, cam_id)] = []
                        outputs[('gaussian_color', frame_id, cam_id)].append(render_rgb_cam[0])

                        del xyz_cam, rot_cam, scale_cam, opacity_cam, sh_cam, e2c_extr_cam, K_cam
                        del render_colors_cam, render_alphas_cam, meta_cam, render_rgb_cam

                else:
                    # Other frames: render all cameras together with all Gaussians
                    e2c_extr_i, K_i = [], []
                    for cam_id in range(self.num_cams):
                        if cam_T_cam_0to_delta is not None:
                            transform_cam = cam_T_cam_0to_delta[i, cam_id, :, :]  # [4, 4]

                            # GT mode: cam_T_cam @ inv(c2e_0)
                            # Note: frame 0 might not be in render_data during training (sampled frames)
                            c2e_data = input_all['c2e_extr'][i, cam_id, ...]
                            e2c_data = torch.linalg.inv(c2e_data)
                            e2c_data = torch.matmul(transform_cam, e2c_data)
                        else:
                            # Frame 0: no transformation, compute e2c from c2e in input_all
                            c2e_data = input_all['c2e_extr'][i, cam_id, ...]
                            e2c_data = torch.linalg.inv(c2e_data)

                        e2c_extr_i.append(e2c_data)

                        k_data = render_data[('K',frame_id, cam_id)][i,:3,:3]
                        K_i.append(k_data)

                    e2c_extr_i = torch.stack(e2c_extr_i, dim=0)
                    K_i = torch.stack(K_i, dim=0)

                    render_colors_i, render_alphas_i, meta_i = rasterization(
                        xyz_i,  # [N, 3]
                        rot_i,  # [N, 4]
                        scale_i,  # [N, 3]
                        opacity_i.squeeze(-1),  # [N]
                        sh_i,  # [N, K, 3]
                        e2c_extr_i,  # [6, 4, 4]
                        K_i,  # [6, 3, 3]
                        self.render_width,
                        self.render_height,
                        sh_degree=self.sh_degree,
                        render_mode="RGB",
                        # sparse_grad=True,
                        # this is to speedup large-scale rendering by skipping far-away Gaussians.
                        # radius_clip=3,
                    )
                    # render_rgb_i, render_depth_i = render_colors_i[...,:3], render_colors_i[...,3]
                    render_rgb_i = render_colors_i[...,:3].permute(0,3,1,2)
                    del xyz_i, rot_i, scale_i, opacity_i, sh_i, e2c_extr_i, K_i, render_colors_i, render_alphas_i, meta_i

                    for cam_id in range(self.num_cams):
                        if ('gaussian_color', frame_id, cam_id) not in outputs:
                            outputs[('gaussian_color', frame_id, cam_id)] = []
                        outputs[('gaussian_color', frame_id, cam_id)].append(render_rgb_i[cam_id])

            for cam_id in range(self.num_cams):
                gaussian_color = torch.stack(outputs[('gaussian_color', frame_id,cam_id)],dim=0).contiguous()
                outputs[('groudtruth', frame_id, cam_id)] = render_data[('groudtruth', frame_id, cam_id)]

                if self.render_cam_mode=='shift':
                    ref_mask = render_data[('gt_mask',frame_id,cam_id)]
                    ref_K = render_data[('K',frame_id, cam_id)]
                    ref_depths = rearrange(recontrast_data['pred_depth'],'b (c h w) -> b c h w',c=self.num_cams*len(self.all_render_frame_ids),h=self.height,w=self.width)[:,frame_id*self.num_cams+cam_id:frame_id*self.num_cams+cam_id+1, ...]
                    cam_T_cam = render_data[('cam_T_cam',frame_id, cam_id)]
                    ref_inv_K = torch.linalg.inv(ref_K)
                    gaussian_color, mask_warped = self.get_virtual_image(
                        gaussian_color, 
                        ref_mask, 
                        ref_depths, 
                        ref_inv_K, 
                        ref_K, 
                        cam_T_cam
                    )
                else:
                    mask_warped = torch.ones_like(gaussian_color[:,0:1,...])

                outputs[('gaussian_color', frame_id, cam_id)] = gaussian_color
                outputs[('warped_mask', frame_id, cam_id)] = mask_warped.detach()

        return outputs
    

    def render_project_imgs(self, data_dict, recontrast_data):
        all_dict = data_dict['all_dict']
        context_data = data_dict['context_frames']
        outputs = {}
        b, context_s, c, h, w = context_data[('color_aug', 0)].shape

        ref_frame_id = 0  # Always frame 0

        # Determine source frames based on self.all_render_frame_ids
        # Get frames from all_render_frame_ids that are in range [1, 2, 3, 4, 5]
        candidate_src_frames = [f for f in self.all_render_frame_ids if 1 <= f <= 5]

        # If no frames in [1, 2, 3, 4, 5], default to frame 1
        if len(candidate_src_frames) == 0:
            src_frame_ids = [1]
        else:
            src_frame_ids = candidate_src_frames

        # Get reference (frame 0) data once for all cameras
        for cam_id in range(self.num_cams):
            ref_all_dict_idx = ref_frame_id * self.num_cams + cam_id
            ref_colors = all_dict[('color_aug', 0)][:, ref_all_dict_idx, ...]
            bs, _, height, width = ref_colors.shape
            ref_depths = rearrange(recontrast_data['pred_depths'],'b (c h w) -> b c h w',c=context_s,h=height,w=width)[:, ref_frame_id*self.num_cams+cam_id, ...]
            ref_depths = ref_depths.unsqueeze(1)
            if 'mask' in all_dict:
                ref_mask = all_dict['mask'][:, ref_all_dict_idx]
            else:
                ref_mask = torch.ones_like(ref_depths)
            ref_K = all_dict['K'][:, ref_all_dict_idx, ]
            ref_inv_K = torch.linalg.inv(ref_K)
            outputs[('ref_colors', ref_frame_id, cam_id)] = ref_colors
            outputs[('ref_depths', ref_frame_id, cam_id)] = ref_depths

            # Process each source frame
            for src_frame_id in src_frame_ids:
                src_all_dict_idx = src_frame_id * self.num_cams + cam_id
                src_colors = all_dict[('color_aug', 0)][:, src_all_dict_idx, ...]

                cam_T_cam = all_dict[('cam_T_cam', 0, src_frame_id)][:, cam_id, ...]

                warped_img, warped_mask = self.get_virtual_image(
                                src_colors,
                                ref_mask,
                                ref_depths,
                                ref_inv_K,
                                ref_K,
                                cam_T_cam
                            )
                warped_img = self.get_norm_image_single(
                    ref_colors,
                    ref_mask,
                    warped_img,
                    warped_mask
                )

                outputs[('warped_gt', ref_frame_id, src_frame_id, cam_id)] = ref_colors
                outputs[('warped_pred', ref_frame_id, src_frame_id, cam_id)] = warped_img
                outputs[('warped_mask', ref_frame_id, src_frame_id, cam_id)] = warped_mask.detach()

                # Store source colors for visualization (if needed)
                outputs[('src_colors', src_frame_id, cam_id)] = src_colors
                outputs[('ref_depths', ref_frame_id, cam_id)] = ref_depths
        return outputs

    def get_virtual_image(self, src_img, src_mask, tar_depth, tar_invK, src_K, T):
        """
        This function warps source image to target image using backprojection and reprojection process. 
        """
        # do reconstruction for target from source   
        pix_coords = self.project(tar_depth, T, tar_invK, src_K)
        
        img_warped = F.grid_sample(src_img, pix_coords, mode='bilinear', 
                                    padding_mode='zeros', align_corners=True)
        mask_warped = F.grid_sample(src_mask, pix_coords, mode='nearest', 
                                    padding_mode='zeros', align_corners=True)

        # nan handling
        inf_img_regions = torch.isnan(img_warped)
        img_warped[inf_img_regions] = 2.0
        inf_mask_regions = torch.isnan(mask_warped)
        mask_warped[inf_mask_regions] = 0

        pix_coords = pix_coords.permute(0, 3, 1, 2)
        invalid_mask = torch.logical_or(pix_coords > 1, 
                                        pix_coords < -1).sum(dim=1, keepdim=True) > 0
        return img_warped, (~invalid_mask).float() * mask_warped
    
    def get_norm_image_single(self, src_img, src_mask, warp_img, warp_mask):
        """
        obtain normalized warped images using the mean and the variance from the overlapped regions of the target frame.
        """
        warp_mask = warp_mask.detach()

        with torch.no_grad():
            mask = (src_mask * warp_mask).bool()
            if mask.size(1) != 3:
                mask = mask.repeat(1,3,1,1)

            mask_sum = mask.sum(dim=(-3,-2,-1))
            # skip when there is no overlap
            if torch.any(mask_sum == 0):
                return warp_img

            s_mean, s_std = self.get_mean_std(src_img, mask)
            w_mean, w_std = self.get_mean_std(warp_img, mask)

        norm_warp = (warp_img - w_mean) / (w_std + 1e-8) * s_std + s_mean
        return norm_warp * warp_mask.float()   

    def get_mean_std(self, feature, mask):
        """
        This function returns mean and standard deviation of the overlapped features. 
        """
        _, c, h, w = mask.size()
        mean = (feature * mask).sum(dim=(1,2,3), keepdim=True) / (mask.sum(dim=(1,2,3), keepdim=True) + 1e-8)
        var = ((feature - mean) ** 2).sum(dim=(1,2,3), keepdim=True) / (c*h*w)
        return mean, torch.sqrt(var + 1e-16)     
    
    def _unproject_depth_map_to_points_map(self, depth_map, K, c2e_extr):
        '''
        depth_map: depth map of shape (Bs, H, W)
        K: pixel -> camera intrinsic matrix of shape (Bs, 4, 4)  
        c2e_extr: camera -> ego extrinsics matrix of shape (Bs, 4, 4)  
        return points_map: points map of shape (Bs, H, W, 3)
        '''
        if depth_map is None:
            return None

        Bs, H, W = depth_map.shape
        
        
        u = torch.arange(W, device=depth_map.device, dtype=torch.float32)
        v = torch.arange(H, device=depth_map.device, dtype=torch.float32)
        u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')  # u_grid: (H, W), v_grid: (H, W)
        
        
        u_grid = u_grid.unsqueeze(0).expand(Bs, -1, -1)  # (Bs, H, W)
        v_grid = v_grid.unsqueeze(0).expand(Bs, -1, -1)  # (Bs, H, W)
        
        
        fx = K[:, 0, 0]  # (Bs,)
        fy = K[:, 1, 1]  # (Bs,)
        cx = K[:, 0, 2]  # (Bs,)
        cy = K[:, 1, 2]  # (Bs,)
        
        
        fx = fx.view(Bs, 1, 1).expand(-1, H, W)
        fy = fy.view(Bs, 1, 1).expand(-1, H, W)
        cx = cx.view(Bs, 1, 1).expand(-1, H, W)
        cy = cy.view(Bs, 1, 1).expand(-1, H, W)
        
        
        x = (u_grid - cx) * depth_map / fx
        y = (v_grid - cy) * depth_map / fy
        z = depth_map
        
        
        cam_points = torch.stack([x, y, z], dim=-1)  # (Bs, H, W, 3)

        
        R = c2e_extr[:, :3, :3]  # (Bs, 3, 3)
        t = c2e_extr[:, :3, 3]   # (Bs, 3)
        
        
        cam_points_flat = cam_points.reshape(Bs, -1, 3)  # (Bs, H*W, 3)
        
        
        ego_points_flat = torch.matmul(cam_points_flat, R.transpose(1, 2)) + t.unsqueeze(1)
        
        
        ego_points = ego_points_flat.reshape(Bs, H, W, 3)  # (Bs, H, W, 3)

        
        mask = (depth_map == 0).unsqueeze(-1).expand(-1, -1, -1, 3)  # (Bs, H, W, 3)
        ego_points[mask] = 0
        del ego_points_flat, cam_points_flat, cam_points
        return ego_points

    @rank_zero_only
    def _save_wordpoints_glb(self, glbfile, inputs, recontrast_data, render_data, bs_id=0):
        predictions = {'images':[],'world_points_from_depth':[],'extrinsic':[]}
        for frame_id in self.all_render_frame_ids:
            for cam_id in range(self.num_cams):
                predictions['images'].append(render_data[('groudtruth', frame_id, cam_id)][bs_id].cpu().numpy())

        predictions['images'] = np.stack(predictions['images'], axis=0)
        predictions['world_points_from_depth'] = recontrast_data['xyz'][bs_id].detach().cpu().numpy()
        predictions['extrinsic'] = inputs['c2e_extr'][bs_id].cpu().numpy()
        os.makedirs(os.path.dirname(glbfile),exist_ok=True)
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=0.0,
            filter_by_frames='all',
            mask_black_bg=False,
            mask_white_bg=False,
            show_cam=True,
            mask_sky=False,
            target_dir=None,
            prediction_mode='',
        )
        del predictions
        print(f'save to {glbfile}')
        glbscene.export(file_obj=glbfile)
        torch.cuda.empty_cache()
    def _filter_visible_gaussians(self,  pts_xyz, full_proj_transform, opacity):
        
        
        points_homogeneous = torch.cat([
            pts_xyz, 
            torch.ones(pts_xyz.shape[0], 1, device=pts_xyz.device)
        ], dim=1)

        clip_points = torch.mm(full_proj_transform, points_homogeneous.t()).t()

        ndc_points = clip_points[:, :3] / clip_points[:, 3:4]

        
        in_frustum = (
            (ndc_points[:, 0] >= -1) & (ndc_points[:, 0] <= 1) &
            (ndc_points[:, 1] >= -1) & (ndc_points[:, 1] <= 1) &
            (ndc_points[:, 2] >= -1) & (ndc_points[:, 2] <= 1)
        )

        
        opaque = opacity.squeeze(-1) > 0.01  

        valid_points = in_frustum & opaque
        del in_frustum, opaque
        return valid_points

    def compute_gaussian_loss(self, batch_data):
        """
        This function computes gaussian loss.
        """
        # self occlusion mask * overlap region mask

        gaussian_loss = 0.0 
        count = 0
        for frame_id in self.all_render_frame_ids:
            for cam_id in range(self.num_cams): 
                pred = batch_data[('gaussian_color', frame_id, cam_id)]
                gt = batch_data[('groudtruth', frame_id, cam_id)]  
                mask = batch_data[('warped_mask', frame_id, cam_id)]

                lpips_loss = self.lpips(pred, gt, normalize=True)
                # lpips_loss = 0.0
                l2_loss = ((pred - gt)**2)
                sum_loss = 1 * l2_loss + 0.05 * lpips_loss
                gaussian_loss += compute_masked_loss(sum_loss, mask, eps=0.1)
                count += 1
        return self.lambda_gaussian * gaussian_loss / count

    def compute_project_loss(self, batch_data):
        """
        This function computes projection loss for warping from source frames to reference frame.
        Keys are now in format: ('warped_pred', ref_frame_id, src_frame_id, cam_id)
        """
        project_loss = 0.0
        count = 0

        ref_frame_id = 0  # Always use frame 0 as reference

        # Iterate over all warped_pred keys to find all (src_frame_id, cam_id) combinations
        for key in batch_data.keys():
            if key[0] == 'warped_pred' and key[1] == ref_frame_id:
                # Key format: ('warped_pred', ref_frame_id, src_frame_id, cam_id)
                src_frame_id = key[2]
                cam_id = key[3]

                pred = batch_data[('warped_pred', ref_frame_id, src_frame_id, cam_id)]
                gt = batch_data[('warped_gt', ref_frame_id, src_frame_id, cam_id)]
                mask = batch_data[('warped_mask', ref_frame_id, src_frame_id, cam_id)]

                # img_loss = compute_photometric_loss(pred, gt)
                l1_loss = self.l1_fn(pred, gt)
                ssim_loss = self.ssim_fn(pred, gt)
                sum_loss = 0.85 * l1_loss + 0.15 * ssim_loss
                project_loss += compute_masked_loss(sum_loss, mask, eps=0.1)
                count += 1

        return self.lambda_project * project_loss / count

    def compute_depth_loss(self, batch_recontrast_data, beta=1.0, eps=1e-6):
        """
        This function computes edge-aware smoothness loss for the disparity map.
        """
        depth_loss = 0.0 
        count = 0

        for frame_id in self.all_render_frame_ids:
            for cam_id in range(self.num_cams):
                gt_depth = batch_recontrast_data[('gt_depths', frame_id, cam_id)]
                pred_depth = batch_recontrast_data[('projected_depths', frame_id, cam_id)]
                gaussian_color = batch_recontrast_data[('gaussian_color', frame_id, cam_id)]

                mask_depth = torch.logical_and(gt_depth > self.min_depth, gt_depth < self.max_depth)
                mask_depth = mask_depth.to(torch.float32)

                abs_diff = torch.abs(gt_depth - pred_depth) * mask_depth
                l1loss = torch.where(abs_diff < beta, 0.5 * abs_diff * abs_diff / beta, abs_diff - 0.5 * beta)
                l1loss = torch.sum(l1loss) / (torch.sum(mask_depth) + eps)
                depth_loss += l1loss * self.lambda_depth
                
                mean_disp = pred_depth.mean(2, True).mean(3, True)
                norm_disp = pred_depth / (mean_disp + 1e-8)
                edge_loss = compute_edg_smooth_loss(gaussian_color, norm_disp)
                depth_loss += self.lambda_edge * edge_loss

                # Save intermediate visualization images
                # self.save_depth_visualization(gt_depth, pred_depth, gaussian_color, frame_id, cam_id)

                count += 1

        return   depth_loss / count

    def compute_norm_loss(self, batch_data):

        scale_loss = self.lambda_scale * torch.mean(torch.norm(batch_data['scale_maps'], dim=-1))
        
        
        opacity_loss = self.lambda_opacity * torch.mean(torch.abs(batch_data['opacity_maps']))
        
        
        total_reg_loss = scale_loss + opacity_loss

        return total_reg_loss

    @torch.no_grad()
    def compute_reconstruction_metrics(self, batch_data, stage):
        """
        This function computes reconstruction metrics.
        """
        psnr = 0.0
        ssim = 0.0
        lpips = 0.0

        novel_count =0
        ## Haibao: self.all_render_frame_ids?? [0, 1, 2, 3, 4, 5, 6]
        for frame_id in self.all_render_frame_ids:
            for cam_id in range(self.num_cams): 
                pred = batch_data[('gaussian_color', frame_id, cam_id)].detach()
                gt = batch_data[('groudtruth', frame_id, cam_id)]    
                psnr += self.compute_psnr(gt, pred).mean()
                ssim += self.compute_ssim(gt, pred).mean()
                lpips += self.compute_lpips(gt, pred).mean()
                novel_count += 1

        psnr /= novel_count
        ssim /= novel_count
        lpips /= novel_count

        self.log(f"{stage}/psnr", psnr.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/ssim", ssim.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/lpips", lpips.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return psnr, ssim, lpips
    

    
    @torch.no_grad()
    def compute_psnr(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, " batch"]:
        ground_truth = ground_truth.clip(min=0, max=1)
        predicted = predicted.clip(min=0, max=1)
        mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
        return -10 * mse.log10()
    
    @torch.no_grad()
    def compute_lpips(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, " batch"]:
        value = self.lpips.forward(ground_truth, predicted, normalize=True)
        return value[:, 0, 0, 0]
    
    @torch.no_grad()
    def compute_ssim(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, " batch"]:
        ssim = [
            structural_similarity(
                gt.detach().cpu().numpy(),
                hat.detach().cpu().numpy(),
                win_size=11,
                gaussian_weights=True,
                channel_axis=0,
                data_range=1.0,
            )
            for gt, hat in zip(ground_truth, predicted)
        ]
        return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)
