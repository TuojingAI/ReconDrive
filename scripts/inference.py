#!/usr/bin/env python3
#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#

"""
Scene-based inference script for ReconDrive
This script demonstrates inference using scene-by-scene iteration
"""

import yaml
import argparse
import os
import sys
import torch
import json
from pathlib import Path
import time
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn.functional as F
from gsplat.rendering import rasterization
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "models"))  # Add models directory for vggt imports

from dataset.vggt3dgs_scene_data_module import VGGT3DGS_SceneDataModule
from dataset.vggt4dgs_scene_dataset import custom_collate_fn
from models.recondrive_model import ReconDrive_LITModelModule


class SceneSampleDataset(Dataset):
    """Dataset wrapper that supports both pre-loaded samples and lazy loading"""
    
    def __init__(self, samples_or_indices, dataset=None, scene_idx=None):
        if dataset is not None and scene_idx is not None:
            # Lazy loading mode: samples_or_indices is list of sample indices
            self.lazy_mode = True
            self.sample_indices = samples_or_indices
            self.dataset = dataset
            self.scene_idx = scene_idx
        else:
            # Pre-loaded mode: samples_or_indices is list of actual samples
            self.lazy_mode = False
            self.samples = samples_or_indices
    
    def __len__(self):
        if self.lazy_mode:
            return len(self.sample_indices) 
        else:
            return len(self.samples)
    
    def __getitem__(self, idx):
        if self.lazy_mode:
            # Load sample on-demand
            sample_idx = self.sample_indices[idx] 
            return self.dataset.__getitem__(sample_idx,self.scene_idx)
        else:
            return self.samples[idx]



def load_model_from_checkpoint(checkpoint_path, model_cfg, device):
    """Load model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Ensure batch_size is in model config
    if 'batch_size' not in model_cfg:
        model_cfg['batch_size'] = 1  # Set default batch_size for inference
    
    # Initialize model
    model = ReconDrive_LITModelModule(
        cfg=model_cfg,
        save_dir='./temp_log',
        logger=None
    )

    model.load_pretrained_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    return model


def run_inference(model_cfg=None, model=None, checkpoint_path=None,
                  scene_dataloader=None, device='cuda:0',
                  save_results=True, output_dir=None, novel_distances=[1.0, 2.0],
                  eval_resolution='280x518'):
    """
    Scene-based inference function for single GPU

    Args:
        model_cfg: Model configuration (when model=None)
        model: Pre-loaded model (optional)
        checkpoint_path: Path to model checkpoint
        scene_dataloader: Scene data loader
        device: Device string (e.g., 'cuda:0')
        save_results: Whether to save results
        output_dir: Output directory
        novel_distances: List of distances for novel view generation
        eval_resolution: Resolution mode - 'original' or 'upsampled'
    """
    print(f"\nStarting single-GPU scene-based inference on {device}")
    print(f"Number of scenes to process: {len(scene_dataloader)}")

    # Load model if not provided
    if model is None:
        if model_cfg is None or checkpoint_path is None:
            raise ValueError("model_cfg and checkpoint_path required when model is None")
        model = load_model_from_checkpoint(checkpoint_path, model_cfg, device)

    return _run_single_gpu_inference(model, scene_dataloader, device, save_results, output_dir, novel_distances, eval_resolution)


def save_rendered_image(tensor_img, save_path, upsample_to=None):
    """Save a tensor image to file with optional upsampling"""
    if tensor_img.dim() == 4:
        tensor_img = tensor_img.squeeze(0)

    if upsample_to is not None:
        target_height, target_width = upsample_to
        device = tensor_img.device
        if tensor_img.device.type == 'cpu':
            tensor_img = tensor_img.cuda()
        
        tensor_img = tensor_img.unsqueeze(0)
        tensor_img = F.interpolate(tensor_img, size=(target_height, target_width), 
                                 mode='bilinear', align_corners=False)
        tensor_img = tensor_img.squeeze(0)
        
        if device.type == 'cpu':
            tensor_img = tensor_img.cpu()
    
    img_np = tensor_img.detach().cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    Image.fromarray(img_np).save(save_path)


def create_lateral_translation_matrices(translation_distances=[1.0, 2.0]):
    """Create transformation matrices for lateral (left/right) ego vehicle translation"""
    transforms = {}
    
    for dist in translation_distances:
        left_transform = torch.eye(4)
        left_transform[1, 3] = dist  # Negative Y for left
        transforms[f'left_{dist}m'] = left_transform
        
        right_transform = torch.eye(4)
        right_transform[1, 3] = -dist   # Positive Y for right
        transforms[f'right_{dist}m'] = right_transform
    
    return transforms


def render_novel_views(model, recontrast_data, render_data, device, scene_name, sample_idx, save_dir, actual_sample_idx, translation_distances=[1.0, 2.0], eval_resolution='280x518', novel_render_frames=[]):
    """Render novel views with lateral ego translation"""
    # Get transformation matrices for translation
    translation_transforms = create_lateral_translation_matrices(translation_distances)
    
    saved_paths = []
    
    xyz_i = recontrast_data['xyz'][sample_idx:sample_idx+1]  # Keep batch dimension
    rot_i = recontrast_data['rot_maps'][sample_idx:sample_idx+1]
    scale_i = recontrast_data['scale_maps'][sample_idx:sample_idx+1]
    opacity_i = recontrast_data['opacity_maps'][sample_idx:sample_idx+1]
    sh_i = recontrast_data['sh_maps'][sample_idx:sample_idx+1]
    
    # Get camera parameters
    frame_id = 0  # Use current frame

    num_cams = getattr(model, 'num_cams', 6)
    model_width = getattr(model, 'width', 518)
    model_height = getattr(model, 'height', 280)
    for transform_name, transform_matrix in translation_transforms.items():
        transform_matrix = transform_matrix.to(device)
        for frame_id in novel_render_frames:
            for cam_id in range(num_cams):
                # Get original camera extrinsics and intrinsics
                original_e2c_extr = render_data[('e2c_extr', frame_id, cam_id)][sample_idx:sample_idx+1]
                K_i = render_data[('K', frame_id, cam_id)][sample_idx:sample_idx+1, :3, :3]
                
                # Apply lateral translation to camera pose
                # Transform ego to camera: new_e2c = e2c @ inv(transform)
                novel_e2c_extr = torch.matmul(original_e2c_extr, torch.linalg.inv(transform_matrix.unsqueeze(0)))
                
                # Render with new camera pose
                render_colors_i, render_alphas_i, meta_i = rasterization(
                    xyz_i.squeeze(0),      # [N, 3]
                    rot_i.squeeze(0),      # [N, 4]
                    scale_i.squeeze(0),    # [N, 3]
                    opacity_i.squeeze(0).squeeze(-1),  # [N]
                    sh_i.squeeze(0),       # [N, K, 3]
                    novel_e2c_extr,        # [1, 4, 4]
                    K_i,                   # [1, 3, 3]
                    model_width,
                    model_height,
                    sh_degree=getattr(model, 'sh_degree', 3),
                    render_mode="RGB",
                )
                
                # Extract RGB and convert to proper format
                render_rgb = render_colors_i[..., :3].permute(0, 3, 1, 2)[0]  # [C, H, W]
                
                # Save the novel view
                global_sample_idx =  actual_sample_idx + frame_id
                save_path = os.path.join(save_dir, scene_name, 
                                        f'sample_{global_sample_idx:04d}', transform_name, f'{eval_resolution}_cam_{cam_id}.png')
            
                resize_height, resize_width = eval_resolution.split('x')
                resize_height = int(resize_height)
                resize_width = int(resize_width)
                if resize_height !=model_height or resize_width != model_width:
                    save_rendered_image(render_rgb, save_path, upsample_to=(resize_height, resize_width))
                else:  # eval_resolution == 'original'
                    save_rendered_image(render_rgb, save_path)
        
    return saved_paths

def to_device(data, device):
    if isinstance(data, dict):
        return {k: (v if k == 'vehicle_annotations' else to_device(v, device)) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(x, device) for x in data)
    elif torch.is_tensor(data):
        return data.to(device)
    else:
        return data

def _extract_first_value(value, default=None):
    """Extract first value from list/tensor or return as-is"""
    if isinstance(value, list):
        return value[0] if value else default
    elif isinstance(value, torch.Tensor):
        if value.numel() > 0:
            return value[0].item() if value.dim() > 0 else value.item()
        return default
    return value

def _extract_scene_idx(scene_batch, default_idx=0):
    """Extract scene_idx from batch, checking multiple locations"""
    for key in ['target_frames', 'context_frames']:
        if key in scene_batch and 'scene_idx' in scene_batch[key]:
            return _extract_first_value(scene_batch[key]['scene_idx'], default_idx)
    return default_idx

def _process_scene_batch(model, scene_batch, device, gpu_id=0, save_renders=True, output_dir=None, novel_distances=[1.0, 2.0], eval_resolution='280x518', batch_idx=0):
    """Process a single scene batch and return results"""
    scene_start_time = time.time()

    scene_name = scene_batch['scene_name']
    scene_token = scene_batch['scene_token']
    
    # Initialize original_sample_indices to track the actual frame indices
    original_sample_indices = None
    frame_skip = 6

    # Support both lazy loading and pre-loaded modes
    if 'samples' in scene_batch:
        scene_samples = scene_batch['samples']
        scene_length = len(scene_samples)
        
        if frame_skip is not None and frame_skip > 1:
            # Calculate which indices to keep: 0, frame_skip, 2*frame_skip, ...
            original_sample_indices = list(range(0, scene_length, frame_skip))
            filtered_samples = [scene_samples[i] for i in original_sample_indices]
            scene_dataset = SceneSampleDataset(filtered_samples)
            actual_samples = len(filtered_samples)
            
            print(f"GPU {gpu_id}: Using frame skip of {frame_skip}, processing {actual_samples} out of {scene_length} samples")
        else:
            scene_dataset = SceneSampleDataset(scene_samples)
            actual_samples = scene_length
            original_sample_indices = list(range(scene_length))
    else:
        scene_length = scene_batch['scene_length']
        all_indices = scene_batch['sample_indices']
        
        if frame_skip is not None and frame_skip > 1:
            # Calculate which indices to keep: 0, frame_skip, 2*frame_skip, ...
            positions_to_keep = list(range(0, len(all_indices), frame_skip))
            original_sample_indices = [all_indices[pos] for pos in positions_to_keep]
            
            print(f"GPU {gpu_id}: Using frame skip of {frame_skip}, processing {len(original_sample_indices)} out of {scene_length} samples")
            
            scene_dataset = SceneSampleDataset(
                original_sample_indices,
                dataset=scene_batch['dataset'], 
                scene_idx=scene_batch['scene_idx']
            )
            actual_samples = len(original_sample_indices)
        else:
            original_sample_indices = all_indices
            scene_dataset = SceneSampleDataset(
                all_indices,
                dataset=scene_batch['dataset'], 
                scene_idx=scene_batch['scene_idx']
            )
            actual_samples = len(all_indices)
    
    print(f"GPU {gpu_id}: Processing Scene: {scene_name} ({actual_samples} samples)")
    
    # Create DataLoader
    scene_loader = DataLoader(
        scene_dataset, batch_size=1, shuffle=False,
        pin_memory=False, num_workers=0, drop_last=False,
        collate_fn=custom_collate_fn
    )

    scene_psnr_list, scene_ssim_list, scene_lpips_list = [], [], []
    batch_count = 0

    for batch_data in scene_loader:
        batch_count += 1
        
        # Get the actual sample index from the original data
        if original_sample_indices is not None:
            if batch_count - 1 < len(original_sample_indices):
                actual_sample_idx = original_sample_indices[batch_count - 1]
            else:
                actual_sample_idx = batch_count - 1
        else:
            actual_sample_idx = batch_count - 1

        num_cams = getattr(model, 'num_cams', 6)

        batch_data = to_device(batch_data, device)

        # Run prediction
        output = model.predict_step(batch_data, batch_idx)

        model_width = getattr(model, 'width', 518)
        model_height = getattr(model, 'height', 280)


        if isinstance(output, tuple):
            batch_recontrast_data, batch_render_data, batch_splating_data = output
            
            # Calculate metrics for this batch
            batch_psnr, batch_ssim, batch_lpips = [], [], []

            # Separate metrics for reconstruction and novel view modes
            recon_psnr, recon_ssim, recon_lpips = [], [], []
            novel_psnr, novel_ssim, novel_lpips = [], [], []

            # Use the batch index as the global sample index


            # Determine which frames are available from the returned data
            available_frames = set()
            for key in batch_splating_data.keys():
                if isinstance(key, tuple) and key[0] == 'gaussian_color':
                    available_frames.add(key[1])  # frame_id is at index 1
            available_frames = sorted(list(available_frames))

            print(f"GPU {gpu_id}: Available frames for evaluation: {available_frames}")

            # === Mode 1: Scene Reconstruction (frame 0) ===
            frame_id = 0
            if frame_id in [0]:
                for cam_id in range(num_cams):
                    pred_key = ('gaussian_color', frame_id, cam_id)
                    gt_key = ('groudtruth', frame_id, cam_id)
                    if pred_key in batch_splating_data and gt_key in batch_splating_data:
                        pred = batch_splating_data[pred_key][0:1]
                        gt = batch_splating_data[gt_key][0:1]

                        resize_height, resize_width = eval_resolution.split('x')
                        resize_height = int(resize_height)
                        resize_width = int(resize_width)
                        if (resize_height == model_height) and (resize_width == model_width):
                            pred_eval = pred.clamp(0, 1)
                            gt_eval = gt.clamp(0, 1)
                        else:
                            if ('color_org', frame_id) in scene_batch:
                                gt_original = scene_batch[('color_org', frame_id)][:, cam_id, ...][0:1]
                                gt_original = gt_original.clamp(0, 1).to(pred.device)
                                gt_eval = F.interpolate(gt_original, size=(resize_height, resize_width), mode='bilinear', align_corners=False)
                            else:
                                gt_eval = F.interpolate(gt, size=(resize_height, resize_width), mode='bilinear', align_corners=False)
                                gt_eval = gt_eval.clamp(0, 1)
                            pred_eval = F.interpolate(pred, size=(resize_height, resize_width), mode='bilinear', align_corners=False)
                            pred_eval = pred_eval.clamp(0, 1)

                        psnr_val = model.compute_psnr(gt_eval, pred_eval).mean().item()
                        ssim_val = model.compute_ssim(gt_eval, pred_eval).mean().item()
                        lpips_val = model.compute_lpips(gt_eval, pred_eval).mean().item()

                        recon_psnr.append(psnr_val)
                        recon_ssim.append(ssim_val)
                        recon_lpips.append(lpips_val)

                        if output_dir:
                            global_sample_idx = actual_sample_idx + frame_id
                            frame_dir = 'gt_views'
                            pred_save_path = os.path.join(output_dir, scene_name,
                                                        f'sample_{global_sample_idx:04d}', frame_dir,  f'{eval_resolution}_cam_{cam_id}_pred.png')
                            gt_save_path = os.path.join(output_dir, scene_name,
                                                        f'sample_{global_sample_idx:04d}', frame_dir,  f'{eval_resolution}_cam_{cam_id}_gt.png')
                            os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)
                            save_rendered_image(pred_eval.squeeze(0), pred_save_path)
                            save_rendered_image(gt_eval.squeeze(0), gt_save_path)

            # === Mode 2: Novel View Synthesis (middle frames) ===
            novel_frames = [f for f in available_frames if f != 0]
            for frame_id in novel_frames:
                for cam_id in range(num_cams):
                    pred_key = ('gaussian_color', frame_id, cam_id)
                    gt_key = ('groudtruth', frame_id, cam_id)
                    if pred_key in batch_splating_data and gt_key in batch_splating_data:
                        pred = batch_splating_data[pred_key][0:1]
                        gt = batch_splating_data[gt_key][0:1]

                        resize_height, resize_width = eval_resolution.split('x')
                        resize_height = int(resize_height)
                        resize_width = int(resize_width)
                        if (resize_height == model_height) and (resize_width == model_width):
                            # Original mode: Use original model resolution (280x518)
                            if frame_id == 0 and cam_id == 0:
                                print(f"GPU {gpu_id}: Original mode - Using model resolution: pred={pred.shape}, gt={gt.shape}")
                                print(f"GPU {gpu_id}: Original mode - pred range: [{pred.min():.3f}, {pred.max():.3f}], gt range: [{gt.min():.3f}, {gt.max():.3f}]")
                            
                            # Ensure both pred and gt are in [0,1] range
                            pred_eval = pred.clamp(0, 1)
                            gt_eval = gt.clamp(0, 1)
                        else:
                            if ('color_org', frame_id) in scene_batch:
                                # Use original high-resolution GT and upsample to 900x1600
                                gt_original = scene_batch[('color_org', frame_id)][:, cam_id, ...][0:1]
                                if frame_id == 0 and cam_id == 0:
                                    print(f"GPU {gpu_id}: Upsampled mode - Original GT shape: {gt_original.shape}")
                                    print(f"GPU {gpu_id}: Upsampled mode - Original GT range: [{gt_original.min():.3f}, {gt_original.max():.3f}]")
                                
                                # Ensure GT is in [0,1] range and on correct device
                                gt_original = gt_original.clamp(0, 1).to(pred.device)
                                gt_eval = F.interpolate(gt_original, size=(resize_height, resize_width), mode='bilinear', align_corners=False)
                            else:
                                # Fallback: use downsampled GT if original not available
                                if frame_id == 0 and cam_id == 0:
                                    print(f"GPU {gpu_id}: Upsampled mode - Warning: Using downsampled GT: {gt.shape}")
                                gt_eval = F.interpolate(gt, size=(resize_height, resize_width), mode='bilinear', align_corners=False)
                                gt_eval = gt_eval.clamp(0, 1)
                            
                            # Upsample predicted image to 900x1600 and ensure in [0,1] range
                            pred_eval = F.interpolate(pred, size=(resize_height, resize_width), mode='bilinear', align_corners=False)
                            pred_eval = pred_eval.clamp(0, 1)

                        # Calculate novel view metrics
                        psnr_val = model.compute_psnr(gt_eval, pred_eval).mean().item()
                        ssim_val = model.compute_ssim(gt_eval, pred_eval).mean().item()
                        lpips_val = model.compute_lpips(gt_eval, pred_eval).mean().item()

                        novel_psnr.append(psnr_val)
                        novel_ssim.append(ssim_val)
                        novel_lpips.append(lpips_val)

                        # Save novel view images
                        if output_dir:
                            global_sample_idx = actual_sample_idx + frame_id
                            frame_dir = f'gt_views'
                            pred_save_path = os.path.join(output_dir, scene_name,
                                                        f'sample_{global_sample_idx:04d}', frame_dir,  f'{eval_resolution}_cam_{cam_id}_pred.png')
                            gt_save_path = os.path.join(output_dir, scene_name,
                                                        f'sample_{global_sample_idx:04d}', frame_dir,  f'{eval_resolution}_cam_{cam_id}_gt.png')
                            os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)
                            save_rendered_image(pred_eval.squeeze(0), pred_save_path)
                            save_rendered_image(gt_eval.squeeze(0), gt_save_path)

            # Print separate metrics for both modes
            if recon_psnr:
                print(f"GPU {gpu_id}: Recon metrics - PSNR: {np.mean(recon_psnr):.3f}, SSIM: {np.mean(recon_ssim):.3f}, LPIPS: {np.mean(recon_lpips):.3f}")
            if novel_psnr:
                print(f"GPU {gpu_id}: Novel metrics - PSNR: {np.mean(novel_psnr):.3f}, SSIM: {np.mean(novel_ssim):.3f}, LPIPS: {np.mean(novel_lpips):.3f}")

            # Generate and save novel views for all samples
            if save_renders and output_dir:
                # Define which frames to render novel views for (use available frames by default)
                novel_render_frames = [0,1,2,3,4,5]
                novel_view_paths = render_novel_views(
                    model, batch_recontrast_data, batch_render_data,
                    device, scene_name, 0, output_dir, actual_sample_idx, novel_distances, eval_resolution, novel_render_frames
                )
                print(f"GPU {gpu_id}: Saved novel views for sample {actual_sample_idx}: {len(novel_view_paths)} images")

            # Aggregate scene-level metrics (keep modes separate)
            scene_psnr_list.extend(recon_psnr + novel_psnr)
            scene_ssim_list.extend(recon_ssim + novel_ssim)
            scene_lpips_list.extend(recon_lpips + novel_lpips)
            

    # Calculate processing time
    scene_processing_time = time.time() - scene_start_time

    # Return scene results with separate reconstruction and novel view metrics
    return {
        'scene_idx': scene_batch.get('scene_idx', 0),
        'scene_name': scene_name,
        'scene_token': scene_token,
        'processed_samples': len(scene_psnr_list),
        'processing_time': scene_processing_time,
        'avg_sample_time': scene_processing_time / max(1, len(scene_psnr_list)),
        'gpu_id': gpu_id,
        'metrics': {
            'psnr': np.mean(scene_psnr_list) if scene_psnr_list else 0.0,
            'ssim': np.mean(scene_ssim_list) if scene_ssim_list else 0.0,
            'lpips': np.mean(scene_lpips_list) if scene_lpips_list else 0.0,
            'psnr_std': np.std(scene_psnr_list) if scene_psnr_list else 0.0,
            'ssim_std': np.std(scene_ssim_list) if scene_ssim_list else 0.0,
            'lpips_std': np.std(scene_lpips_list) if scene_lpips_list else 0.0
        },
        'sample_metrics': {
            'psnr_list': scene_psnr_list,
            'ssim_list': scene_ssim_list,
            'lpips_list': scene_lpips_list
        },
        'recon_metrics': {
            'psnr_list': recon_psnr,
            'ssim_list': recon_ssim,
            'lpips_list': recon_lpips
        },
        'novel_metrics': {
            'psnr_list': novel_psnr,
            'ssim_list': novel_ssim,
            'lpips_list': novel_lpips
        }
    }


def _run_single_gpu_inference(model, scene_dataloader, device, save_results=True, output_dir=None, novel_distances=[1.0, 2.0], eval_resolution='280x518'):
    """Run inference on all scenes - simplified using unified scene processing"""
    print(f"\nStarting scene-based inference on device: {device}")
    print(f"Number of scenes: {len(scene_dataloader)}")
    all_scene_results = []
    overall_psnr, overall_ssim, overall_lpips = [], [], []
    recon_psnr, recon_ssim, recon_lpips = [], [], []
    novel_psnr, novel_ssim, novel_lpips = [], [], []

    with torch.no_grad():
        for scene_idx, scene_batch in enumerate(scene_dataloader):

            # Get scene_idx from the batch efficiently
            scene_batch['scene_idx'] = scene_idx

            result = _process_scene_batch(model, scene_batch, device, gpu_id=0,
                                        save_renders=save_results, output_dir=output_dir,
                                        novel_distances=novel_distances, eval_resolution=eval_resolution,
                                        batch_idx=scene_idx)

            all_scene_results.append(result)
            overall_psnr.extend(result['sample_metrics']['psnr_list'])
            overall_ssim.extend(result['sample_metrics']['ssim_list'])
            overall_lpips.extend(result['sample_metrics']['lpips_list'])

            # Collect separate reconstruction and novel view metrics
            recon_psnr.extend(result['recon_metrics']['psnr_list'])
            recon_ssim.extend(result['recon_metrics']['ssim_list'])
            recon_lpips.extend(result['recon_metrics']['lpips_list'])
            novel_psnr.extend(result['novel_metrics']['psnr_list'])
            novel_ssim.extend(result['novel_metrics']['ssim_list'])
            novel_lpips.extend(result['novel_metrics']['lpips_list'])

    # Print final results - separate for reconstruction and novel view
    final_psnr = np.mean(overall_psnr) if overall_psnr else 0.0
    final_ssim = np.mean(overall_ssim) if overall_ssim else 0.0
    final_lpips = np.mean(overall_lpips) if overall_lpips else 0.0

    final_recon_psnr = np.mean(recon_psnr) if recon_psnr else 0.0
    final_recon_ssim = np.mean(recon_ssim) if recon_ssim else 0.0
    final_recon_lpips = np.mean(recon_lpips) if recon_lpips else 0.0

    final_novel_psnr = np.mean(novel_psnr) if novel_psnr else 0.0
    final_novel_ssim = np.mean(novel_ssim) if novel_ssim else 0.0
    final_novel_lpips = np.mean(novel_lpips) if novel_lpips else 0.0

    print(f"\n{'='*60}")
    print(f"SINGLE-GPU INFERENCE COMPLETED")
    print(f"{'='*60}")
    print(f"Scenes: {len(all_scene_results)}, Samples: {sum(r['processed_samples'] for r in all_scene_results)}")
    print(f"Overall PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}, LPIPS: {final_lpips:.4f}")
    print(f"\nScene Reconstruction (Frame 0):")
    print(f"  PSNR: {final_recon_psnr:.4f}, SSIM: {final_recon_ssim:.4f}, LPIPS: {final_recon_lpips:.4f}")
    print(f"\nNovel View Synthesis (Middle Frames):")
    print(f"  PSNR: {final_novel_psnr:.4f}, SSIM: {final_novel_ssim:.4f}, LPIPS: {final_novel_lpips:.4f}")
    
    # Save results
    # if save_results and output_dir:
    if output_dir:
        final_results = {
            'overall_metrics': {'psnr': final_psnr, 'ssim': final_ssim, 'lpips': final_lpips,
                               'psnr_std': np.std(overall_psnr), 'ssim_std': np.std(overall_ssim), 'lpips_std': np.std(overall_lpips)},
            'scene_results': all_scene_results
        }
        save_inference_results(final_results, output_dir)

    return all_scene_results


def save_inference_results(results, output_dir):
    """Save inference results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    if 'overall_metrics' in results:
        # New format with overall metrics
        scene_results = results['scene_results']
        overall_metrics = results['overall_metrics']
        
        # Create summary
        summary = {
            'overall_metrics': overall_metrics,
            'total_scenes': len(scene_results),
            'total_samples': sum(r['processed_samples'] for r in scene_results),
            'total_time': sum(r['processing_time'] for r in scene_results),
            'scenes': [
                {
                    'scene_idx': r['scene_idx'],
                    'scene_name': r['scene_name'],
                    'scene_token': r['scene_token'],
                    'processed_samples': r['processed_samples'],
                    'processing_time': r['processing_time'],
                    'metrics': r['metrics']
                }
                for r in scene_results
            ]
        }
        
        # Save summary
        summary_file = os.path.join(output_dir, 'inference_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        detailed_file = os.path.join(output_dir, 'inference_detailed.json')
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save per-scene evaluation results
        for scene_result in scene_results:
            scene_name = scene_result['scene_name']
            scene_eval_file = os.path.join(output_dir, f'scene_{scene_name}_evaluation.json')
            scene_eval_data = {
                'scene_name': scene_name,
                'scene_token': scene_result['scene_token'],
                'metrics': scene_result['metrics'],
                'sample_metrics': scene_result['sample_metrics'],
                'processing_info': {
                    'processed_samples': scene_result['processed_samples'],
                    'processing_time': scene_result['processing_time'],
                    'avg_sample_time': scene_result['avg_sample_time']
                }
            }
            
            with open(scene_eval_file, 'w') as f:
                json.dump(scene_eval_data, f, indent=2)
        
        print(f"\nResults saved:")
        print(f"  Summary: {summary_file}")
        print(f"  Detailed: {detailed_file}")
        print(f"  Per-scene evaluations: {output_dir}/scene_*_evaluation.json")
        
    else:
        # Legacy format
        summary = {
            'total_scenes': len(results),
            'total_samples': sum(r['processed_samples'] for r in results),
            'total_time': sum(r['processing_time'] for r in results),
            'scenes': [
                {
                    'scene_idx': r['scene_idx'],
                    'scene_name': r['scene_name'],
                    'scene_token': r['scene_token'],
                    'processed_samples': r['processed_samples'],
                    'processing_time': r['processing_time'],
                    'scene_stats': r.get('scene_stats', {})
                }
                for r in results
            ]
        }
        
        # Save summary
        summary_file = os.path.join(output_dir, 'inference_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        detailed_file = os.path.join(output_dir, 'inference_detailed.json')
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved:")
        print(f"  Summary: {summary_file}")
        print(f"  Detailed: {detailed_file}")


def main():
    parser = argparse.ArgumentParser(description='Scene-based inference for VGGT3DGS')
    parser.add_argument('--cfg_path', type=str, required=True, help='Configuration file path')
    parser.add_argument('--restore_ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--max_scenes', type=int, default=None, help='Maximum number of scenes to process (default: all scenes)')
    parser.add_argument('--device', type=str, default=None, help='Device to use (e.g., cuda:0)')

    parser.add_argument('--no_renders', action='store_true', help='Disable saving rendered images and novel views')
    parser.add_argument('--novel_distances', type=str, default='0.5,1.0,2.0,3.0', 
                       help='Novel view translation distances in meters (comma-separated, e.g., "0.5,1.0,2.0,3.0")')
    parser.add_argument('--eval_resolution', type=str, default='original',# choices=['original', 'upsampled'],
                       help='Evaluation resolution mode: "original" for 280x518, "upsampled" for 900x1600')

    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.cfg_path}")
    with open(args.cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Set batch_size to 1 for inference (CRITICAL: must be 1 for proper scene processing)
    config['model_cfg']['batch_size'] = 1
    config['data_cfg']['batch_size'] = 1
    
    # Pass temporal config from data_cfg to model_cfg for time_delta calculation
    if 'context_span' in config['data_cfg']:
        config['model_cfg']['context_span'] = config['data_cfg']['context_span']
    if 'nuscenes_version' in config['data_cfg']:
        config['model_cfg']['nuscenes_version'] = config['data_cfg']['nuscenes_version']


    # Parse device
    if args.device:
        # Single GPU specified: "cuda:0" or "0"
        if args.device.startswith('cuda:'):
            device = args.device
        else:
            device = f"cuda:{args.device}"
    elif config.get('devices'):
        device = f"cuda:{config['devices'][0]}"
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {device}")
    print(f"Batch size: {config['data_cfg']['batch_size']}")
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(config['save_dir'], 'scene_inference_results')
    
    print(f"Output directory: {args.output_dir}")
    
    # Parse save renders flag
    save_renders = not args.no_renders
    
    # Parse novel view distances
    try:
        novel_distances = [float(d.strip()) for d in args.novel_distances.split(',')]
    except ValueError:
        raise ValueError(f"Invalid novel_distances format: {args.novel_distances}. Use comma-separated floats like '0.5,1.0,2.0,3.0'")
    
    print(f"Save renders: {save_renders}")
    print(f"Novel view distances: {novel_distances}")
    print(f"Evaluation resolution: {args.eval_resolution}")

    # CRITICAL: Ensure batch_size is 1 before creating data module
    print(f"Original batch_size in config: {config['data_cfg'].get('batch_size', 'not set')}")
    config['data_cfg']['batch_size'] = 1  # Must be 1 for proper scene processing
    print(f"Override batch_size to: {config['data_cfg']['batch_size']}")

    # Initialize scene-based data module
    print("Initializing scene-based data module...")
    data_module = VGGT3DGS_SceneDataModule(cfg=config['data_cfg'])
    data_module.setup(stage='test')
    
    # Get scene dataloader
    scene_dataloader = data_module.test_scene_dataloader()
    total_scenes = len(scene_dataloader)
    
    if args.max_scenes:
        print(f"Limiting to {args.max_scenes} scenes (out of {total_scenes})")
        # Limit scenes if requested
        scene_list = []
        for i, scene_batch in enumerate(scene_dataloader):
            if i >= args.max_scenes:
                break
            scene_list.append(scene_batch)
        scene_dataloader = scene_list
    else:
        print(f"Processing all {total_scenes} scenes")


    # Run single-GPU inference
    results = run_inference(
        model_cfg=config['model_cfg'],
        checkpoint_path=args.restore_ckpt,
        scene_dataloader=scene_dataloader,
        device=device,
        save_results=save_renders,
        output_dir=args.output_dir,
        novel_distances=novel_distances,
        eval_resolution=args.eval_resolution,
    )
    
    print(f"\nScene-based inference completed successfully!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()