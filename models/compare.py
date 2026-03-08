#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#

from collections import defaultdict
import PIL.Image as Image
import torch
import gc
import torch.nn.functional as F
import torch.optim as optim
import os
import shutil
import numpy as np
import pytorch_lightning as pl
from einops import rearrange, reduce
from torch import Tensor
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

from models.recondrive_model import ReconDriveModel

from models.gaussian_util import render, focal2fov, getProjectionMatrix,  depth2pc, pc2depth, rotate_sh

from models.loss_util import compute_photometric_loss, compute_masked_loss, compute_edg_smooth_loss
from utils.visual_util import predictions_to_glb
from models.geometry_util import Projection

def print_memory(msg):
    # torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[MEM] {msg}: Alloc={allocated:.2f}GB, Reserved={reserved:.2f}GB")

class ReconDrive_LITModelModule(pl.LightningModule):
    def __init__(self, cfg, save_dir='.', logger=None):
        super().__init__()
        self.read_config(cfg)

        # Set default values for ego transformation configuration if not in config
        if not hasattr(self, 'use_gt_ego_trans'):
            self.use_gt_ego_trans = False

        self.save_dir = save_dir
        self.model = ReconDriveModel(sh_degree=self.sh_degree,min_depth=self.min_depth, max_depth=self.max_depth) 
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
        
        # Image saving configuration
        self.save_training_images = False # True # False  # Set to True to enable saving
        self.saved_steps_count = 0
        self.max_save_steps = 7
        self.save_images_dir = os.path.join('work_dirs/vggt4dgs_1030/debug_vggt_scene', 'no_velocity_flow')
        self.camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                            'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
        
        # Performance optimization flags
        self.compute_alternative_flow = cfg.get('model_cfg', {}).get('compute_alternative_flow', False)  # Mode 2 flow for comparison
    
    def load_pretrained_checkpoint(self, checkpoint_path, strict=False, verbose=True):
   
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if strict:
            
            self.load_state_dict(checkpoint['state_dict'])
            if verbose:
                print(f"strict model checkpoint: {checkpoint_path}")
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
        # self.render_cam_mode = 'scale'
        # self.render_width = 640
        # self.render_scale = self.render_width * 1.0 / self.width
        # self.render_height = int(self.height * self.render_scale)


        # self.render_cam_mode = 'shift'
        # self.render_shift_T = torch.tensor([
        #         [1, 0, 0, -2],
        #         [0, 1, 0, 0],
        #         [0, 0, 1, 0],
        #         [0, 0, 0, 1]
        #     ],dtype=torch.float32).unsqueeze(0)
        

    def save_depth_visualization(self, gt_depth, pred_depth, gaussian_color, frame_id, cam_id):
        """Save intermediate visualization images for depth comparison"""
        import torchvision.utils as vutils
        import torch.nn.functional as F
        
        # Only save occasionally to avoid too many files
        if not hasattr(self, '_depth_save_counter'):
            self._depth_save_counter = 0
        
        # Save every 10 steps during training or always during validation/test
        if self.stage in ['val', 'test'] or self._depth_save_counter % 10 == 0:
            # Create directory
            depth_vis_dir = os.path.join(self.save_dir, 'depth_visualization')
            os.makedirs(depth_vis_dir, exist_ok=True)
            
            # Get camera name
            cam_name = self.camera_names[cam_id] if cam_id < len(self.camera_names) else f'CAM_{cam_id}'
            
            # Process tensors (take first batch item and first channel)
            gt_depth_vis = gt_depth[0, 0].detach().cpu()  # [H, W]
            pred_depth_vis = pred_depth[0, 0].detach().cpu()  # [H, W]
            gaussian_color_vis = gaussian_color[0].detach().cpu()  # [3, H, W]
            
            # Normalize depth maps for visualization
            gt_depth_norm = (gt_depth_vis - gt_depth_vis.min()) / (gt_depth_vis.max() - gt_depth_vis.min() + 1e-8)
            pred_depth_norm = (pred_depth_vis - pred_depth_vis.min()) / (pred_depth_vis.max() - pred_depth_vis.min() + 1e-8)
            
            # Create colormaps for depth visualization
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            import numpy as np
            
            # Apply colormap
            gt_depth_colored = torch.from_numpy(cm.viridis(gt_depth_norm.numpy())[:,:,:3]).permute(2,0,1).float()
            pred_depth_colored = torch.from_numpy(cm.viridis(pred_depth_norm.numpy())[:,:,:3]).permute(2,0,1).float()
            
            # Save individual images
            step_counter = getattr(self, 'global_step', 0)
            
            # GT depth
            gt_path = os.path.join(depth_vis_dir, f'step_{step_counter}_frame_{frame_id}_{cam_name}_gt_depth.png')
            vutils.save_image(gt_depth_colored, gt_path)
            
            # Predicted depth  
            pred_path = os.path.join(depth_vis_dir, f'step_{step_counter}_frame_{frame_id}_{cam_name}_pred_depth.png')
            vutils.save_image(pred_depth_colored, pred_path)
            
            # Gaussian color
            color_path = os.path.join(depth_vis_dir, f'step_{step_counter}_frame_{frame_id}_{cam_name}_gaussian_color.png')
            vutils.save_image(gaussian_color_vis, color_path)
            
        self._depth_save_counter += 1

    def aug_novel_view_mode(self,):
        # self.render_frame_ids = np.random.choice(self.frame_ids,size=size).tolist()
        novel_prob1 = np.random.rand()
        if novel_prob1<0.7:
            self.render_frame_ids = [0]
            self.render_cam_mode = np.random.choice(['origin','scale','shift'])
            
        elif novel_prob1<0.85:
            self.render_frame_ids = [1]
            self.render_cam_mode = 'origin'
        else:
            self.render_frame_ids = [-1]
            self.render_cam_mode = 'origin'
        
        # self.render_cam_mode = 'origin'

        if self.render_cam_mode=='origin':
            self.render_width = self.width
            self.render_height = self.height
            self.render_scale = 1.0
            self.render_shift_x = 0.0
            self.render_shift_y = 0.0
            self.render_shift_T = torch.eye(4,dtype=torch.float32).unsqueeze(0)
        elif self.render_cam_mode=='scale':
            self.render_width = np.random.randint(self.render_width_min,self.render_width_max)
            self.render_scale = self.render_width * 1.0 / self.width
            self.render_height = int(self.height * self.render_scale)
            self.render_shift_x = 0.0
            self.render_shift_y = 0.0
            self.render_shift_T = torch.eye(4,dtype=torch.float32).unsqueeze(0)
        elif self.render_cam_mode=='shift':
            self.render_width = self.width
            self.render_height = self.height
            self.render_scale = 1.0
            self.render_shift_x = np.random.randn()*2
            self.render_shift_y = np.random.randn()*0.5
            self.render_shift_T = torch.eye(4,dtype=torch.float32).unsqueeze(0)
            self.render_shift_T[:,0,3] = self.render_shift_x
            self.render_shift_T[:,1,3] = self.render_shift_y

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
        # Frame IDs: [0, 1, 2, 3, 4, 5, 6]
        # Probabilities: [0.7, 0.4, 0.3, 0.2, 0.1, 0.05, 0]
        # prob_all_render_frame_ids = [0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0]
        # prob_all_render_frame_ids = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
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

        # Add rank info to debug message for multi-GPU
        if hasattr(self, 'global_rank') and self.global_rank is not None:
            print(f"[GPU {self.global_rank}] Sampling rendered ids: {selected_ids}")
        else:
            print(f"Sampling rendered ids: {selected_ids}")

        self.all_render_frame_ids = selected_ids
        self.context_span = 6
         

    def training_step(self, batch_input, batch_idx):
        self.stage = stage = 'train'

        # self.set_normal_params(batch_input)
        self.prob_sample_rendered_ids()
        # self.aug_novel_view_mode()  
        self._log_weights_and_grads(batch_input)

        batch_recontrast_data = self.get_recontrast_data(batch_input)

        loss_norm = self.compute_norm_loss(batch_recontrast_data)

        batch_render_data = self.get_render_data(batch_input)
        batch_splating_data  = self.render_splating_imgs(batch_recontrast_data,batch_render_data)

        loss_gaussian = self.compute_gaussian_loss(batch_splating_data)
        loss_depth = torch.tensor(0.0, device=self.device)

        batch_render_project_data = self.render_project_imgs(batch_input, batch_recontrast_data)
        loss_project = self.compute_project_loss(batch_render_project_data)


        self.log(f'{stage}/gs', loss_gaussian.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log(f'{stage}/depth', loss_depth.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}/proj', loss_project.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/norm', loss_norm.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        # Exclude projection loss from total loss
        loss_all = loss_gaussian + loss_depth + loss_project + loss_norm
        psnr, ssim, lpips = self.compute_reconstruction_metrics(batch_splating_data,stage)
        
        # Save training images if enabled
        if self.save_training_images:
            self.save_training_step_images(batch_idx, batch_splating_data)

        if self.save_image_duration > 0 and batch_idx%self.save_image_duration==0:
            # Save warping visualization images
            warping_file = f'{stage}_{batch_idx}_warping'
            self._save_warping_images(warping_file, batch_render_project_data)

            splating_file = f'{stage}_{batch_idx}_splating'
            self._save_splating_images(splating_file,batch_splating_data)

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

        if self.save_training_images and hasattr(self, 'save_images_dir'):
            self.save_training_step_images(batch_idx, batch_splating_data_recon, batch_recontrast_data)

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

        # Save training images if enabled
        if self.save_training_images:
            self.save_training_step_images(batch_idx, batch_splating_data, batch_recontrast_data)

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
                    print(f"NaN : {name}")
                    valid_gradients = False
                if torch.isinf(param.grad).any():
                    print(f"Inf : {name}")
                    valid_gradients = False
        
        if not valid_gradients:
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
                    torch.use_deterministic_algorithms(True)

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
                    torch.use_deterministic_algorithms(True)
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
    
    def refine_velocities_with_transformation(self, original_velocities, original_T, refined_T):
        """
        Refine velocities based on the refined ego transformation
        Args:
            original_velocities: list of 3D velocities computed with original transformation
            original_T: original ego_T_ego transformation (1, 4, 4)
            refined_T: refined ego_T_ego transformation from ICP (1, 4, 4)
        Returns:
            refined_velocities: list of refined 3D velocities
        """
        time_delta = self.time_delta
        if torch.is_tensor(original_T):
            original_T = original_T[0].cpu().numpy()
            refined_T = refined_T[0].cpu().numpy()
        else:
            original_T = np.array(original_T[0])
            refined_T = np.array(refined_T[0])
            
        correction_T = np.linalg.inv(original_T) @ refined_T
        rotation_correction = correction_T[:3, :3]
        translation_diff = refined_T[:3, 3] - original_T[:3, 3]
        
        refined_velocities = []
        for velocity in original_velocities:
            vel_array = np.array(velocity[:3], dtype=np.float32)
            displacement = vel_array * time_delta
            displacement_corrected = rotation_correction @ displacement + translation_diff
            refined_vel = displacement_corrected / time_delta
            refined_velocities.append(refined_vel.tolist())
        
        return refined_velocities
    
    def get_recontrast_data(self, data_dict):
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
        # Transform each camera's rotations using its c2e rotation matrix
        # bfc_rot_maps_ego = self.transform_gaussian_rotations(bfc_rot_maps, bfc_c2e)

        outputs['pred_depths'] = rearrange(bfc_depth_maps, '(b c) h w -> b (c h w)', b=batch_size, c=frame_camrea)#.contiguous()

        outputs['xyz'] = rearrange(bfc_xyz, '(b c) p k -> b (c p) k', b=batch_size, c=frame_camrea)#.contiguous()
        outputs['rot_maps'] = rearrange(bfc_rot_maps, '(b c) p d -> b (c p) d', b=batch_size, c=frame_camrea, d=4)#.contiguous()

        outputs['scale_maps'] = rearrange(scale_maps, 'b c h w d -> b (c h w) d', d=3)#.contiguous()
        outputs['opacity_maps'] = rearrange(opacity_maps, 'b c h w d -> b (c h w) d')#.contiguous()
        outputs['sh_maps'] = rearrange(bfc_sh, '(b c) h w p d -> b (c h w) d p', b=batch_size, c=frame_camrea)#.contiguous()
        
        # Perform ICP refinement early if we have multiple frames (needed for ego pose and velocity refinement)
        ego_T_ego_key = ('ego_T_ego', 0, self.context_span)
        if frame_camrea > self.num_cams and ego_T_ego_key in inputs:
            # Get ego_T_ego transformations
            ego_T_ego_0toN_initial = inputs[ego_T_ego_key]

            # Enable ICP refinement (can be controlled by config)
                # Extract frame 0 and frame N points for ICP
                mid_point = outputs['xyz'].shape[1] // 2
                xyz_frame0_all = outputs['xyz'][:, :mid_point, :]
                xyz_frameN_all = outputs['xyz'][:, mid_point:, :]

                points_per_camera = xyz_frame0_all.shape[1] // self.num_cams
                subsample_rate = getattr(self, 'icp_subsample_rate', 10)

                # Refine transformation for each camera
                refined_ego_T_ego_list = []

                for cam_id in range(self.num_cams):
                    # Extract this camera's points
                    cam_start = cam_id * points_per_camera
                    cam_end = (cam_id + 1) * points_per_camera
                    xyz_frame0_cam = xyz_frame0_all[:, cam_start:cam_end, :]
                    xyz_frameN_cam = xyz_frameN_all[:, cam_start:cam_end, :]

                    # Extract second half (bottom half of image) for ICP
                    # Points are in row-major order: [batch, H*W, 3]
                    # Second half starts at H*W//2
                    half_point = points_per_camera // 2
                    xyz_frame0_cam = xyz_frame0_cam[:, half_point:, :]
                    xyz_frameN_cam = xyz_frameN_cam[:, half_point:, :]

                    # Subsample points for efficiency
                    xyz_frame0_sub = xyz_frame0_cam[:, ::subsample_rate, :]
                    xyz_frameN_sub = xyz_frameN_cam[:, ::subsample_rate, :]

                    # Get camera-specific transformation
                    if ego_T_ego_0toN_initial.dim() == 4:
                        # [batch_size, num_cameras, 4, 4] - Use camera-specific transformation
                        ego_T_ego_0toN_cam = ego_T_ego_0toN_initial[:, cam_id]
                    else:
                        # [batch_size, 4, 4] - Use unified transformation for all cameras
                        ego_T_ego_0toN_cam = ego_T_ego_0toN_initial

                    # Refine transformation using ICP
                    # Following reference implementation convention:
                    # - Use frame0 as target, frameN as source
                    # - Requires transformation: egoN → ego0, so we invert ego0 → egoN
                    ego_T_ego_Nto0_cam = torch.linalg.inv(ego_T_ego_0toN_cam)

                    refined_ego_T_ego_Nto0 = self.refine_ego_transformation_icp(
                        xyz_frame0_sub,           # Target: frame0 points (in ego0 coordinates)
                        xyz_frameN_sub,           # Source: frameN points (in egoN coordinates)
                        ego_T_ego_Nto0_cam,       # Initial transformation: egoN → ego0
                        max_iterations=getattr(self, 'icp_max_iterations', 20),
                        tolerance=getattr(self, 'icp_tolerance', 1e-6),
                        cam_id=cam_id             # Camera ID for debug logging
                    )

                    # Convert back to ego0 → egoN for storage
                    refined_ego_T_ego_cam = torch.linalg.inv(refined_ego_T_ego_Nto0)
                    refined_ego_T_ego_list.append(refined_ego_T_ego_cam)

                # Stack refined transformations: [batch_size, num_cameras, 4, 4]
                refined_ego_T_ego_all = torch.stack(refined_ego_T_ego_list, dim=1)

                # Store both original and refined transformations
                outputs['ego_T_ego_original'] = ego_T_ego_0toN_initial
                outputs['ego_T_ego_refined'] = refined_ego_T_ego_all
            else:
                # Store the transformation being used (no refinement)
                outputs['ego_T_ego_original'] = ego_T_ego_0toN_initial
                # outputs['ego_T_ego_refined'] = ego_T_ego_0toN_initial

        # Generate vehicle-based 3D velocity flow
        if self.use_vehicle_flow:
            new_forward_flow = []

            # ALWAYS compute vehicle masks using SAM2 for correct flow application
            # The masks are essential for applying velocity to the correct pixels
            compute_vehicle_masks = True  # Always compute for proper flow
            
            # Separately control whether to save masks for visualization
            save_masks_for_viz = (
                self.stage in ['test', 'predict'] or  # Save during test/predict
                (hasattr(self, 'save_image_duration') and self.save_image_duration > 0)  # Or when saving is enabled
            )

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

                                # # Use refined or raw velocities
                                #     original_T = outputs['ego_T_ego_original'][b]
                                #     refined_T = outputs['ego_T_ego_refined'][b]
                                #     vehicle_velocities = self.refine_velocities_with_transformation(
                                #         raw_velocities, original_T, refined_T
                                #     )
                                # else:
                                #     vehicle_velocities = raw_velocities
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
                    
                    # Store individual masks separately if needed for visualization
                    if save_masks_for_viz:
                        # Store reference to individual masks for visualization
                        # This will be stored in a separate field
                        pass  # We'll handle this separately

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

        if frame_camrea > self.num_cams:
            num_frames = frame_camrea // self.num_cams
            if num_frames != 2:
                raise NotImplementedError(f"Context frames should have exactly 2 frames (frame 0 and frame {self.context_span}), but got {num_frames} frames")

            xyz_transformed = outputs['xyz'].clone()
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
        else:
            outputs['xyz_transformed'] = outputs['xyz']

        del bfc_K, bfc_c2e, bfc_depth_maps, bfc_xyz, rot_maps, scale_maps, opacity_maps, bfc_sh,
        return outputs

    def compute_target_frame_depths(self, data_dict, context_depth_maps, context_outputs):
        """
        Compute depth maps for target frames (intermediate frames between key frames)
        by warping/interpolating from context frame depths.

        Args:
            data_dict: Dictionary containing target_frames data
            context_depth_maps: Depth maps from context frames [b, (context_cams * h * w)]
            context_outputs: Outputs from get_recontrast_data for context frames

        Returns:
            target_depth_maps: Depth maps for target frames [b, (target_cams * h * w)]
            target_xyz: 3D points for target frames [b, (target_cams * h * w), 3]
        """
        if 'target_frames' not in data_dict or len(data_dict['target_frames']) == 0:
            return None, None

        target_inputs = data_dict['target_frames']

        # Get dimensions from context frames
        batch_size = context_depth_maps.shape[0]
        num_target_cams = target_inputs[('color_aug', 0)].shape[1] if ('color_aug', 0) in target_inputs else 0

        if num_target_cams == 0:
            return None, None

        height = self.height
        width = self.width

        # Prepare to collect target frame depths
        target_depth_list = []
        target_xyz_list = []

        # Get number of key frames and cameras
        total_key_cams = context_depth_maps.shape[1] // (height * width)
        num_key_frames = total_key_cams // self.num_cams

        # Reshape context depths for easier access
        context_depths_reshaped = rearrange(
            context_depth_maps,
            'b (c h w) -> b c h w',
            c=total_key_cams,
            h=height,
            w=width
        )

        # Process each target frame camera
        for target_cam_id in range(num_target_cams):
            # Get target frame data
            target_K = target_inputs['K'][:, target_cam_id, ...]
            target_c2e_extr = target_inputs['c2e_extr'][:, target_cam_id, ...]

            # Determine camera index and frame position
            cam_idx = target_cam_id % self.num_cams
            frame_position = target_cam_id // self.num_cams  # Position in the sequence

            if num_key_frames >= 2:
                # Get depths from first and last key frame for this camera
                frame0_depth_idx = cam_idx
                frame1_depth_idx = (num_key_frames - 1) * self.num_cams + cam_idx

                depth_frame0 = context_depths_reshaped[:, frame0_depth_idx, :, :]
                depth_frame1 = context_depths_reshaped[:, frame1_depth_idx, :, :]

                # Interpolation weight based on frame position
                # If we have n intermediate frames, interpolate linearly
                num_total_frames = num_target_cams // self.num_cams
                if num_total_frames > 1:
                    alpha = (frame_position + 1) / (num_total_frames + 1)  # Interpolation weight
                else:
                    alpha = 0.5

                # Linear interpolation of depth
                ## Haibao: why?
                target_depth = (1 - alpha) * depth_frame0 + alpha * depth_frame1
            else:
                # Only one key frame available, use its depth
                target_depth = context_depths_reshaped[:, cam_idx, :, :]

            # Compute 3D points for target frame
            target_e2c = torch.inverse(target_c2e_extr)
            target_depth_flat = rearrange(target_depth, 'b h w -> b h w')

            # Unproject to 3D
            from utils.gs_utils import depth2pc
            target_xyz_single = depth2pc(target_depth_flat, target_e2c, target_K).detach()

            target_depth_list.append(rearrange(target_depth, 'b h w -> b (h w)'))
            target_xyz_list.append(target_xyz_single)

        if len(target_depth_list) > 0:
            # Concatenate all target frame depths
            target_depth_maps = torch.cat(target_depth_list, dim=1)  # [b, (target_cams * h * w)]

            # Stack and rearrange target xyz
            target_xyz = rearrange(
                torch.stack(target_xyz_list, dim=1),
                'b c h w k -> b (c h w) k'
            )  # [b, (target_cams * h * w), 3]

            return target_depth_maps, target_xyz
        else:
            return None, None

    def to_depth(self, depth_in, K_in):
        """
        This function transforms disparity value into depth map while multiplying the value with the focal length.
        """

        min_depth = self.min_depth
        max_depth = self.max_depth
        depth_range = max_depth-min_depth
        depth = min_depth + depth_range * depth_in

        return depth

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

    def transform_gaussian_rotations(self, quaternions, transform_matrix):
        """
        Transform Gaussian rotation quaternions using the rotation part of a 4x4 transformation matrix
        Args:
            quaternions: [B, N, 4] tensor of quaternions (w, x, y, z)
            transform_matrix: [B, 4, 4] transformation matrix
        Returns:
            transformed_quaternions: [B, N, 4] transformed quaternions
        """
        # Extract rotation matrix from transformation (3x3 upper-left block)
        rotation_matrix = transform_matrix[:, :3, :3]  # [B, 3, 3]

        # Convert quaternions to rotation matrices
        # Quaternion format: [w, x, y, z]
        w, x, y, z = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]

        # Build rotation matrices from quaternions
        # R = [[1-2(y²+z²), 2(xy-wz), 2(xz+wy)],
        #      [2(xy+wz), 1-2(x²+z²), 2(yz-wx)],
        #      [2(xz-wy), 2(yz+wx), 1-2(x²+y²)]]
        R = torch.stack([
            torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1),
            torch.stack([2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)], dim=-1),
            torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)], dim=-1)
        ], dim=-2)  # [B, N, 3, 3]

        # Apply transformation: R_new = rotation_matrix @ R_old
        # rotation_matrix: [B, 3, 3], R: [B, N, 3, 3]
        rotation_matrix_expanded = rotation_matrix.unsqueeze(1)  # [B, 1, 3, 3]
        R_transformed = torch.matmul(rotation_matrix_expanded, R)  # [B, N, 3, 3]

        # Convert back to quaternions
        # Using Shepperd's method for numerical stability
        trace = R_transformed[..., 0, 0] + R_transformed[..., 1, 1] + R_transformed[..., 2, 2]

        w_new = torch.sqrt(1.0 + trace) / 2.0
        w4 = 4.0 * w_new
        x_new = (R_transformed[..., 2, 1] - R_transformed[..., 1, 2]) / w4
        y_new = (R_transformed[..., 0, 2] - R_transformed[..., 2, 0]) / w4
        z_new = (R_transformed[..., 1, 0] - R_transformed[..., 0, 1]) / w4

        quaternions_new = torch.stack([w_new, x_new, y_new, z_new], dim=-1)

        # Normalize quaternions
        quaternions_new = quaternions_new / (torch.norm(quaternions_new, dim=-1, keepdim=True) + 1e-8)

        return quaternions_new
    
    def interpolate_transformation(self, T_0to1, t):
        """
        Interpolate transformation matrices using SE(3) exponential map (screw interpolation).
        This properly handles the coupling between rotation and translation.

        Args:
            T_0to1: [batch_size, 4, 4] transformation matrices from identity to final pose
            t: scalar or [batch_size] tensor of interpolation parameters (0 to 1)

        Returns:
            T_interp: [batch_size, 4, 4] interpolated transformation matrices
        """
        if T_0to1.dim() == 4 and T_0to1.shape[1] == 1:
            T_0to1 = T_0to1.squeeze(1)

        bs, device = T_0to1.shape[0], T_0to1.device

        # Compute the matrix logarithm to get twist coordinates
        # log(T) gives us the se(3) representation (twist)
        twist = self.matrix_log_SE3(T_0to1)  # [batch_size, 6] - (ω, v)

        # Scale twist by interpolation parameter
        # Handle both scalar and per-batch t
        if isinstance(t, torch.Tensor) and t.dim() == 1:
            t = t.unsqueeze(1)  # [batch_size, 1] for broadcasting
        twist_scaled = t * twist  # [batch_size, 6]

        # Convert back to SE(3) using matrix exponential
        T_interp = self.matrix_exp_SE3(twist_scaled)  # [batch_size, 4, 4]

        return T_interp

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

    def matrix_log_SE3(self, T):
        """
        Compute the matrix logarithm of SE(3) transformation to get twist coordinates.

        Args:
            T: [batch_size, 4, 4] transformation matrices

        Returns:
            twist: [batch_size, 6] twist coordinates (ω, v) where ω is angular velocity, v is linear velocity
        """
        bs = T.shape[0]
        device = T.device
        dtype = T.dtype

        R = T[:, :3, :3]  # [batch_size, 3, 3]
        p = T[:, :3, 3]   # [batch_size, 3]

        # Compute rotation angle from trace
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        theta = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))  # [batch_size]

        # Initialize output
        omega = torch.zeros(bs, 3, device=device, dtype=dtype)
        v = torch.zeros(bs, 3, device=device, dtype=dtype)

        # Case 1: theta ≈ 0 (small rotation)
        small_angle_mask = theta < 1e-6
        if small_angle_mask.any():
            # For small angles, use linear approximation
            v[small_angle_mask] = p[small_angle_mask]
            # omega stays zero for small angles

        # Case 2: theta ≈ π (180° rotation)
        large_angle_mask = (theta > (np.pi - 1e-6))
        if large_angle_mask.any():
            # Special handling for 180° rotations
            # Find the axis from the eigenvector corresponding to eigenvalue 1
            # Use the diagonal elements to find the axis
            for i in range(bs):
                if large_angle_mask[i]:
                    # Find largest diagonal element
                    diag = torch.diagonal(R[i] + torch.eye(3, device=device, dtype=dtype))
                    k = torch.argmax(diag)
                    omega_k = torch.sqrt(diag[k] / 2)
                    omega_vec = (R[i, :, k] + torch.eye(3, device=device, dtype=dtype)[:, k]) / (2 * omega_k)
                    omega[i] = theta[i] * omega_vec

                    # Compute V inverse
                    omega_hat = self.skew_symmetric(omega[i:i+1])[0]
                    V_inv = (torch.eye(3, device=device, dtype=dtype)
                            - 0.5 * omega_hat
                            + (1.0 / (theta[i] ** 2)) * (1 - (theta[i] / 2) / torch.tan(theta[i] / 2))
                            * (omega_hat @ omega_hat))
                    v[i] = V_inv @ p[i]

        # Case 3: normal case (0 < theta < π)
        normal_mask = ~small_angle_mask & ~large_angle_mask
        if normal_mask.any():
            # Extract rotation axis from skew-symmetric part
            theta_normal = theta[normal_mask]  # [n]
            R_normal = R[normal_mask]  # [n, 3, 3]
            p_normal = p[normal_mask]  # [n, 3]
            n = theta_normal.shape[0]

            # omega = theta * axis, where axis is from skew-symmetric part of log(R)
            theta_expanded = theta_normal.view(n, 1, 1)  # [n, 1, 1]
            omega_hat = (R_normal - R_normal.transpose(1, 2)) / (2 * torch.sin(theta_expanded))  # [n, 3, 3]

            # Extract axis from skew-symmetric matrix
            omega_normal = torch.stack([
                omega_hat[:, 2, 1],
                omega_hat[:, 0, 2],
                omega_hat[:, 1, 0]
            ], dim=1)  # [n, 3]

            omega_normal = theta_normal.unsqueeze(-1) * omega_normal  # Scale by theta -> [n, 3]
            omega[normal_mask] = omega_normal

            # Compute V inverse to get v from p
            # V^{-1} = I - (1/2)[ω] + (1/θ²)(1 - (θ/2)/tan(θ/2))[ω]²
            omega_hat_scaled = self.skew_symmetric(omega_normal)  # [n, 3, 3]
            theta_sq = theta_normal ** 2  # [n]

            # Compute coefficient for [ω]²
            theta_half = theta_normal / 2  # [n]
            coeff = (1.0 / theta_sq) * (1 - theta_half / torch.tan(theta_half))  # [n]
            coeff = coeff.view(n, 1, 1)  # [n, 1, 1]

            eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(n, -1, -1)
            omega_hat_sq = torch.bmm(omega_hat_scaled, omega_hat_scaled)  # [n, 3, 3]

            V_inv = (eye
                    - 0.5 * omega_hat_scaled
                    + coeff * omega_hat_sq)

            v_normal = torch.bmm(V_inv, p_normal.unsqueeze(-1)).squeeze(-1)  # [n, 3]
            v[normal_mask] = v_normal

        # Combine omega and v into twist
        twist = torch.cat([omega, v], dim=1)  # [batch_size, 6]
        return twist

    def matrix_exp_SE3(self, twist):
        """
        Compute the matrix exponential of se(3) twist to get SE(3) transformation.

        Args:
            twist: [batch_size, 6] twist coordinates (ω, v)

        Returns:
            T: [batch_size, 4, 4] transformation matrices
        """
        bs = twist.shape[0]
        device = twist.device
        dtype = twist.dtype

        omega = twist[:, :3]  # [batch_size, 3] angular velocity
        v = twist[:, 3:]      # [batch_size, 3] linear velocity

        # Compute rotation angle
        theta = torch.norm(omega, dim=1)  # [batch_size]

        # Initialize outputs
        R = torch.zeros(bs, 3, 3, device=device, dtype=dtype)
        p = torch.zeros(bs, 3, device=device, dtype=dtype)

        # Case 1: theta ≈ 0 (no rotation)
        small_angle_mask = theta < 1e-6
        if small_angle_mask.any():
            num_small = small_angle_mask.sum().item()
            R[small_angle_mask] = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(num_small, -1, -1)
            p[small_angle_mask] = v[small_angle_mask]

        # Case 2: normal case (theta > 0)
        normal_mask = ~small_angle_mask
        if normal_mask.any():
            theta_normal = theta[normal_mask]  # [n]
            omega_normal = omega[normal_mask]  # [n, 3]
            v_normal = v[normal_mask]  # [n, 3]
            n = theta_normal.shape[0]

            # Normalize omega to get axis
            axis = omega_normal / theta_normal.unsqueeze(-1)  # [n, 3]

            # Compute rotation using Rodrigues' formula
            # R = I + sin(θ)[ω̂] + (1-cos(θ))[ω̂]²
            omega_hat = self.skew_symmetric(axis)  # [n, 3, 3]
            eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(n, -1, -1)

            sin_theta = torch.sin(theta_normal).view(n, 1, 1)  # [n, 1, 1]
            cos_theta = torch.cos(theta_normal).view(n, 1, 1)  # [n, 1, 1]

            # Use torch.bmm for batch matrix multiplication
            omega_hat_sq = torch.bmm(omega_hat, omega_hat)  # [n, 3, 3]

            R_normal = (eye
                       + sin_theta * omega_hat
                       + (1 - cos_theta) * omega_hat_sq)
            R[normal_mask] = R_normal

            # Compute translation using V matrix
            # V = I + ((1-cos(θ))/θ)[ω̂] + ((θ-sin(θ))/θ)[ω̂]²
            theta_normal_expanded = theta_normal.view(n, 1, 1)  # [n, 1, 1]
            coeff1 = (1 - cos_theta) / theta_normal_expanded  # [n, 1, 1]
            coeff2 = (theta_normal_expanded - sin_theta) / theta_normal_expanded  # [n, 1, 1]

            V = (eye
                + coeff1 * omega_hat
                + coeff2 * omega_hat_sq)

            # Matrix-vector multiplication: V @ v
            p_normal = torch.bmm(V, v_normal.unsqueeze(-1)).squeeze(-1)  # [n, 3]
            p[normal_mask] = p_normal

        # Construct homogeneous transformation matrix
        T = torch.zeros(bs, 4, 4, device=device, dtype=dtype)
        T[:, :3, :3] = R
        T[:, :3, 3] = p
        T[:, 3, 3] = 1.0

        return T
    
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
    
    def quaternion_to_matrix(self, q):
        """Convert quaternion to rotation matrix"""
        q = q / torch.norm(q, dim=1, keepdim=True)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        R = torch.zeros(q.shape[0], 3, 3, device=q.device)
        R[:, 0, 0] = 1 - 2*(y*y + z*z)
        R[:, 0, 1] = 2*(x*y - w*z)
        R[:, 0, 2] = 2*(x*z + w*y)
        R[:, 1, 0] = 2*(x*y + w*z)
        R[:, 1, 1] = 1 - 2*(x*x + z*z)
        R[:, 1, 2] = 2*(y*z - w*x)
        R[:, 2, 0] = 2*(x*z - w*y)
        R[:, 2, 1] = 2*(y*z + w*x)
        R[:, 2, 2] = 1 - 2*(x*x + y*y)
        
        return R
    
    def quaternion_slerp(self, q1, q2, t):
        """Spherical linear interpolation between two quaternions"""
        dot = torch.sum(q1 * q2, dim=1)
        
        # Take shorter path if needed
        mask = dot < 0
        q2 = torch.where(mask.unsqueeze(1), -q2, q2)
        dot = torch.where(mask, -dot, dot)
        
        dot = torch.clamp(dot, -0.999999, 0.999999)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        
        # Linear interpolation for small angles
        mask_linear = torch.abs(sin_theta) < 1e-6
        
        s1 = torch.where(mask_linear, 1.0 - t, torch.sin((1.0 - t) * theta) / sin_theta)
        s2 = torch.where(mask_linear, t, torch.sin(t * theta) / sin_theta)
        
        q_interp = s1.unsqueeze(1) * q1 + s2.unsqueeze(1) * q2
        
        return q_interp / torch.norm(q_interp, dim=1, keepdim=True)
    
    
    def refine_ego_transformation_icp(self, xyz_frame0, xyz_frame1, initial_T, max_iterations=20, tolerance=1e-6, cam_id=0):
        """
        Refine ego transformation using ICP (Iterative Closest Point) algorithm.
        Memory-efficient version that processes points in batches.

        Args:
            xyz_frame0: [batch_size, n_points, 3] - points from frame 0 in ego0 coordinate
            xyz_frame1: [batch_size, n_points, 3] - points from frame 1 in ego1 coordinate
            initial_T: [batch_size, 4, 4] - initial transformation (ego0 to ego1 per variable naming,
                       but semantically used to transform frame1 points to align with frame0)
            max_iterations: maximum ICP iterations
            tolerance: convergence tolerance
            cam_id: camera ID for debug logging (default: 0)
        Returns:
            refined_T: [batch_size, 4, 4] - refined transformation
        """
        batch_size = xyz_frame0.shape[0]
        device = xyz_frame0.device

        # Initialize with the provided transformation
        T = initial_T.clone()

        # For each batch
        refined_T_list = []
        for b in range(batch_size):
            src_points = xyz_frame1[b]  # Source points from frame1 to transform
            tgt_points = xyz_frame0[b]  # Target points from frame0 (reference)
            
            # Further subsample if still too many points to avoid OOM
            max_points = 30000  # Maximum points to use for ICP
            if src_points.shape[0] > max_points:
                indices = torch.randperm(src_points.shape[0], device=device)[:max_points]
                src_points = src_points[indices]
            if tgt_points.shape[0] > max_points:
                indices = torch.randperm(tgt_points.shape[0], device=device)[:max_points]
                tgt_points = tgt_points[indices]
            
            # Current transformation for this batch
            T_b = T[b:b+1]

            prev_error = float('inf')

            for iteration in range(max_iterations):
                # Transform source points to target frame: P_target = T @ P_source
                src_transformed = self.transform_points(src_points.unsqueeze(0), T_b).squeeze(0)
                
                # Find nearest neighbors using chunked computation to save memory
                chunk_size = 1000  # Process this many source points at a time
                nearest_indices = []
                min_dists = []
                
                for i in range(0, src_transformed.shape[0], chunk_size):
                    end_idx = min(i + chunk_size, src_transformed.shape[0])
                    src_chunk = src_transformed[i:end_idx]
                    
                    # Compute distances for this chunk
                    dists_chunk = torch.cdist(src_chunk, tgt_points, p=2)
                    
                    # Find nearest neighbors for this chunk
                    min_dists_chunk, nearest_idx_chunk = dists_chunk.min(dim=1)
                    nearest_indices.append(nearest_idx_chunk)
                    min_dists.append(min_dists_chunk)
                
                # Concatenate results
                nearest_indices = torch.cat(nearest_indices)
                min_dists = torch.cat(min_dists)
                
                # Get corresponding points
                src_matched = src_transformed
                tgt_matched = tgt_points[nearest_indices]
                
                # Filter outliers based on distance threshold
                dist_threshold = min_dists.median() * 3.0  # Adaptive threshold
                valid_mask = min_dists < dist_threshold

                if valid_mask.sum() < 100:  # Need minimum points for robust estimation
                    break

                src_matched = src_matched[valid_mask]
                tgt_matched = tgt_matched[valid_mask]

                # Compute centroids
                src_centroid = src_matched.mean(dim=0, keepdim=True)
                tgt_centroid = tgt_matched.mean(dim=0, keepdim=True)
                
                # Center the points
                src_centered = src_matched - src_centroid
                tgt_centered = tgt_matched - tgt_centroid

                # Compute cross-covariance matrix using Kabsch algorithm
                # H = src_centered.T @ tgt_centered [3, N] @ [N, 3] = [3, 3]
                H = src_centered.T @ tgt_centered

                # SVD for rotation using Kabsch algorithm
                try:
                    U, S, Vt = torch.linalg.svd(H)
                    # Optimal rotation R = Vt.T @ U.T (standard Kabsch formula)
                    R = Vt.T @ U.T

                    # Ensure proper rotation (det(R) = 1)
                    if torch.det(R) < 0:
                        # Correct for reflection
                        Vt_corrected = Vt.clone()
                        Vt_corrected[-1, :] *= -1
                        R = Vt_corrected.T @ U.T
                        
                except Exception as e:
                    # If SVD fails or other issues, return initial transformation
                    print(f"[ICP Cam {cam_id}] Refinement failed for batch {b}, iteration {iteration}: {e}")
                    refined_T_list.append(initial_T[b])
                    break

                # Compute translation
                t = tgt_centroid.squeeze() - (R @ src_centroid.squeeze().unsqueeze(-1)).squeeze()

                # Build incremental transformation
                T_increment = torch.eye(4, device=device)
                T_increment[:3, :3] = R
                T_increment[:3, 3] = t

                # Check if transformation is valid (no NaN or inf values)
                if torch.isnan(T_increment).any() or torch.isinf(T_increment).any():
                    print(f"[ICP Cam {cam_id}] Invalid transformation detected at batch {b}, iteration {iteration}")
                    refined_T_list.append(initial_T[b])
                    break
                
                # Update total transformation
                T_b = T_increment.unsqueeze(0) @ T_b

                # Check convergence
                error = (src_matched - tgt_matched).norm(dim=1).mean().item()
                if abs(prev_error - error) < tolerance:
                    break
                prev_error = error

            # Add refined transformation for this batch (if not already added due to early break/failure)
            if len(refined_T_list) <= b:
                # Verify refined transformation is valid and reasonable
                is_valid = not (torch.isnan(T_b).any() or torch.isinf(T_b).any())

                if is_valid:
                    # Check if refinement changed too much (likely wrong convergence)
                    initial_trans = initial_T[b, :3, 3]
                    refined_trans = T_b.squeeze(0)[:3, 3]
                    trans_diff = torch.norm(refined_trans - initial_trans).item()

                    # Reject if translation changed by more than 1 meters (suspicious)
                    # print(f"[ICP Cam {cam_id}] Refinement for batch {b}: translation changed by {trans_diff:.2f}m")
                    if trans_diff > 1.0:
                        print(f"[ICP Cam {cam_id}] Refinement rejected for batch {b}: translation changed by {trans_diff:.2f}m (>1.0m)")
                        is_valid = False

                if is_valid:
                    refined_T_list.append(T_b.squeeze(0))
                else:
                    refined_T_list.append(initial_T[b])
        
        refined_T = torch.stack(refined_T_list, dim=0)
        return refined_T

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
                # Get transformation based on configuration
                if self.use_gt_ego_trans:
                    # Use per-frame GT cam_T_cam transformations: [batch, num_cameras, 4, 4]
                    cam_T_cam_key = ('cam_T_cam', 0, frame_id)
                    cam_T_cam_0to_delta = input_all[cam_T_cam_key]
                else:
                    # Interpolate ego transformation for each camera
                        ego_T_ego_0to_cs = recontrast_data['ego_T_ego_refined']  # [batch, 4, 4]
                    else:
                        ego_T_ego_0to_cs = recontrast_data['ego_T_ego_original']  # [batch, 4, 4]

                    # Compute per-camera delta_t and interpolate for each camera
                    # Use cam 0 as reference for ego trajectory endpoints (consistent with dataset)
                    if 'timestamp' in input_all:
                        # Timestamps in microseconds, keep float64 precision
                        # Timestamp indexing: frame N, camera C at index (N * num_cams + C)
                        timestamp_0 = input_all['timestamp'][:, 0].double()  # Frame 0, cam 0: [batch]
                        timestamp_cs = input_all['timestamp'][:, self.context_span * self.num_cams].double()
                        time_range = timestamp_cs - timestamp_0

                        # Get all camera timestamps at frame_id: [batch, num_cams]
                        cam_timestamp_indices = [frame_id * self.num_cams + cam_id for cam_id in range(self.num_cams)]
                        cam_timestamps = input_all['timestamp'][:, cam_timestamp_indices].double()

                        # Compute per-batch, per-camera delta_t: [batch, num_cams]
                        time_offsets = cam_timestamps - timestamp_0.unsqueeze(1)
                        delta_ts = time_offsets / (time_range.unsqueeze(1) + 1e-6)

                        # Interpolate ego transformation for each camera's timestamp
                        ego_T_ego_list = []
                        for cam_id in range(self.num_cams):
                            delta_t_cam = delta_ts[:, cam_id].float()  # [batch] - per-batch delta_t for this camera
                            ego_T_ego_cam = self.interpolate_transformation(ego_T_ego_0to_cs, delta_t_cam)  # [batch, 4, 4]
                            ego_T_ego_list.append(ego_T_ego_cam)
                    else:
                        # Fallback to uniform spacing if timestamps not available
                        delta_t = frame_id / self.context_span
                        ego_T_ego_list = [self.interpolate_transformation(ego_T_ego_0to_cs, delta_t) for _ in range(self.num_cams)]

                    cam_T_cam_0to_delta = torch.stack(ego_T_ego_list, dim=1)
            else:
                cam_T_cam_0to_delta = None

            # Project xyz to depth maps (xyz in ego_0 frame, pc2depth handles visibility)
            projected_depths = {}
            gt_depths = {}
            for cam_id in range(self.num_cams):
                K = render_data[('K', frame_id, cam_id)]  # [batch, 4, 4]

                if cam_T_cam_0to_delta is not None:
                    transform_cam = cam_T_cam_0to_delta[:, cam_id, :, :]  # [batch, 4, 4]

                    if self.use_gt_ego_trans:
                        # GT mode: cam_T_cam @ inv(c2e_0)
                        # Note: frame 0 might not be in render_data during training (sampled frames)
                        c2e_extr_0 = input_all['c2e_extr'][:, cam_id, ...]  # [batch, 4, 4]
                        e2c_extr_0 = torch.linalg.inv(c2e_extr_0)
                        e2c_extr = torch.matmul(transform_cam, e2c_extr_0)
                    else:
                        # Interpolation mode: e2c_N @ ego_T_ego
                        e2c_extr_N = render_data[('e2c_extr', frame_id, cam_id)]  # [batch, 4, 4]
                        e2c_extr = torch.matmul(e2c_extr_N, transform_cam)
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
                rot_i = recontrast_data['rot_maps'][i]
                scale_i = recontrast_data['scale_maps'][i]
                opacity_i = recontrast_data['opacity_maps'][i]
                sh_i = recontrast_data['sh_maps'][i]

                e2c_extr_i, K_i = [], []
                for cam_id in range(self.num_cams):
                    if cam_T_cam_0to_delta is not None:
                        transform_cam = cam_T_cam_0to_delta[i, cam_id, :, :]  # [4, 4]

                        if self.use_gt_ego_trans:
                            # GT mode: cam_T_cam @ inv(c2e_0)
                            # Note: frame 0 might not be in render_data during training (sampled frames)
                            c2e_data = input_all['c2e_extr'][i, cam_id, ...]
                            e2c_data = torch.linalg.inv(c2e_data)
                            e2c_data = torch.matmul(transform_cam, e2c_data)
                        else:
                            # Interpolation mode: e2c_N @ ego_T_ego
                            e2c_data = render_data[('e2c_extr', frame_id, cam_id)][i]
                            e2c_data = torch.matmul(e2c_data, transform_cam)
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
                    e2c_extr_i,  # [1, 4, 4]
                    K_i,  # [1, 3, 3]
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

    def save_warping_visualization(self, ref_colors, ref_depths, src_colors,
                                   warped_pred, warped_mask,
                                   ref_frame_id, src_frame_id, cam_id):
        """Save visualization for image warping results"""
        import torchvision.utils as vutils
        import matplotlib.pyplot as plt
        import numpy as np

        # Create directory - use stage and batch_idx based naming
        if hasattr(self, 'stage'):
            stage_name = self.stage
        else:
            stage_name = 'train'

        if hasattr(self, 'current_batch_idx'):
            batch_idx = self.current_batch_idx
        else:
            batch_idx = self.saved_steps_count if hasattr(self, 'saved_steps_count') else 0

        warp_dir = os.path.join(self.save_images_dir, f'{stage_name}_{batch_idx}_warping')
        os.makedirs(warp_dir, exist_ok=True)

        # Get camera name
        cam_name = self.camera_names[cam_id] if cam_id < len(self.camera_names) else f'CAM_{cam_id}'

        # Save first batch item only
        ref_colors_vis = ref_colors[0].detach().cpu()
        ref_depths_vis = ref_depths[0].detach().cpu()
        src_colors_vis = src_colors[0].detach().cpu()
        warped_pred_vis = warped_pred[0].detach().cpu()
        warped_mask_vis = warped_mask[0].detach().cpu()

        # 1. Save reference color
        ref_color_path = os.path.join(
            warp_dir,
            f'ref_frame_{ref_frame_id}_{cam_name}_ref_color.png'
        )
        vutils.save_image(ref_colors_vis, ref_color_path)

        # 2. Save source color
        src_color_path = os.path.join(
            warp_dir,
            f'ref_frame_{ref_frame_id}_{cam_name}_src_frame_{src_frame_id}_color.png'
        )
        vutils.save_image(src_colors_vis, src_color_path)

        # 3. Save warped prediction
        warped_pred_path = os.path.join(
            warp_dir,
            f'ref_frame_{ref_frame_id}_{cam_name}_src_frame_{src_frame_id}_warped_pred.png'
        )
        vutils.save_image(warped_pred_vis, warped_pred_path)

        # 4. Save warped mask
        warped_mask_path = os.path.join(
            warp_dir,
            f'ref_frame_{ref_frame_id}_{cam_name}_src_frame_{src_frame_id}_warped_mask.png'
        )
        vutils.save_image(warped_mask_vis, warped_mask_path)

        # 5. Save depth as heatmap
        depth_heatmap_path = os.path.join(
            warp_dir,
            f'ref_frame_{ref_frame_id}_{cam_name}_ref_depth.png'
        )
        # Normalize depth for visualization
        depth_np = ref_depths_vis.squeeze().numpy()
        depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)

        plt.figure(figsize=(10, 6))
        plt.imshow(depth_normalized, cmap='turbo')
        plt.colorbar(label='Normalized Depth')
        plt.title(f'Ref Frame {ref_frame_id} - {cam_name} - Depth')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(depth_heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()

        # 6. Save comparison grid (ref_color, src_color, warped_pred, warped_mask)
        comparison_path = os.path.join(
            warp_dir,
            f'ref_frame_{ref_frame_id}_{cam_name}_src_frame_{src_frame_id}_comparison.png'
        )

        # Create a 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Reference color
        ref_img = ref_colors_vis.permute(1, 2, 0).numpy()
        axes[0, 0].imshow(np.clip(ref_img, 0, 1))
        axes[0, 0].set_title(f'Reference (Frame {ref_frame_id})')
        axes[0, 0].axis('off')

        # Source color
        src_img = src_colors_vis.permute(1, 2, 0).numpy()
        axes[0, 1].imshow(np.clip(src_img, 0, 1))
        axes[0, 1].set_title(f'Source (Frame {src_frame_id})')
        axes[0, 1].axis('off')

        # Warped prediction
        warped_img = warped_pred_vis.permute(1, 2, 0).numpy()
        axes[1, 0].imshow(np.clip(warped_img, 0, 1))
        axes[1, 0].set_title(f'Warped Prediction (F{src_frame_id}→F{ref_frame_id})')
        axes[1, 0].axis('off')

        # Warped mask
        mask_img = warped_mask_vis.squeeze().numpy()
        axes[1, 1].imshow(mask_img, cmap='gray', vmin=0, vmax=1)
        axes[1, 1].set_title('Warped Mask (Valid Regions)')
        axes[1, 1].axis('off')

        plt.suptitle(f'{cam_name} - Warping Visualization', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()

        # 7. Save overlay visualization (warped on ref with mask)
        overlay_path = os.path.join(
            warp_dir,
            f'step_{self.saved_steps_count}_ref_frame_{ref_frame_id}_{cam_name}_overlay.png'
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Reference
        axes[0].imshow(np.clip(ref_img, 0, 1))
        axes[0].set_title(f'Reference (Frame {ref_frame_id})')
        axes[0].axis('off')

        # Warped with mask overlay
        warped_masked = warped_img * mask_img[..., None]
        axes[1].imshow(np.clip(warped_masked, 0, 1))
        axes[1].set_title(f'Warped × Mask')
        axes[1].axis('off')

        # Difference
        diff = np.abs(ref_img - warped_masked)
        axes[2].imshow(diff)
        axes[2].set_title('Absolute Difference')
        axes[2].axis('off')

        plt.suptitle(f'{cam_name} - Warping Overlay & Difference', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
        plt.close()

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
        
        
        # X = (u - cx) * depth / fx
        # Y = (v - cy) * depth / fy
        # Z = depth
          # (Bs, H, W)
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
    def _save_warping_images(self, save_name, batch_render_project_data, bs_id=0):
        """Save warping visualization images during training"""
        # Extract batch_idx from save_name (format: stage_batchidx_warping)
        parts = save_name.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            self.current_batch_idx = int(parts[1])

        # Set save directory for save_warping_visualization
        self.save_images_dir = self.save_dir

        # Extract warping data and save visualizations
        ref_frame_id = 0  # Always frame 0 as reference

        # Dynamically determine available source frames from the data
        src_frame_ids = set()
        for key in batch_render_project_data.keys():
            if isinstance(key, tuple) and len(key) == 4:
                if key[0] == 'warped_pred' and key[1] == ref_frame_id:
                    src_frame_ids.add(key[2])  # key[2] is src_frame_id
        src_frame_ids = sorted(list(src_frame_ids))

        # If no source frames found, return early
        if not src_frame_ids:
            return

        for cam_id in range(self.num_cams):
            # Get reference data
            if ('ref_colors', ref_frame_id, cam_id) in batch_render_project_data:
                ref_colors = batch_render_project_data[('ref_colors', ref_frame_id, cam_id)]
                ref_depths = batch_render_project_data[('ref_depths', ref_frame_id, cam_id)]

                for src_frame_id in src_frame_ids:
                    # Get source colors
                    if ('src_colors', src_frame_id, cam_id) in batch_render_project_data:
                        src_colors = batch_render_project_data[('src_colors', src_frame_id, cam_id)]
                    else:
                        continue

                    # Get warped results
                    warped_pred_key = ('warped_pred', ref_frame_id, src_frame_id, cam_id)
                    warped_mask_key = ('warped_mask', ref_frame_id, src_frame_id, cam_id)

                    if warped_pred_key in batch_render_project_data and warped_mask_key in batch_render_project_data:
                        warped_pred = batch_render_project_data[warped_pred_key]
                        warped_mask = batch_render_project_data[warped_mask_key]

                        # Call the original save_warping_visualization
                        self.save_warping_visualization(
                            ref_colors, ref_depths, src_colors,
                            warped_pred, warped_mask,
                            ref_frame_id, src_frame_id, cam_id
                        )

    @rank_zero_only
    def _save_reprojection_images(self, save_name, batch_data, bs_id=0):
        if os.path.exists(os.path.join(self.save_dir, save_name)):
            shutil.rmtree(os.path.join(self.save_dir, save_name))
        os.makedirs(os.path.join(self.save_dir, save_name))
        
        for frame_id in self.all_render_frame_ids:
            for cam_id in range(self.num_cams):
                ref_colors = batch_data[('ref_colors', frame_id, cam_id)].detach()
                ref_depths = batch_data[('ref_depths', frame_id, cam_id)].detach()
                ref_depths = torch.log(ref_depths)
                max_depth = log(self.max_depth)
                min_depth = log(self.min_depth)
                norm_depths = 1 - (ref_depths - min_depth) / (max_depth - min_depth)
                self.save_image(ref_colors[bs_id], os.path.join(self.save_dir, save_name, f"{frame_id}_{cam_id}_gt.png"))
                self.save_image(norm_depths[bs_id], os.path.join(self.save_dir, save_name, f"{frame_id}_{cam_id}_depth.png"))

                warped_img = batch_data[('warped_pred', frame_id, cam_id)].detach()
                self.save_image(warped_img[bs_id], os.path.join(self.save_dir, save_name, f"{frame_id}_{cam_id}_preds.png"))


    
    @rank_zero_only
    def _save_splating_images(self, save_name, batch_data, bs_id=0):
        if os.path.exists(os.path.join(self.save_dir, save_name)):
            shutil.rmtree(os.path.join(self.save_dir, save_name))
        os.makedirs(os.path.join(self.save_dir, save_name))
        for frame_id in self.all_render_frame_ids:
            for cam_id in range(self.num_cams): 
                pred = batch_data[('gaussian_color', frame_id, cam_id)].detach()
                mask = batch_data[('warped_mask', frame_id, cam_id)]
                gt = batch_data[('groudtruth', frame_id, cam_id)]   
                self.save_image(pred[bs_id], os.path.join(self.save_dir, save_name, f"{frame_id}_{self.render_cam_mode}_{cam_id}_preds.png"))
                self.save_image(mask[bs_id], os.path.join(self.save_dir, save_name, f"{frame_id}_{self.render_cam_mode}_{cam_id}_mask.png"))
                self.save_image(gt[bs_id], os.path.join(self.save_dir, save_name, f"{frame_id}_{self.render_cam_mode}_{cam_id}_gt.png")) 
    @rank_zero_only
    def save_image(self, image, path):
        """Save an image. Assumed to be in range 0-1."""

        # Create the parent directory if it doesn't already exist.
        # os.makedirs(os.path.dirname(path),exist_ok=True)

        image = image.detach().cpu().numpy().clip(min=0, max=1)
        image_uint8 = (image * 255).astype(np.uint8)

        if len(image_uint8.shape)==2:
            pass
        elif (len(image_uint8.shape)==3) and (image_uint8.shape[0] == 1):
            image_uint8 = image_uint8[0]
        else:
            assert len(image_uint8.shape)==3, image_uint8.shape
            assert image_uint8.shape[0] == 3, image_uint8.shape
            image_uint8 = image_uint8.transpose(1, 2, 0)
        # Save the image.
        Image.fromarray(image_uint8).save(path)
        torch.cuda.empty_cache()
    
    def save_flow_heatmap(self, flow_3d, original_image, save_path, batch_idx=0, frame_id=0, cam_id=0):
        """Save flow visualization as velocity magnitude mask with different colors."""
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        # Convert tensors to numpy if needed
        if torch.is_tensor(flow_3d):
            flow_3d = flow_3d.detach().cpu().numpy()
        if torch.is_tensor(original_image):
            original_image = original_image.detach().cpu().numpy()
        
        # Reshape flow to image dimensions if needed
        if flow_3d.ndim == 2:
            # Assuming [H*W, 3]
            h = self.height
            w = self.width
            flow_3d = flow_3d.reshape(h, w, 3)
        elif flow_3d.ndim == 1:
            # Assuming flattened [H*W*3]
            h = self.height
            w = self.width
            flow_3d = flow_3d.reshape(h, w, 3)
        
        # Compute flow magnitude (3D velocity magnitude)
        flow_magnitude = np.sqrt(flow_3d[..., 0]**2 + flow_3d[..., 1]**2 + flow_3d[..., 2]**2)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        
        # Original image as background
        if original_image.shape[0] == 3:  # CHW to HWC
            original_image = np.transpose(original_image, (1, 2, 0))
        
        # Show original image
        ax.imshow(original_image.clip(0, 1))
        
        # Create mask for significant flow
        mask = flow_magnitude > 0.01
        
        if mask.any():
            # Overlay velocity magnitude only where there's significant flow
            masked_magnitude = np.ma.masked_where(~mask, flow_magnitude)
            im = ax.imshow(masked_magnitude, cmap='jet', alpha=0.7, interpolation='nearest', 
                          vmin=0, vmax=flow_magnitude[mask].max() if mask.any() else 1.0)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Velocity Magnitude (m)', rotation=270, labelpad=20)
            
            # Add statistics
            non_zero_flow = flow_magnitude[mask]
            stats_text = (f"Vehicles detected: {mask.sum()} pixels\n"
                         f"Max velocity: {non_zero_flow.max():.2f} m\n"
                         f"Mean velocity: {non_zero_flow.mean():.2f} m")
        else:
            stats_text = "No vehicle motion detected"
        
        # Set title with statistics
        ax.set_title(f'Vehicle Velocity Map - Frame {frame_id}, Camera {cam_id}\n{stats_text}')
        ax.axis('off')
        
        # Save figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        # Also save just the velocity magnitude as a separate image
        magnitude_path = save_path.replace('.png', '_magnitude.png')
        plt.figure(figsize=(10, 6))
        plt.imshow(flow_magnitude, cmap='jet', interpolation='nearest')
        plt.colorbar(label='Velocity Magnitude (m)')
        
        # Determine mode from filename
        if 'mode1' in save_path:
            mode_text = 'Mode 1: Using t+1 actual masks'
        elif 'mode2' in save_path:
            mode_text = 'Mode 2: Using t masks with velocity transform'
        else:
            mode_text = ''
        
        plt.title(f'Pure Velocity Magnitude - Frame {frame_id}, Camera {cam_id}\n{mode_text}')
        plt.axis('off')
        plt.savefig(magnitude_path, dpi=100, bbox_inches='tight')
        plt.close()

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
    
    def compute_flow_reg_loss(self, batch_recontrast_data):
        pred_flow = batch_recontrast_data["forward_flow"]
        device = pred_flow.device
        zero_flow = torch.zeros_like(batch_recontrast_data["forward_flow"]).to(device)
        forward_flow_reg = F.mse_loss(pred_flow, zero_flow, reduction="none")
        flow_reg_loss = self.flow_reg_coeff * forward_flow_reg.mean()
        return flow_reg_loss

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
    


if __name__=='__main__':

    import PIL.Image as pil
    import yaml
    import torch
    import numpy as np
    from dataset.vggt3dgs_data_module import VGGT3DGS_LITDataModule

    import math

    config_file = 'configs/nuscenes/vggt3dgs.yaml'
    with open(config_file) as f:
        main_cfg = yaml.load(f, Loader=yaml.FullLoader)
    main_cfg['data_cfg']['batch_size'] = 1
    main_cfg['model_cfg']['batch_size'] = main_cfg['data_cfg']['batch_size']

    pl.seed_everything(main_cfg['seed'], workers=True)

    main_datamodule = VGGT3DGS_LITDataModule(main_cfg['data_cfg'])
    main_datamodule.setup('test')
    test_dataloader = main_datamodule.test_dataloader()

    from pytorch_lightning.loggers import TensorBoardLogger
    

    litmodel = ReconDrive_LITModelModule(
        cfg=main_cfg['model_cfg'],
        logger=TensorBoardLogger(save_dir='.',name='logs')
    )
    restore_ckpt = '/mnt/data_storage/kuntaoxiao/datas/results/DrivingForward/vggt3dgs/test7/ckpt/last-v2.ckpt'
    load_ckpt = torch.load(restore_ckpt,map_location=f"cuda:0")
    litmodel.load_state_dict(load_ckpt['state_dict'])


    litmodel.to('cuda:0')
    for batch_inputs in test_dataloader:
        for key in batch_inputs:
            if isinstance(batch_inputs[key],torch.Tensor):
                batch_inputs[key] = batch_inputs[key].to('cuda:0')
        outputs = litmodel.predict_step(batch_inputs,0)
        recontrast_data = outputs[0]
        render_data = outputs[1]
        splating_data = outputs[-1]

        for i in range(1):
            frame_id = 0

            cam_id = 0
            means = recontrast_data['xyz'][i]
            quats = recontrast_data['rot_maps'][i]
            scales = recontrast_data['scale_maps'][i]
            opacities = recontrast_data['opacity_maps'][i]
            sh_maps = recontrast_data['sh_maps'][i]

            height, width = litmodel.height, litmodel.width

            sh_degree = 4
            K = batch_inputs['K'][i,cam_id]
            recontrast_data['K'] = K
            recontrast_data['height'] = height
            recontrast_data['width'] = width
            recontrast_data['c2w_extr'] = batch_inputs['c2e_extr'][i, cam_id, ...]

            gs_dict = {
                'means':means,
                'quats':quats,
                'scales':scales,
                'opacities':opacities,
                'sh_maps':sh_maps,
                'K':K,
                'height':height,
                'width':width,
                'c2w_extr':batch_inputs['c2e_extr'][i, cam_id, ...],
            }

            e2c_extr =  torch.linalg.inv(batch_inputs['c2e_extr'][i, cam_id, ...]) 

            scale_factor =  640.0 / 518.0
            width, height = int(width * scale_factor), int(height * scale_factor)
            print(K,height,width)
            K[:2] = K[:2] * scale_factor
            print(K)

            render_color, render_alphas, meta = rasterization(
                means,  # [N, 3]
                quats,  # [N, 4]
                scales,  # [N, 3]
                opacities.squeeze(1),  # [N]
                sh_maps,  # [N, K, 3]
                e2c_extr[None],  # [1, 4, 4]
                K[None,:3,:3],  # [1, 3, 3]
                width,
                height,
                sh_degree=sh_degree,
                render_mode="RGB",
                # this is to speedup large-scale rendering by skipping far-away Gaussians.
                # radius_clip=0.1,
            )
            render_rgbs = render_color[0].permute(2,0,1)

            print(render_rgbs.shape,render_rgbs.min(),render_rgbs.mean(),render_rgbs.max())
            render_rgbs_uint8 = (render_rgbs.detach().cpu().numpy().transpose(1,2,0).clip(0,1.0) * 255).astype(np.uint8)
            pil.fromarray(render_rgbs_uint8).save(f'./test_bs{i}_novel.png')

            pred_img = splating_data[('gaussian_color', frame_id, cam_id)][i]

            gt_img = batch_inputs[('color_org', frame_id)][cam_id][i]
            print(gt_img.shape)
            new_gt_img =  F.interpolate(gt_img.unsqueeze(0), 
                                          size=(height,width),
                                          mode = 'bilinear',
                                          align_corners=False)

            img_warped = render_rgbs.unsqueeze(0)
            new_pred_img = pred_img.unsqueeze(0)
            print(img_warped.shape,new_gt_img.shape,new_pred_img.shape)
            psnr = litmodel.compute_psnr(img_warped,new_gt_img)
            ssim = litmodel.compute_ssim(img_warped,new_gt_img)
            lpips = litmodel.compute_lpips(img_warped,new_gt_img)
            print(psnr,ssim,lpips)
            psnr = litmodel.compute_psnr(new_pred_img,new_gt_img)
            ssim = litmodel.compute_ssim(new_pred_img,new_gt_img)
            lpips = litmodel.compute_lpips(new_pred_img,new_gt_img)
            print(psnr,ssim,lpips)

            render_rgbs_uint8 = (new_pred_img[0].detach().cpu().numpy().transpose(1,2,0).clip(0,1.0) * 255).astype(np.uint8)
            pil.fromarray(render_rgbs_uint8).save(f'./test_bs{i}_origin.png')

            render_rgbs_uint8 = (new_gt_img[0].detach().cpu().numpy().transpose(1,2,0).clip(0,1.0) * 255).astype(np.uint8)
            pil.fromarray(render_rgbs_uint8).save(f'./test_bs{i}_gt.png')

        break