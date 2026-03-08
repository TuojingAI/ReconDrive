#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#

import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image as pil

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from .visualize import colormap
from .misc import pretty_ts, cal_depth_error


def set_tb_title(*args):
    """
    This function sets title for tensorboard plot.
    """    
    title = ''
    for i, s in enumerate(args):
        if not i%2: title += '/'
        s = s if isinstance(s, str) else str(s)
        title += s
    return title[1:]
    

def resize_for_tb(image):
    """
    This function resizes images for tensorboard plot.
    """     
    h, w = image.size()[-2:]
    return F.interpolate(image, [h//2, w//2], mode='bilinear', align_corners=True) 
    

def plot_tb(writer, step, img, title, j=0):
    """
    This function plots images on tensotboard.
    """     
    img_resized = resize_for_tb(img)    
    writer.add_image(title, img_resized[j].data, step)


def plot_norm_tb(writer, step, img, title, j=0):
    """
    This function plots normalized images on tensotboard.
    """     
    img_resized = torch.clamp(resize_for_tb(img), 0., 1.)
    writer.add_image(title, img_resized[j].data, step)


def plot_disp_tb(writer, step, disp, title, j=0):
    """
    This function plots disparity maps on tensotboard.
    """  
    disp_resized = resize_for_tb(disp).float()
    disp_resized = colormap(disp_resized[j, 0])
    writer.add_image(title, disp_resized, step)    

    
class Logger:
    """
    Logger class to monitor training
    """
    def __init__(self, cfg, use_tb=True):
        self.read_config(cfg)
        os.makedirs(self.log_path, exist_ok=True)
        
        if use_tb: 
            self.init_tb()
            
        if self.eval_visualize:
            self.init_vis()

        self._metric_names = ['abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3']

    def read_config(self, cfg):
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)
        
    def init_tb(self):
        self.writers = {}
        for mode in ['train', 'val']:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))
        
    def close_tb(self):
        for mode in ['train', 'val']:
            self.writers[mode].close()

    def init_vis(self):
        vis_path = os.path.join(self.log_path, 'vis_results')
        os.makedirs(vis_path, exist_ok=True)
        
        self.cam_paths = []
        for cam_id in range(self.num_cams):
            cam_path = os.path.join(vis_path, f'cam{cam_id:d}')
            os.makedirs(cam_path, exist_ok=True)
            self.cam_paths.append(cam_path)
            
    def get_metric_names(self):
        return self._metric_names
    
    def to_disp(self, depth_in, K_in):
        """
        This function transforms depth values into disparity values while dividing the value by the focal length.
        """
        disp = depth_in * self.focal_length_scale / K_in[:, 0:1, 0:1].unsqueeze(2)
        disp = 1 / (disp + 0.00001)

        min_disp = 1/self.max_depth
        max_disp = 1/self.min_depth
        disp_range = max_disp - min_disp

        disp = (disp - min_disp) / disp_range

        return disp
    
    def update(self, mode, epoch, world_size, batch_idx, step, start_time, before_op_time, inputs, outputs, losses):
        """
        Display logs with respect to the log frequency
        """    
        # iteration duration
        duration = time.time() - before_op_time

        if self.is_checkpoint(step):
            self.log_time(epoch, batch_idx * world_size, duration, losses, start_time)
            self.log_tb(mode, inputs, outputs, losses, step)
                
    def is_checkpoint(self, step):
        """ 
        Log less frequently after the early phase steps
        """
        early_phase = (step % self.log_frequency == 0) and (step < self.early_phase)
        late_phase = step % self.late_log_frequency == 0
        return (early_phase or late_phase)

    def log_time(self, epoch, batch_idx, duration, loss, start_time):
        """
        This function prints epoch, iteration, duration, loss and spent time.
        """
        rep_loss = loss['total_loss'].item()
        samples_per_sec = self.batch_size / duration
        time_sofar = time.time() - start_time
        print("")
        print(f'epoch: {epoch:2d} | batch: {batch_idx:6d} |' + \
              f'examples/s: {samples_per_sec:5.1f} | loss: {rep_loss:.3f} | time elapsed: {pretty_ts(time_sofar)}')
        
    def log_tb(self, mode, inputs, outputs, losses, step):
        """
        This function logs outputs for monitoring using tensorboard.
        """
        writer = self.writers[mode]
        # loss
        for l, v in losses.items():
            writer.add_scalar(f'{l}', v, step)
        
        scale = 0 # plot the maximum scale
        for cam_id in range(self.num_cams):
            target_view = outputs[('cam', cam_id)]
            
            plot_tb(writer, step, inputs[('color', 0, scale)][:, cam_id, ...], set_tb_title('cam', cam_id, 'frame_0')) # frame_id 0
            plot_tb(writer, step, inputs[('color', 1, scale)][:, cam_id, ...], set_tb_title('cam', cam_id, 'frame_1')) # frame_id 1
            plot_tb(writer, step, inputs[('color', -1, scale)][:, cam_id, ...], set_tb_title('cam', cam_id, 'frame_-1')) # frame_id -1
            plot_disp_tb(writer, step, target_view[('disp', scale)], set_tb_title('cam', cam_id, 'disp')) # disparity
            depth_gt = inputs['gt_depth'][:, cam_id, ...]
            far = depth_gt == 0
            depth_gt[far] += 80.0
            disp_gt = self.to_disp(depth_gt, inputs[('K', 0)][:, cam_id, ...])
            plot_disp_tb(writer, step, disp_gt, set_tb_title('cam', cam_id, 'disp_gt'))

            plot_tb(writer, step, target_view[('reproj_loss', scale)], set_tb_title('cam', cam_id, 'reproj')) # reprojection image
            plot_tb(writer, step, target_view[('reproj_mask', scale)], set_tb_title('cam', cam_id, 'reproj_mask')) # reprojection mask
            plot_tb(writer,  step, inputs['mask'][:, cam_id, ...], set_tb_title('cam', cam_id, 'self_occ_mask'))
    
            if self.spatio:
                plot_norm_tb(writer, step, target_view[('overlap', 0, scale)], set_tb_title('cam', cam_id, 'sp'))
                plot_tb(writer, step, target_view[('overlap_mask', 0, scale)], set_tb_title('cam', cam_id, 'sp_mask'))
                
            if self.spatio_temporal:
                for frame_id in self.frame_ids:
                    if frame_id == 0:
                        continue
                    plot_norm_tb(writer, step, target_view[('color', frame_id, scale)], set_tb_title('cam', cam_id, 'pred_', frame_id))                      
                    plot_norm_tb(writer, step, target_view[('overlap', frame_id, scale)], set_tb_title('cam', cam_id, 'sp_tm_', frame_id))
                    plot_tb(writer, step, target_view[('overlap_mask', frame_id, scale)], set_tb_title('cam', cam_id, 'sp_tm_mask_', frame_id))
            
            if self.gaussian:
                if self.novel_view_mode == 'SF':
                    frame_id = 1
                elif self.novel_view_mode == 'MF':
                    frame_id = 0
                else:
                    raise ValueError(f'Novel view mode {self.novel_view_mode} not supported.')
                plot_norm_tb(writer, step, target_view[('gaussian_color', frame_id, scale)], set_tb_title('cam', cam_id, f'gaussian_pred_frame_{frame_id}'))

    def log_result(self, inputs, outputs, idx, syn_visualize=False):
        """
        This function logs outputs for visualization.
        """        
        scale = 0
        for cam_id in range(self.num_cams):
            target_view = outputs[('cam', cam_id)]
            disps = target_view['disp', scale]
            for jdx, disp in enumerate(disps):       
                disp = colormap(disp)[0,...].transpose(1,2,0)
                disp = pil.fromarray((disp * 255).astype(np.uint8))
                cur_idx = idx*self.batch_size + jdx 
                disp.save(os.path.join(self.cam_paths[cam_id], f'{cur_idx:03d}_disp.jpg'))
                
    def compute_depth_losses(self, inputs, outputs, vis_scale=False):
        """
        This function computes depth metrics, to allow monitoring of training process on validation dataset.
        """
        min_eval_depth = self.eval_min_depth
        max_eval_depth = self.eval_max_depth
        
        med_scale = []
        
        error_metric_dict = defaultdict(float)
        error_median_dict = defaultdict(float)
        
        for cam in range(self.num_cams):
            target_view = outputs['cam', cam]

            depth_gt = inputs['gt_depth'][:, cam, ...]
            
            _, _, h, w = depth_gt.shape

            depth_pred = target_view[('depth', 0, 0)].to(depth_gt.device)
            depth_pred = torch.clamp(F.interpolate(
                        depth_pred, [h, w], mode='bilinear', align_corners=False), 
                         min_eval_depth, max_eval_depth)
            depth_pred = depth_pred.detach()

            mask = (depth_gt > min_eval_depth) * (depth_gt < max_eval_depth) * inputs['mask'][:, cam, ...]
            mask = mask.bool()
            
            depth_gt = depth_gt[mask]
            depth_pred = depth_pred[mask]
            
            # calculate median scale
            scale_val = torch.median(depth_gt) / torch.median(depth_pred)
            med_scale.append(round(scale_val.cpu().numpy().item(), 2))
                            
            depth_pred_metric = torch.clamp(depth_pred, min=min_eval_depth, max=max_eval_depth)
            depth_errors_metric = cal_depth_error(depth_pred_metric, depth_gt)
            
            depth_pred_median = torch.clamp(depth_pred * scale_val, min=min_eval_depth, max=max_eval_depth)
            depth_errors_median = cal_depth_error(depth_pred_median, depth_gt)
            
            for i in range(len(depth_errors_metric)):
                key = self._metric_names[i]
                error_metric_dict[key] += depth_errors_metric[i]
                error_median_dict[key] += depth_errors_median[i]

        if vis_scale==True:
            # print median scale
            print(f'          | median scale = {med_scale}')
                
        for key in error_metric_dict.keys():
            error_metric_dict[key] = error_metric_dict[key].cpu().numpy() / self.num_cams
            error_median_dict[key] = error_median_dict[key].cpu().numpy() / self.num_cams
            
        return error_metric_dict, error_median_dict         
                
    def print_perf(self, loss, scale): 
        """
        This function prints various metrics for depth estimation accuracy.
        """
        perf = ' '*3 + scale
        for k, v in loss.items():
            perf += ' | ' + str(k) + f': {v:.3f}'
        print(perf)
            