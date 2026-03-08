
#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#


import os

import numpy as np
import PIL.Image as pil
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from pyquaternion import Quaternion

from dataset.data_util import img_loader, mask_loader_scene, align_dataset, stack_sample




class NuScenesdataset4D(Dataset):
    """
    Loaders for NuScenes dataset
    """
    def __init__(self, path, stage,
                 cameras=None,
                 back_context=0,
                 forward_context=1,
                 data_transform=None,
                 depth_type=None,
                 with_pose=None,
                 with_ego_pose=None,
                 with_mask=None,
                 min_context_num: int = 1,
                 num_context_timesteps: int = 4,
                 num_target_timesteps: int = 4,
                 cache_dir="",
                 nuscenes_version='interp_12Hz_trainval',
                 context_span=6
                 ):        
        super().__init__()
        self.version = nuscenes_version
        self.context_span = context_span
        self.path = path
        self.cache_dir = cache_dir
        self.stage = stage
        self.dataset_idx = 0

        self.cameras = cameras
        self.num_cameras = len(cameras)
        self.bwd = back_context
        self.fwd = forward_context
        
        # self.has_context = back_context + forward_context > 0
        self.has_context = False
        self.data_transform = data_transform

        self.with_depth = depth_type is not None
        self.with_pose = with_pose
        self.with_ego_pose = with_ego_pose

        self.loader = img_loader

        self.with_mask = with_mask
        cur_path = os.path.dirname(os.path.realpath(__file__))        
        self.mask_path = os.path.join(cur_path, 'nuscenes_mask')
        self.mask_loader = mask_loader_scene

        self.dataset = NuScenes(version=self.version, dataroot=self.path, verbose=True)

        if stage == 'train':
            official_scene_names = splits.train  
        elif stage == 'val':
            official_scene_names = splits.val    
        elif stage == 'test':
            official_scene_names = [
                'scene-0014', 'scene-0018', 'scene-0906', 'scene-0098',
                'scene-0100', 'scene-0103', 'scene-0270', 'scene-0271',
                'scene-0278', 'scene-0553', 'scene-0558', 
                'scene-0802', 'scene-0968',  'scene-1065',
            ]
        else:
            raise ValueError("stage should be 'train' / 'val'/ 'test' ")

        self.min_context_num = min_context_num
        self.num_context_timesteps = num_context_timesteps
        self.num_target_timesteps = num_target_timesteps
        
        self.sample_tokens = []
        self.scenes_data = []
        self.scene_names = []
        self.scene_tokens = []
        for scene in self.dataset.scene:
            if scene['name'] in official_scene_names:
                scene_name = scene['name']
                scene_token = scene['token']
                
                # Get all sample tokens for this scene
                sample_token = scene['first_sample_token']
                scene_sample_tokens = []
                visited_tokens = set()  # Prevent infinite loops
                while sample_token:
                    if sample_token in visited_tokens:
                        print(f"Warning: Detected circular reference in scene {scene_name}, breaking loop")
                        break
                    visited_tokens.add(sample_token)
                    scene_sample_tokens.append(sample_token)
                    sample = self.dataset.get('sample', sample_token)
                    sample_token = sample['next']
                
                drop_last = len(scene_sample_tokens) - context_span - 1

                if drop_last > 0:  # Only add scenes with valid samples
                    self.scenes_data.append(scene_sample_tokens[:drop_last])
                    self.scene_names.append(scene_name)
                    self.scene_tokens.append(scene_token)

        print(f'Number of scenes: {len(self.scenes_data)}')
        print(f'Total samples: {sum(len(scene) for scene in self.scenes_data)}')

    def get_current(self, key, cam_sample):
        """
        This function returns samples for current contexts
        """        
        # get current timestamp rgb sample
        if key == 'rgb':
            rgb_path = cam_sample['filename']
            return self.loader(os.path.join(self.path, rgb_path))
        # get current timestamp camera intrinsics
        elif key == 'intrinsics':
            cam_param = self.dataset.get('calibrated_sensor', 
                                         cam_sample['calibrated_sensor_token'])
            return np.array(cam_param['camera_intrinsic'], dtype=np.float32)
        # get current timestamp camera extrinsics
        elif key == 'extrinsics':
            cam_param = self.dataset.get('calibrated_sensor', 
                                         cam_sample['calibrated_sensor_token'])
            return self.get_tranformation_mat(cam_param)
        else:
            raise ValueError('Unknown key: ' +key)

    def get_context(self, key, cam_sample):
        """
        This function returns samples for backward and forward contexts
        """
        bwd_context, fwd_context = [], []
        if self.bwd != 0 and cam_sample['prev']:
            bwd_sample = self.dataset.get('sample_data', cam_sample['prev'])
            bwd_context = [self.get_current(key, bwd_sample)]

        if self.fwd != 0 and cam_sample['next']:
            fwd_sample = self.dataset.get('sample_data', cam_sample['next'])
            fwd_context = [self.get_current(key, fwd_sample)]
        return bwd_context + fwd_context
    
    def get_ego_pose(self, cam_sample):
        """
        Get the absolute ego pose (world-to-ego transformation) for a frame
        Returns a 4x4 transformation matrix
        """
        ego_pose = self.dataset.get('ego_pose', cam_sample['ego_pose_token'])
        ego_rotation = Quaternion(ego_pose['rotation'])
        ego_translation = np.array(ego_pose['translation'])[:, None]

        # ego-to-world transformation
        ego_to_world = np.vstack([
            np.hstack((ego_rotation.rotation_matrix, ego_translation)),
            np.array([0, 0, 0, 1])
        ])

        return ego_to_world.astype(np.float64)

    @staticmethod
    def _to_numpy(x):
        """Convert tensor to numpy array if needed"""
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

    def compute_frame_transforms(self, ego_pose_0, ego_pose_N, c2e_extr):
        """
        Compute relative transformations from frame 0 to frame N
        Args:
            ego_pose_0: ego-to-world transformation for frame 0 [4, 4]
            ego_pose_N: ego-to-world transformation for frame N [4, 4]
            c2e_extr: camera-to-ego extrinsics [4, 4]
        Returns:
            ego_T_ego: transformation from ego_0 to ego_N [4, 4]
            cam_T_cam: transformation from cam_0 to cam_N [4, 4]
        """
        # Convert to numpy if needed
        ego_pose_0 = self._to_numpy(ego_pose_0)
        ego_pose_N = self._to_numpy(ego_pose_N)
        c2e_extr = self._to_numpy(c2e_extr)

        # Compute world_to_ego transformations
        world_to_ego_0 = np.linalg.inv(ego_pose_0)
        world_to_ego_N = np.linalg.inv(ego_pose_N)

        # Compute transformation through camera chain: ego_0 -> cam_0 -> cam_N -> ego_N
        e2c_extr = np.linalg.inv(c2e_extr)
        cam_T_cam = e2c_extr @ world_to_ego_N @ ego_pose_0 @ c2e_extr
        ego_T_ego = c2e_extr @ cam_T_cam @ e2c_extr

        # Ensure valid homogeneous transformation matrix
        ego_T_ego[3, :] = [0, 0, 0, 1]
        cam_T_cam[3, :] = [0, 0, 0, 1]

        return ego_T_ego.astype(np.float64), cam_T_cam.astype(np.float64)

    def get_vehicle_annotations(self, sample_token, cam_name, velocity_target_token=None, velocity_source_token=None):
        """
        Get vehicle annotations for a specific sample and camera
        Returns list of vehicles with 2D boxes, 3D boxes, tracking IDs, and velocities
        Args:
            sample_token: Token for the current frame (for getting box positions)
            cam_name: Camera name
            velocity_target_token: Token for the target frame to calculate velocity (default: next frame)
            velocity_source_token: Token for the source frame to calculate velocity (default: sample_token)
                                   This allows calculating velocity between two arbitrary frames
        """
        sample = self.dataset.get('sample', sample_token)
        cam_data = self.dataset.get('sample_data', sample['data'][cam_name])
        
        # Get calibration data
        sensor = self.dataset.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        intrinsic = np.array(sensor['camera_intrinsic'])
        
        # Get ego pose
        ego_pose = self.dataset.get('ego_pose', cam_data['ego_pose_token'])
        
        # Vehicle categories to include
        vehicle_categories = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle']
        
        vehicles = []
        for ann_token in sample['anns']:
            ann = self.dataset.get('sample_annotation', ann_token)
            
            # Check if this is a vehicle
            category = ann['category_name']
            if not any(veh in category for veh in vehicle_categories):
                continue
            
            # Get instance for tracking
            instance = self.dataset.get('instance', ann['instance_token'])
            
            # Create 3D box in global coordinates
            box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']),
                     name=ann['category_name'], token=ann['token'])
            
            # Transform to ego vehicle coordinates
            box.translate(-np.array(ego_pose['translation']))
            box.rotate(Quaternion(ego_pose['rotation']).inverse)
            
            # Transform to camera coordinates
            box.translate(-np.array(sensor['translation']))
            box.rotate(Quaternion(sensor['rotation']).inverse)
            
            # Project 3D box to 2D
            corners_3d = box.corners()
            corners_2d = view_points(corners_3d, intrinsic, normalize=True)[:2, :]
            
            # Get 2D bounding box from projected corners
            x_min = np.min(corners_2d[0, :])
            x_max = np.max(corners_2d[0, :])
            y_min = np.min(corners_2d[1, :])
            y_max = np.max(corners_2d[1, :])
            
            # Get image dimensions (NuScenes standard resolution)
            img_width = 1600  # Original NuScenes width
            img_height = 900  # Original NuScenes height
            
            # Clip to image bounds
            x_min_clipped = max(0, min(x_min, img_width))
            x_max_clipped = max(0, min(x_max, img_width))
            y_min_clipped = max(0, min(y_min, img_height))
            y_max_clipped = max(0, min(y_max, img_height))
            
            # Check if box is visible in camera (in front and within image bounds after clipping)
            if np.all(corners_3d[2, :] > 0) and x_max_clipped > x_min_clipped and y_max_clipped > y_min_clipped:  # Valid box
                # Calculate velocity in ego frame (without ego motion)
                # This computes: velocity_ego = (pos_target_in_ego_t - pos_t_in_ego_t) / time_delta
                # Use velocity_source_token if provided, otherwise use sample_token
                source_token = velocity_source_token if velocity_source_token is not None else sample_token
                velocity_ego = self.calculate_vehicle_velocity_from_tracking(
                    ann['instance_token'], source_token, cam_name, velocity_target_token
                )
                
                # Scale bbox from original NuScenes resolution to model resolution
                # Original: 1600x900, Model: 518x280
                # Note: The actual scaling might be different if aspect ratio is not preserved
                scale_x = 518.0 / 1600.0  # Use fixed original dimensions
                scale_y = 280.0 / 900.0    # Use fixed original dimensions
                
                # No offset - removed to fix flow alignment
                offset_x = 0  # No horizontal shift
                offset_y = 0  # No vertical shift
                
                bbox_2d_scaled = [
                    x_min_clipped * scale_x + offset_x,
                    y_min_clipped * scale_y + offset_y,
                    x_max_clipped * scale_x + offset_x,
                    y_max_clipped * scale_y + offset_y
                ]
                
                vehicles.append({
                    'instance_token': ann['instance_token'],
                    'tracking_id': instance['token'],
                    'category': category,
                    'bbox_2d': bbox_2d_scaled,
                    # Don't include Box object as it can't be collated
                    # 'bbox_3d': box,
                    'velocity': velocity_ego.tolist() if hasattr(velocity_ego, 'tolist') else list(velocity_ego),  # velocity in ego frame
                    'center_3d': box.center.tolist() if hasattr(box.center, 'tolist') else list(box.center),
                    'depth': float(box.center[2]),  # Z coordinate as depth
                    'visibility': ann.get('visibility_token', ''),
                    'num_lidar_pts': ann.get('num_lidar_pts', 0),
                    'camera_intrinsic': intrinsic.tolist(),  # Add camera intrinsics for accurate projection
                })
        
        return vehicles
    
    def calculate_vehicle_velocity_from_tracking(self, instance_token, sample_token, cam_name, next_sample_token=None):
        """
        Calculate vehicle velocity in ego frame at time t (without ego motion)
        """
        if next_sample_token is None:
            sample = self.dataset.get('sample', sample_token)
            next_sample_token = sample.get('next', None)
            if next_sample_token is None:
                return np.array([0, 0, 0])
        
        # Get current and next annotations for this instance
        current_ann = None
        next_ann = None
        
        sample = self.dataset.get('sample', sample_token)
        for ann_token in sample['anns']:
            ann = self.dataset.get('sample_annotation', ann_token)
            if ann['instance_token'] == instance_token:
                current_ann = ann
                break
        
        if next_sample_token:
            next_sample = self.dataset.get('sample', next_sample_token)
            for ann_token in next_sample['anns']:
                ann = self.dataset.get('sample_annotation', ann_token)
                if ann['instance_token'] == instance_token:
                    next_ann = ann
                    break
        
        if current_ann is None or next_ann is None:
            return np.array([0, 0, 0])
        
        # Get ego poses for both frames
        cam_data_t = self.dataset.get('sample_data', sample['data'][cam_name])
        ego_pose_t = self.dataset.get('ego_pose', cam_data_t['ego_pose_token'])
        
        cam_data_t1 = self.dataset.get('sample_data', next_sample['data'][cam_name])
        ego_pose_t1 = self.dataset.get('ego_pose', cam_data_t1['ego_pose_token'])
        
        # Transform both positions to ego frame at time t
        # Position at time t in ego_t frame
        pos_world_t = np.array(current_ann['translation'])
        pos_ego_t = pos_world_t - np.array(ego_pose_t['translation'])
        pos_ego_t = Quaternion(ego_pose_t['rotation']).inverse.rotate(pos_ego_t)
        
        # Position at time t+1, but transformed to ego_t frame (not ego_t+1)
        pos_world_t1 = np.array(next_ann['translation'])
        pos_in_ego_t = pos_world_t1 - np.array(ego_pose_t['translation'])  # Use ego_t, not ego_t1
        pos_in_ego_t = Quaternion(ego_pose_t['rotation']).inverse.rotate(pos_in_ego_t)
        
        # Calculate displacement in ego_t frame
        displacement_ego = pos_in_ego_t - pos_ego_t
        
        # Get time delta - for 12Hz data with context_span frames: context_span / 12 Hz
        if hasattr(self, 'version') and '12Hz' in self.version:
            time_delta = self.context_span / 12.0  # 12Hz sampling rate
        else:
            time_delta = 0.5  # Default fallback for other versions
        
        # Calculate velocity in ego frame
        velocity_ego = displacement_ego / time_delta
        
        return velocity_ego
    
    def generate_depth_map(self, sample, sensor, cam_sample):
        """
        This function returns depth map for nuscenes dataset,
        result of depth map is saved in nuscenes/samples/DEPTH_MAP
        """        
        # generate depth filename
        filename = '{}/{}.npz'.format(
                        os.path.join(os.path.dirname(self.cache_dir), 'samples'),
                        'DEPTH_MAP/{}/{}'.format(sensor, cam_sample['filename']))
        
        load_flag = False
        # load and return if exists
        if os.path.exists(filename):
            try:
                depth = np.load(filename, allow_pickle=True)['depth']
                load_flag = True
            except:
                load_flag = False
            
        if not load_flag:
            lidar_sample = self.dataset.get(
                'sample_data', sample['data']['LIDAR_TOP'])

            # lidar points                
            lidar_file = os.path.join(
                self.path, lidar_sample['filename'])
            lidar_points = np.fromfile(lidar_file, dtype=np.float32)
            lidar_points = lidar_points.reshape(-1, 5)[:, :3]

            # lidar -> world
            lidar_pose = self.dataset.get(
                'ego_pose', lidar_sample['ego_pose_token'])
            lidar_rotation= Quaternion(lidar_pose['rotation'])
            lidar_translation = np.array(lidar_pose['translation'])[:, None]
            lidar_to_world = np.vstack([
                np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
                np.array([0, 0, 0, 1])
            ])

            # lidar -> ego
            sensor_sample = self.dataset.get(
                'calibrated_sensor', lidar_sample['calibrated_sensor_token'])
            lidar_to_ego_rotation = Quaternion(
                sensor_sample['rotation']).rotation_matrix
            lidar_to_ego_translation = np.array(
                sensor_sample['translation']).reshape(1, 3)

            ego_lidar_points = np.dot(
                lidar_points[:, :3], lidar_to_ego_rotation.T)
            ego_lidar_points += lidar_to_ego_translation

            homo_ego_lidar_points = np.concatenate(
                (ego_lidar_points, np.ones((ego_lidar_points.shape[0], 1))), axis=1)


            # world -> ego
            ego_pose = self.dataset.get(
                    'ego_pose', cam_sample['ego_pose_token'])
            ego_rotation = Quaternion(ego_pose['rotation']).inverse
            ego_translation = - np.array(ego_pose['translation'])[:, None]
            world_to_ego = np.vstack([
                    np.hstack((ego_rotation.rotation_matrix,
                               ego_rotation.rotation_matrix @ ego_translation)),
                    np.array([0, 0, 0, 1])
                    ])

            # Ego -> sensor
            sensor_sample = self.dataset.get(
                'calibrated_sensor', cam_sample['calibrated_sensor_token'])
            sensor_rotation = Quaternion(sensor_sample['rotation'])
            sensor_translation = np.array(
                sensor_sample['translation'])[:, None]
            sensor_to_ego = np.vstack([
                np.hstack((sensor_rotation.rotation_matrix, 
                           sensor_translation)),
                np.array([0, 0, 0, 1])
               ])
            ego_to_sensor = np.linalg.inv(sensor_to_ego)
            
            # lidar -> sensor
            lidar_to_sensor = ego_to_sensor @ world_to_ego @ lidar_to_world
            homo_ego_lidar_points = torch.from_numpy(homo_ego_lidar_points).float()
            cam_lidar_points = np.matmul(lidar_to_sensor, homo_ego_lidar_points.T).T

            # depth > 0
            depth_mask = cam_lidar_points[:, 2] > 0
            cam_lidar_points = cam_lidar_points[depth_mask]

            # sensor -> image
            intrinsics = np.eye(4)
            intrinsics[:3, :3] = sensor_sample['camera_intrinsic']
            pixel_points = np.matmul(intrinsics, cam_lidar_points.T).T
            pixel_points[:, :2] /= pixel_points[:, 2:3]
            
            # load image for pixel range
            image_filename = os.path.join(
                self.path, cam_sample['filename'])
            img = pil.open(image_filename)
            h, w, _ = np.array(img).shape
            
            # mask points in pixel range
            pixel_mask = (pixel_points[:, 0] >= 0) & (pixel_points[:, 0] <= w-1)\
                        & (pixel_points[:,1] >= 0) & (pixel_points[:,1] <= h-1)
            valid_points = pixel_points[pixel_mask].round().int()
            valid_depth = cam_lidar_points[:, 2][pixel_mask]
        
            depth = np.zeros([h, w])
            depth[valid_points[:, 1], valid_points[:,0]] = valid_depth
        
            # save depth map
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            np.savez_compressed(filename, depth=depth)
        return depth

    def get_tranformation_mat(self, pose):
        """
        This function transforms pose information in accordance with DDAD dataset format
        """
        extrinsics = Quaternion(pose['rotation']).transformation_matrix
        extrinsics[:3, 3] = np.array(pose['translation'])
        return extrinsics.astype(np.float32)

    def get_num_scenes(self):
        """Return the number of scenes"""
        return len(self.scenes_data)
    
    def get_scene_length(self, scene_idx):
        """Return the number of samples in a specific scene"""
        return len(self.scenes_data[scene_idx])
    
    def get_scene_name(self, scene_idx):
        """Return the name of a specific scene"""
        return self.scene_names[scene_idx]
    
    def get_scene_token(self, scene_idx):
        """Return the token of a specific scene"""
        return self.scene_tokens[scene_idx]

    def get_scene_sample(self, scene_idx, sample_idx):
        """
        Get a specific sample from a specific scene
        Args:
            scene_idx: Index of the scene
            sample_idx: Index of the sample within the scene
        Returns:
            Processed sample data
        """
        if scene_idx >= len(self.scenes_data):
            raise IndexError(f"Scene index {scene_idx} out of range")
        if sample_idx >= len(self.scenes_data[scene_idx]):
            raise IndexError(f"Sample index {sample_idx} out of range for scene {scene_idx}")
        
        frame_idx = self.scenes_data[scene_idx][sample_idx]
        sample_nusc = self.dataset.get('sample', frame_idx)
        scene_token = sample_nusc['scene_token']
        
        sample = []
        contexts = []
        if self.bwd:
            contexts.append(-1)
        if self.fwd:
            contexts.append(1)

        # loop over all cameras            
        for cam in self.cameras:
            cam_sample = self.dataset.get(
                'sample_data', sample_nusc['data'][cam])

            data = {
                'idx': f"{scene_idx}_{sample_idx}",  # Combined index for tracking
                'scene_idx': scene_idx,
                'sample_idx': sample_idx,
                'token': frame_idx,
                'scene_token': scene_token,
                'scene_name': self.scene_names[scene_idx],
                'scene_idx': scene_idx,
                'sensor_name': cam,
                'contexts': contexts,
                'filename': cam_sample['filename'],
                'rgb': self.get_current('rgb', cam_sample),
                'intrinsics': self.get_current('intrinsics', cam_sample),
                'timestamp': cam_sample['timestamp']  # Add timestamp in microseconds
            }

            if self.with_depth:
                data.update({
                    'gt_depth': self.generate_depth_map(sample_nusc, cam, cam_sample)
                })
            if self.with_pose:
                data.update({
                    'extrinsics': self.get_current('extrinsics', cam_sample)
                })
            if self.with_ego_pose:
                data.update({
                    'ego_pose': self.get_ego_pose(cam_sample)  # Direct ego pose
                })
            if self.with_mask:
                data.update({
                    'mask': self.mask_loader(self.mask_path, '', cam)
                })        
            if self.has_context:
                rgb_context = self.get_context('rgb', cam_sample)
                # Only add rgb_context if it's not empty
                if rgb_context:
                    data.update({
                        'rgb_context': rgb_context
                    })

            sample.append(data)

        # apply same data transformations for all sensors
        if self.data_transform:
            sample = [self.data_transform(smp,) for smp in sample]

        # stack and align dataset for our trainer
        sample = stack_sample(sample)
        sample = align_dataset(sample, contexts)
        return sample

    def get_scene_samples(self, scene_idx):
        """
        Get all samples from a specific scene
        Args:
            scene_idx: Index of the scene
        Returns:
            List of processed sample data for the entire scene
        """
        scene_samples = []
        scene_length = self.get_scene_length(scene_idx)
        
        for sample_idx in range(scene_length):
            sample = self.get_scene_sample(scene_idx, sample_idx)
            scene_samples.append(sample)
        
        return scene_samples

    def __len__(self):
        """Total number of samples across all scenes (for standard Dataset compatibility)"""
        return sum(len(scene)//self.sample_rate for scene in self.scenes_data)

    def get_frame(self, idx, frame_idx, scene_token="", scene_name="", scene_idx="", is_key_frame=True, velocity_target_frame_idx=None, velocity_source_frame_idx=None):
        sample_nusc = self.dataset.get('sample', frame_idx)
        sample = []
        contexts = []
        if self.bwd:
            contexts.append(-1)
        if self.fwd:
            contexts.append(1)

        # loop over all cameras
        for cam in self.cameras:
            cam_sample = self.dataset.get(
                'sample_data', sample_nusc['data'][cam])

            data = {
                'idx': idx,
                'token': frame_idx,
                'scene_token': scene_token,
                'scene_name': scene_name,
                'scene_idx': scene_idx,
                'sensor_name': cam,
                'contexts': contexts,
                'filename': cam_sample['filename'],
                'rgb': self.get_current('rgb', cam_sample),
                'intrinsics': self.get_current('intrinsics', cam_sample),
                'timestamp': cam_sample['timestamp']  # Add timestamp in microseconds
            }

            # if depth is returned
            if self.with_depth:
                data.update({
                    'gt_depth': self.generate_depth_map(sample_nusc, cam, cam_sample)
                })
            # if pose is returned
            if self.with_pose:
                data.update({
                    'extrinsics':self.get_current('extrinsics', cam_sample)
                })
            # if ego_pose is returned
            if self.with_ego_pose:
                data.update({
                    'ego_pose': self.get_ego_pose(cam_sample)  # Direct ego pose
                })
            # if mask is returned
            if self.with_mask:
                data.update({
                    'mask': self.mask_loader(self.mask_path, '', cam)
                })

            if is_key_frame:
                # Add vehicle annotations (always add, even if empty)
                vehicle_annotations = self.get_vehicle_annotations(
                    frame_idx, cam, velocity_target_frame_idx, velocity_source_frame_idx
                )
                data.update({
                    'vehicle_annotations': vehicle_annotations if vehicle_annotations else []
                })
            
            # if context is returned
            if self.has_context:
                rgb_context = self.get_context('rgb', cam_sample)
                # Only add rgb_context if it's not empty
                if rgb_context:
                    data.update({
                        'rgb_context': rgb_context
                    })

            sample.append(data)


        # apply same data transformations for all sensors
        if self.data_transform:
            sample = [self.data_transform(smp,) for smp in sample]
        
        # stack and align dataset for our trainer
        # n, batch_size, h, w, c
        sample = stack_sample(sample)

        return sample

    def __getitem__(
        self, sample_idx: int, scene_idx: int = -1
    ) -> Dict[str, Any]:
        idx = sample_idx
        frame_idx = self.scenes_data[scene_idx][sample_idx]
        scene_token = self.scene_tokens[scene_idx]
        scene_name = self.scene_names[scene_idx]

        source_sample = self.dataset.get('sample', frame_idx)

        # Navigate context_span frames forward to get the second context frame
        source_frame_idx = None
        if source_sample:
            current_sample = source_sample
            for i in range(self.context_span):
                if current_sample and current_sample.get('next'):
                    current_sample = self.dataset.get('sample', current_sample['next'])
                else:
                    current_sample = None
                    break
            if current_sample:
                source_frame_idx = current_sample['token']

        context_dict_list = []
        target_dict_list = []

        # Get first context frame (frame 0)
        # Calculate velocity from frame 0 to frame context_span
        cur_sample = self.get_frame(
            idx=idx,
            frame_idx=frame_idx,
            scene_token=scene_token,
            scene_name=scene_name,
            scene_idx=scene_idx,
            velocity_source_frame_idx=frame_idx,  # Velocity source: frame 0
            velocity_target_frame_idx=source_frame_idx  # Velocity target: frame context_span
        )
        cur_sample = align_dataset(cur_sample)

        # Get intermediate frames (frames 1 to context_span-1)
        current_sample = self.dataset.get('sample', frame_idx)
        for offset in range(1, self.context_span):
            if current_sample and current_sample.get('next'):
                next_token = current_sample['next']
                current_sample = self.dataset.get('sample', next_token)
                intermediate_frame_token = current_sample['token']

                intermediate_dict = self.get_frame(
                    idx=idx,
                    frame_idx=intermediate_frame_token,
                    scene_token=scene_token,
                    scene_name=scene_name,
                    scene_idx=scene_idx,
                    is_key_frame=False
                )
                intermediate_dict = align_dataset(intermediate_dict)
                target_dict_list.append(intermediate_dict)
            else:
                print(f"WARNING: Could not navigate to intermediate frame offset={offset}")
                break

        # Get second context frame (frame context_span)
        # Use the same velocity calculation as frame 0 (from frame 0 to frame context_span)
        if source_frame_idx:
            context_dict = self.get_frame(
                idx=idx+1,
                frame_idx=source_frame_idx,
                scene_token=scene_token,
                scene_name=scene_name,
                scene_idx=scene_idx,
                is_key_frame=True,  # Ensure vehicle annotations are added
                velocity_source_frame_idx=frame_idx,  # Velocity source: frame 0
                velocity_target_frame_idx=source_frame_idx  # Velocity target: frame context_span
            )
            context_dict = align_dataset(context_dict)
            context_dict_list.append(context_dict)
        
        if target_dict_list:
            target_dict = default_collate(target_dict_list)
            for k, v in target_dict.items():
                if isinstance(v, torch.Tensor) and len(v.shape) >= 2:
                    target_dict[k] = torch.cat([d for d in v], dim=0)
        else:
            target_dict = {}

        if context_dict_list:
            # Extract vehicle annotations from both context frames
            vehicle_anns_frame_0 = cur_sample.pop('vehicle_annotations', None)
            vehicle_anns_frame_context_span = [ctx.pop('vehicle_annotations', None) for ctx in context_dict_list]

            cur_sample['idx'] = 0
            for i, target_dict in enumerate(target_dict_list):
                target_dict['idx'] = i + 1
            for i, context_dict in enumerate(context_dict_list):
                context_dict['idx'] = i + self.context_span

            # Compute ego_T_ego and cam_T_cam transformations from frame 0 to all other frames
            # Compute for all cameras, not just camera 0
            if self.with_ego_pose and 'ego_pose' in cur_sample and 'c2e_extr' in cur_sample:
                num_cameras = cur_sample['ego_pose'].shape[0]

                # Compute for intermediate frames (1 to context_span-1)
                for i, target_dict in enumerate(target_dict_list):
                    ego_T_ego_list = []
                    cam_T_cam_list = []
                    for cam_idx in range(num_cameras):
                        ego_pose_0 = cur_sample['ego_pose'][cam_idx]
                        ego_pose_N = target_dict['ego_pose'][cam_idx]
                        c2e_extr = cur_sample['c2e_extr'][cam_idx]

                        ego_T_ego, cam_T_cam = self.compute_frame_transforms(ego_pose_0, ego_pose_N, c2e_extr)
                        ego_T_ego_list.append(torch.from_numpy(ego_T_ego).float())
                        cam_T_cam_list.append(torch.from_numpy(cam_T_cam).float())

                    target_dict[('ego_T_ego', 0, i + 1)] = torch.stack(ego_T_ego_list, dim=0)
                    target_dict[('cam_T_cam', 0, i + 1)] = torch.stack(cam_T_cam_list, dim=0)

                # Compute for the final context frame (frame context_span)
                for context_dict in context_dict_list:
                    ego_T_ego_list = []
                    cam_T_cam_list = []
                    for cam_idx in range(num_cameras):
                        ego_pose_0 = cur_sample['ego_pose'][cam_idx]
                        ego_pose_N = context_dict['ego_pose'][cam_idx]
                        c2e_extr = cur_sample['c2e_extr'][cam_idx]

                        ego_T_ego, cam_T_cam = self.compute_frame_transforms(ego_pose_0, ego_pose_N, c2e_extr)
                        ego_T_ego_list.append(torch.from_numpy(ego_T_ego).float())
                        cam_T_cam_list.append(torch.from_numpy(cam_T_cam).float())

                    context_dict[('ego_T_ego', 0, self.context_span)] = torch.stack(ego_T_ego_list, dim=0)
                    context_dict[('cam_T_cam', 0, self.context_span)] = torch.stack(cam_T_cam_list, dim=0)

            all_context_dict = default_collate([cur_sample] + context_dict_list)
            all_dict = default_collate([cur_sample] + target_dict_list + context_dict_list)

            # Concatenate tensors along batch dimension
            for k, v in all_dict.items():
                if isinstance(v, torch.Tensor) and len(v.shape) >= 2:
                    all_dict[k] = torch.cat([d for d in v], dim=0)
            for k, v in all_context_dict.items():
                if isinstance(v, torch.Tensor) and len(v.shape) >= 2:
                    all_context_dict[k] = torch.cat([d for d in v], dim=0)

            # Re-add tuple keys lost by default_collate (will get batch dim from custom_collate_fn)
            for i, target_dict in enumerate(target_dict_list):
                ego_key = ('ego_T_ego', 0, i + 1)
                cam_key = ('cam_T_cam', 0, i + 1)
                if ego_key in target_dict:
                    all_dict[ego_key] = target_dict[ego_key]
                if cam_key in target_dict:
                    all_dict[cam_key] = target_dict[cam_key]

            if context_dict_list:
                ego_key = ('ego_T_ego', 0, self.context_span)
                cam_key = ('cam_T_cam', 0, self.context_span)
                if ego_key in context_dict_list[0]:
                    all_context_dict[ego_key] = all_dict[ego_key] = context_dict_list[0][ego_key]
                if cam_key in context_dict_list[0]:
                    all_context_dict[cam_key] = all_dict[cam_key] = context_dict_list[0][cam_key]

            # Store vehicle annotations separately for each context frame
            # This allows generating masks for frame 0 and frame context_span independently
            if vehicle_anns_frame_0 is not None:
                all_dict['vehicle_annotations_frame_0'] = vehicle_anns_frame_0
                all_context_dict['vehicle_annotations_frame_0'] = vehicle_anns_frame_0

            if vehicle_anns_frame_context_span and vehicle_anns_frame_context_span[0] is not None:
                all_dict[f'vehicle_annotations_frame_{self.context_span}'] = vehicle_anns_frame_context_span[0]
                all_context_dict[f'vehicle_annotations_frame_{self.context_span}'] = vehicle_anns_frame_context_span[0]

            # Also keep combined annotations for backward compatibility (optional)
            combined_vehicle_anns = []
            if vehicle_anns_frame_0 is not None:
                combined_vehicle_anns.extend(vehicle_anns_frame_0)
            if vehicle_anns_frame_context_span and vehicle_anns_frame_context_span[0] is not None:
                combined_vehicle_anns.extend(vehicle_anns_frame_context_span[0])

            if combined_vehicle_anns:
                all_dict['vehicle_annotations'] = combined_vehicle_anns
                all_context_dict['vehicle_annotations'] = combined_vehicle_anns

            context_dict = default_collate(context_dict_list)
            for k, v in context_dict.items():
                if isinstance(v, torch.Tensor) and len(v.shape) >= 2:
                    context_dict[k] = torch.cat([d for d in v], dim=0)
        else:
            all_context_dict = {}
            all_dict = {}
            for k, v in cur_sample.items():
                if isinstance(v, torch.Tensor):
                    all_dict[k] = v.unsqueeze(0)
                elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    all_dict[k] = [item.unsqueeze(0) for item in v]
                else:
                    all_dict[k] = v
            for k, v in cur_sample.items():
                if isinstance(v, torch.Tensor):
                    all_context_dict[k] = v.unsqueeze(0)
                elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    all_context_dict[k] = [item.unsqueeze(0) for item in v]
                else:
                    all_context_dict[k] = v
            if 'vehicle_annotations' in cur_sample:
                all_dict['vehicle_annotations'] = cur_sample['vehicle_annotations']
                all_context_dict['vehicle_annotations'] = cur_sample['vehicle_annotations']

        ret_sample = {
            'cur_sample': cur_sample,
            'context_frames': all_context_dict,
            'target_frames': target_dict,
            'all_dict': all_dict
        }
        return ret_sample

def custom_collate_fn(batch):
    """
    Custom collate function that handles vehicle_annotations and preserves tuple keys
    """
    elem = batch[0]
    if not isinstance(elem, dict):
        return default_collate(batch)

    result = {}
    for key in elem:
        values = [d[key] for d in batch]

        # Handle tuple keys first (e.g., ('ego_T_ego', 0, frame_id))
        if isinstance(key, tuple):
            try:
                result[key] = torch.stack(values, dim=0)
            except:
                result[key] = values
        # Keep vehicle annotations as list to preserve batch dimension
        elif isinstance(key, str) and (key == 'vehicle_annotations' or key.startswith('vehicle_annotations_frame_')):
            result[key] = values
        elif key in ['all_dict', 'context_frames']:
            result[key] = custom_collate_fn(values)
        else:
            try:
                result[key] = default_collate(values)
            except:
                result[key] = values

    return result

if __name__ == '__main__':
    
    # 
    print('test')