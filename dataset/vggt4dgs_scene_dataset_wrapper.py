#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#

"""
Scene-based dataset wrapper for proper scene-level batching during inference.
This wrapper groups samples by scene and provides scene-based iteration.

IMPORTANT: This file is essential for inference to work correctly!
It fixes the scene batching issue where inference was processing 200+ individual
samples instead of ~40 temporal windows for a 20s scene.

Key features:
1. Groups samples by scene instead of mixing them
2. Adds batch dimension to tensors for model compatibility (4D -> 5D)
3. Yields temporal windows correctly (~40 windows per 20s scene at 12Hz with context_span=6)

Used by: vggt4dgs_data_module.py -> test_dataloader(scene_based=True)
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional


class SceneBasedDataset(Dataset):
    """
    Wrapper dataset that groups samples by scene for proper scene-level processing.
    Each item returned represents all samples from a single scene.
    """

    def __init__(self, base_dataset):
        """
        Initialize the scene-based dataset wrapper.

        Args:
            base_dataset: The base NuScenesdataset4D instance
        """
        self.base_dataset = base_dataset

        # Build scene-to-samples mapping
        self._build_scene_mapping()

    def _build_scene_mapping(self):
        """Build mapping of scene indices to their sample indices"""
        self.scene_samples = []  # List of lists, each containing sample indices for a scene
        self.scene_names = []
        self.scene_tokens = []

        # Get scene information from base dataset
        if hasattr(self.base_dataset, 'scenes_data') and hasattr(self.base_dataset, 'sample_tokens'):
            # For datasets that already have scene grouping
            self.scene_names = self.base_dataset.scene_names
            self.scene_tokens = self.base_dataset.scene_tokens

            # Build a mapping from sample token to index in sample_tokens
            token_to_idx = {token: idx for idx, token in enumerate(self.base_dataset.sample_tokens)}

            # Build sample indices for each scene
            for scene_idx, scene_data in enumerate(self.base_dataset.scenes_data):
                scene_sample_indices = []
                # Only add indices for tokens that exist in sample_tokens
                # (For 12Hz data, only valid start frames are in sample_tokens)
                for token in scene_data:
                    if token in token_to_idx:
                        scene_sample_indices.append(token_to_idx[token])

                self.scene_samples.append(scene_sample_indices)
        else:
            raise ValueError("Base dataset doesn't have scene information or sample_tokens")

        print(f"SceneBasedDataset initialized with {len(self.scene_samples)} scenes")
        total_samples = sum(len(samples) for samples in self.scene_samples)
        print(f"Total samples across all scenes: {total_samples}")
        for i, (name, samples) in enumerate(zip(self.scene_names, self.scene_samples)):
            if len(samples) > 0:
                print(f"  Scene {i}: {name} - {len(samples)} temporal windows")

    def __len__(self):
        """Return number of scenes"""
        return len(self.scene_samples)

    def __getitem__(self, scene_idx: int) -> Dict[str, Any]:
        """
        Get all samples from a specific scene.

        Args:
            scene_idx: Index of the scene

        Returns:
            Dictionary containing:
                - scene_name: Name of the scene
                - scene_token: Token of the scene
                - scene_idx: Index of the scene
                - samples: List of all samples in the scene
        """
        if scene_idx >= len(self.scene_samples):
            raise IndexError(f"Scene index {scene_idx} out of range")

        scene_name = self.scene_names[scene_idx]
        scene_token = self.scene_tokens[scene_idx]
        sample_indices = self.scene_samples[scene_idx]

        # Collect all samples for this scene
        scene_samples = []
        for sample_idx in sample_indices:
            sample = self.base_dataset[sample_idx]
            scene_samples.append(sample)

        return {
            'scene_name': scene_name,
            'scene_token': scene_token,
            'scene_idx': scene_idx,
            'num_samples': len(scene_samples),
            'samples': scene_samples
        }

    def get_scene_info(self, scene_idx: int) -> Dict[str, Any]:
        """Get information about a scene without loading samples"""
        if scene_idx >= len(self.scene_samples):
            raise IndexError(f"Scene index {scene_idx} out of range")

        return {
            'scene_name': self.scene_names[scene_idx],
            'scene_token': self.scene_tokens[scene_idx],
            'scene_idx': scene_idx,
            'num_samples': len(self.scene_samples[scene_idx])
        }


class SceneBatchDataLoader:
    """
    Custom DataLoader for scene-based batching.
    Processes one scene at a time, yielding properly grouped temporal windows.
    """

    def __init__(self, scene_dataset, batch_size=1, shuffle=False, context_span=6):
        """
        Initialize the scene batch data loader.

        Args:
            scene_dataset: SceneBasedDataset instance
            batch_size: Number of temporal windows per batch (usually 1 for inference)
            shuffle: Whether to shuffle scenes (not samples within scenes)
            context_span: Number of frames in each temporal window (default: 6)
        """
        self.scene_dataset = scene_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.context_span = context_span

        # Scene order
        self.scene_order = list(range(len(self.scene_dataset)))
        if self.shuffle:
            import random
            random.shuffle(self.scene_order)

    def __len__(self):
        """Return total number of temporal windows across all scenes"""
        total_windows = 0
        for scene_idx in range(len(self.scene_dataset)):
            scene_info = self.scene_dataset.get_scene_info(scene_idx)
            # Each sample from the base dataset is already a complete temporal window
            # So the number of windows equals the number of samples
            num_windows = scene_info['num_samples']
            total_windows += num_windows
        return total_windows

    def _add_batch_dimension(self, sample):
        """Add batch dimension to all tensors in the sample"""
        import torch

        def add_batch_to_tensor(tensor):
            """Add batch dimension to a single tensor if needed"""
            if tensor.dim() == 4:  # (s, c, h, w) -> (b, s, c, h, w)
                return tensor.unsqueeze(0)
            elif tensor.dim() == 3:  # (s, h, w) or (c, h, w) -> add batch
                return tensor.unsqueeze(0)
            elif tensor.dim() == 2:  # (s, f) or similar -> add batch
                return tensor.unsqueeze(0)
            else:
                # Keep 1D tensors and scalars as-is
                return tensor

        batched = {}
        for key, value in sample.items():
            if isinstance(value, dict):
                # Recursively handle nested dictionaries
                batched[key] = self._add_batch_dimension(value)
            elif isinstance(value, torch.Tensor):
                # Add batch dimension to tensors
                batched[key] = add_batch_to_tensor(value)
            elif isinstance(value, list):
                # Add batch dimension to vehicle annotations
                if key == 'vehicle_annotations' or key.startswith('vehicle_annotations_frame_'):
                    batched[key] = [value]
                else:
                    batched[key] = value
            else:
                # Keep other types as-is
                batched[key] = value
        return batched

    def __iter__(self):
        """Iterate over all scenes, yielding samples that already contain temporal windows"""
        for scene_idx in self.scene_order:
            # Get all samples for this scene
            scene_data = self.scene_dataset[scene_idx]
            scene_name = scene_data['scene_name']
            scene_token = scene_data['scene_token']
            samples = scene_data['samples']

            print(f"\nProcessing Scene {scene_idx}: {scene_name} with {len(samples)} temporal windows")
            print(f"  Each window already contains {self.context_span} frames")

            # The base dataset already groups frames into temporal windows
            # Each sample is a complete temporal window with context_span frames
            # We need to add a batch dimension for compatibility with the model
            for window_idx, sample in enumerate(samples):
                # IMPORTANT: Update scene metadata for each sample to ensure correct scene tracking
                # The sample might have outdated scene info from the base dataset
                if 'context_frames' in sample:
                    sample['context_frames']['scene_name'] = scene_name
                    sample['context_frames']['scene_token'] = scene_token
                    sample['context_frames']['scene_idx'] = scene_idx
                if 'all_dict' in sample:
                    sample['all_dict']['scene_name'] = scene_name
                    sample['all_dict']['scene_token'] = scene_token
                    sample['all_dict']['scene_idx'] = scene_idx
                if 'cur_sample' in sample:
                    sample['cur_sample']['scene_name'] = scene_name
                    sample['cur_sample']['scene_token'] = scene_token
                    sample['cur_sample']['scene_idx'] = scene_idx

                # Add batch dimension to all tensors in the sample
                batched_sample = self._add_batch_dimension(sample)
                yield batched_sample

                # Debug print to track progress
                if window_idx % 10 == 0 and window_idx > 0:
                    print(f"    Processed {window_idx}/{len(samples)} windows")