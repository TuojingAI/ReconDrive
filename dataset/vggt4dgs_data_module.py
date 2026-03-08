import pytorch_lightning as pl
from torch.utils.data import DataLoader
from functools import partial


from dataset.data_util import train_transforms
from dataset.vggt3dgs_dataset import NuScenesdataset
from dataset.vggt4dgs_dataset import NuScenesdataset4D, custom_collate_fn
from dataset.vggt4dgs_scene_dataset_wrapper import SceneBasedDataset, SceneBatchDataLoader


class VGGT4DGS_LITDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.read_config(cfg)
        # Store test_scenes if provided
        self.test_scenes = cfg.get('test_scenes', None)

    def read_config(self, cfg):    
        for k, v in cfg.items():
            setattr(self, k, v)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:

            self.train_dataset = self.construct_dataset('train')

            self.val_dataset = self.construct_dataset('val')


        if stage == "test" or stage is None:

            self.test_dataset = self.construct_dataset('test')

    def train_dataloader(self):
        if self.train_dataset is None:
            return None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.data_shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn,
        )
            
    
    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn,
        )
    
    def test_dataloader(self, scene_based=True):
        """
        Create test dataloader.

        Args:
            scene_based: If True, returns a scene-based dataloader that processes
                        one scene at a time. If False, returns standard dataloader.
        """
        if self.test_dataset is None:
            return None

        if scene_based:
            # Use scene-based dataset wrapper for proper scene batching
            scene_dataset = SceneBasedDataset(self.test_dataset)
            # Get context_span from config or use default
            context_span = getattr(self, 'context_span', 6)
            return SceneBatchDataLoader(
                scene_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                context_span=context_span
            )
        else:
            # Standard dataloader (original behavior)
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=custom_collate_fn,
            )
    
    def construct_dataset(self, mode):
        """
        This function constructs datasets.
        """
        # dataset arguments for the dataloader

        if hasattr(self,'crop_scale'):
            crop_scale = self.crop_scale
        else:
            crop_scale = []
        if hasattr(self,'crop_ratio'):
            crop_ratio = self.crop_ratio
        else:
            crop_ratio = []
        if hasattr(self,'crop_prob'):
            crop_prob = self.crop_prob
        else:
            crop_prob = 0.0
        if hasattr(self,'jittering'):
            jittering = self.jittering
        else:
            jittering = []
        if hasattr(self,'jittering_prob'):
            jittering_prob = self.jittering_prob
        else:
            jittering_prob = 0.0

        dataset_args = {
            'cameras': self.cameras,
            'back_context': self.back_context,
            'forward_context': self.forward_context,
            'data_transform': partial(train_transforms,
                       image_shape=(int(self.height), int(self.width)),
                       crop_scale=crop_scale if mode=='train' else [],
                       crop_ratio=crop_ratio if mode=='train' else [],
                       crop_prob=crop_prob if mode=='train' else 0.0,
                       jittering=jittering if mode=='train' else [],
                       jittering_prob=jittering_prob if mode=='train' else 0.0,),
            'depth_type': self.depth_type if 'gt_depth' in self.train_requirements else None,
            'with_pose': 'gt_pose' in self.train_requirements,
            'with_ego_pose': 'gt_ego_pose' in self.train_requirements,
            'with_mask': 'mask' in self.train_requirements,
            'cache_dir': self.cache_dir,
            'nuscenes_version': getattr(self, 'nuscenes_version', 'v1.0-trainval'),
            'context_span': getattr(self, 'context_span', 6)
        }
        stage_dict = {
            'train':'train',
            'val': 'val',
            'test': 'val'
        }
        if mode=='train':
            dataset = NuScenesdataset4D(
                self.data_path, 'train',
                **dataset_args            
            )
        elif (mode=='val') or (mode=='test'):
            dataset = NuScenesdataset4D(
                self.data_path, 'val',
                **dataset_args            
            )
        else:
            raise ValueError('Unknown mode: ' + mode)

        return dataset
    


if __name__ == '__main__':

    import PIL.Image as pil
    import yaml
    import torch
    import numpy as np
    # from external.dataset import get_transforms

    config_file = 'configs/nuscenes/vggt3dgs.yaml'
    with open(config_file) as f:
        main_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    main_datamodule = VGGT4DGS_LITDataModule(main_cfg['data_cfg'])
    main_datamodule.setup('fit')
    train_dataset = main_datamodule.train_dataset

    inputs = train_dataset[0]
    print(inputs.keys())

    # print(img_aug.shape,img.max(),img.dtype)
    for keys, value in inputs.items():
        print(keys,type(value))
        try:
            print(value.shape)
            if isinstance(value, torch.Tensor):
                print(value.amin(),value.mean(),value.amax(),value.dtype)
            elif isinstance(value, np.ndarray):
                print(value.min(),value.mean(),value.max(),value.dtype)
        except:
            print(value)

    for cam_id in range(6):
        rgb = inputs[('color_aug', 0)][cam_id]
        rgb = rgb.cpu().numpy().transpose(1, 2, 0).clip(0, 1) * 255
        pil.fromarray(rgb.astype(np.uint8)).save('rgb_aug_%d.png' % cam_id)
        rgb = inputs[('color_org', 0)][cam_id]
        rgb = rgb.cpu().numpy().transpose(1, 2, 0).clip(0, 1) * 255
        pil.fromarray(rgb.astype(np.uint8)).save('rgb_org_%d.png' % cam_id)


