#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#

import yaml
import argparse
import os
import sys
import subprocess
import torch
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger

# Add models directory to path for vggt imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "models"))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from utils.train_callback import ExportBestModelCallback, ExportMetricCallback
from dataset.vggt4dgs_data_module import VGGT4DGS_LITDataModule
from models.recondrive_model import ReconDrive_LITModelModule


from utils.snapshot import save_pipeline_snapshot, PIPELINE_DEPLOYMENT

torch.set_float32_matmul_precision('highest')


def load_and_merge_configs(main_cfg_path):
    """Load and merge main config with sub-configs"""
    with open(main_cfg_path) as f:
        main_cfg = yaml.load(f, Loader=yaml.FullLoader)

    return main_cfg


def main():
    parser = argparse.ArgumentParser(description='eval argparse')
    parser.add_argument('--cfg_path', type=str, required=True, help='Main config file path')
    parser.add_argument('--pretrained_ckpt', type=str, default='')
    parser.add_argument('--train_4d', action='store_true', help='4dgs')
    parser.add_argument('--devices', type=int, default=None, help='Number of GPUs to use (overrides config)')
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        main_cfg = yaml.load(f, Loader=yaml.FullLoader)

    main_cfg['model_cfg']['batch_size'] = main_cfg['data_cfg']['batch_size']

    # Override devices if specified via command line
    if args.devices is not None:
        main_cfg['devices'] = args.devices
        print(f"Using {args.devices} GPU(s) from command line (overriding config)")
    else:
        print(f"Using {main_cfg['devices']} GPU(s) from config file")

    save_dir = main_cfg['save_dir']

    log_dir = os.path.join(save_dir, 'log')
    ckpt_dir = os.path.join(save_dir, 'ckpt')
    code_dir = os.path.join(save_dir, 'code')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(code_dir, exist_ok=True)
    save_pipeline_snapshot(PIPELINE_DEPLOYMENT, code_dir)
    with open(os.path.join(save_dir,'cfg.yaml'),'w') as fw:
        yaml.dump(main_cfg, fw)

    pl.seed_everything(main_cfg['seed'], workers=True)

    logger = TensorBoardLogger(
        save_dir=log_dir,
        name='logs'
    )

    if args.train_4d:
        data_module = VGGT4DGS_LITDataModule(
            cfg=main_cfg['data_cfg'],
        )

    if args.pretrained_ckpt:
        litmodel = ReconDrive_LITModelModule(
            cfg=main_cfg['model_cfg'],
            save_dir=log_dir,
            logger=logger
        )
        litmodel.load_pretrained_checkpoint(args.pretrained_ckpt, strict=False, verbose=True)
    else:
        litmodel = ReconDrive_LITModelModule(
            cfg=main_cfg['model_cfg'],
            save_dir=log_dir,
            logger=logger
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='best_module',
        save_top_k=1,
        monitor="val/psnr",
        mode="max",
        save_last=True,
        every_n_epochs=1
    )

    periodic_checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='epoch_{epoch:02d}',
        save_top_k=-1,
        every_n_epochs=1,
        save_last=False,
    )

    export_metric_callback = ExportMetricCallback(
        export_dir=log_dir,
        monitor='all',
        best_metric_name='val/psnr',
        best_mode='max',
        start_after_epoch=1,
    )

    trainer = pl.Trainer(
        max_epochs=main_cfg.get('train_epoch', 50),
        accelerator="gpu",
        devices=main_cfg['devices'],
        precision="32-true",
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=8,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, periodic_checkpoint_callback, LearningRateMonitor(), export_metric_callback],
        deterministic=True,
        log_every_n_steps=100,
        enable_progress_bar=True,
        enable_model_summary=True,
        strategy='ddp_find_unused_parameters_true',
        profiler="simple",
        logger=logger
    )

    torch.use_deterministic_algorithms(mode=True,warn_only=True)
    trainer.fit(litmodel, data_module)

    data_module.setup(stage='test')

    print(f"\nTesting best model...{checkpoint_callback.best_model_path}")
    best_model = ReconDrive_LITModelModule.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(best_model, data_module)

if __name__ == "__main__":
    main()