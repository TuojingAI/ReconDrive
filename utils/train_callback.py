#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from rich.text import Text

class CustomRichProgressBar(RichProgressBar):
    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        items = [f"{k}: {v:.4f}" for k, v in metrics.items()]
        return Text("\n".join(items)), Text("")
                    

class ExportMetricCallback(pl.Callback):
    def __init__(self, export_dir, monitor='all', best_metric_name=None, best_mode='max', start_after_epoch=1):
        super().__init__()
        self.export_dir = export_dir
        self.monitor = monitor
        self.start_after_epoch = start_after_epoch
        self.best_metric_name = best_metric_name
        self.best_mode = best_mode
        self.best_score = -float('inf') if best_mode == 'max' else float('inf')
        self.best_epoch = None
        self.metric_dict = defaultdict(list)
        os.makedirs(export_dir, exist_ok=True)
        self.save_name = os.path.join(self.export_dir,'metric.png')
    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):

        if trainer.current_epoch <= self.start_after_epoch:
            return

        if self.monitor == 'all':

            for key, val in trainer.callback_metrics.items():
                if key.endswith('_epoch') or key.endswith('/epoch'):
                    self.metric_dict[key].append(val.item() if torch.is_tensor(val) else val)
        else:
            for key in self.monitor:
                if key.endswith('_epoch') or key.endswith('/epoch'):
                    val = trainer.callback_metrics.get(key)
                    self.metric_dict[key].append(val.item() if torch.is_tensor(val) else val)

        if self.best_metric_name and self.best_metric_name in self.metric_dict and len(self.metric_dict[self.best_metric_name]) > 0:
            current_score = self.metric_dict[self.best_metric_name][-1]
        else:
            current_score = -float('inf') if self.best_mode == 'max' else float('inf')

        if (self.best_mode == 'max' and current_score > self.best_score) or \
           (self.best_mode == 'min' and current_score < self.best_score):
            self.best_score = current_score
            self.best_epoch = trainer.current_epoch
        print('Plotting metrics curves')
        self._plot_metrics()
    def _plot_metrics(self):
        axs_num = len(self.metric_dict)

        fig, axs = plt.subplots(nrows=axs_num,figsize=(20,axs_num*6))

        for fig_idx, (y_name,y_data) in enumerate(self.metric_dict.items()):
            y_data = np.array(y_data)

            x = list(range(self.start_after_epoch,self.start_after_epoch+len(y_data)))
            axs[fig_idx].plot(x,y_data,'b-',label=y_name)
            axs[fig_idx].legend(loc='lower right')
            axs[fig_idx].grid()
            if self.best_epoch is not None:
                axs[fig_idx].axvline(self.best_epoch,color='red')

        fig.savefig(self.save_name)
        plt.close()

class ExportBestModelCallback(pl.Callback):
    def __init__(self, export_dir, monitor='val/meanf1score', mode='max', start_after_epoch=1):
        super().__init__()
        self.export_dir = export_dir
        self.monitor = monitor
        self.mode = mode
        self.start_after_epoch = start_after_epoch
        self.best_score = -float('inf') if mode == 'max' else float('inf')
        self.best_model_path = None
        os.makedirs(export_dir, exist_ok=True)
    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):

        if trainer.current_epoch <= self.start_after_epoch:
            return

        
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return

        current_score = current_score.item() if torch.is_tensor(current_score) else current_score
        
        
        if (self.mode == 'max' and current_score > self.best_score) or \
           (self.mode == 'min' and current_score < self.best_score):
            
            self.best_score = current_score
            # print(f"New best model at epoch {trainer.current_epoch}: {self.monitor}={current_score:.4f}")
            model_device = pl_module.device
            pl_module.to('cpu')
            model = pl_module.models
            model.eval()

            input_shape = model.get_input_shape()
            example_input = torch.randn(input_shape)

            traced_model = torch.jit.trace(model, example_input)
            model_path = os.path.join(self.export_dir, f"best_model.pt")
            self.best_model_path = model_path
            traced_model.save(model_path)
            model.train()
            pl_module.to(model_device)

            
class ExportALLModelCallback(pl.Callback):
    def __init__(self,
                 ckpt_dir: str,
                 monitor: str = 'acc',
                 mode: str = 'max',
                 start_after_epoch: int = 1,
                 export_device: str = 'cpu'):
        """
        Unified model callback: skip initial epochs, save latest and best checkpoints, and export best TorchScript model

        :param ckpt_dir: checkpoint save directory
        :param monitor: metric name to monitor
        :param mode: metric mode ('max' or 'min')
        :param start_after_epoch: start saving after this epoch (skip previous epochs)
        :param export_device: TorchScript export device ('cuda' or 'cpu')
        """
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.monitor = monitor
        self.mode = mode
        self.start_after_epoch = start_after_epoch
        self.export_device = export_device
        assert self.mode in ['min', 'max'], self.mode
        self.best_score = -float('inf') if mode == 'max' else float('inf')
        self.best_epoch = -1
        self.last_ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
        self.best_ckpt_path = os.path.join(ckpt_dir, "best.ckpt")
        self.best_pt_path = os.path.join(ckpt_dir, "best.pt")

        os.makedirs(ckpt_dir, exist_ok=True)

    def on_validation_end(self, trainer, pl_module):
        """Execute at the end of each validation epoch"""
        if trainer.current_epoch < self.start_after_epoch:
            return

        self._save_last_checkpoint(trainer, pl_module)

        current_score = self._get_current_score(trainer)
        if current_score is None:
            return

        is_better = (self.mode == 'max' and current_score > self.best_score) or \
                    (self.mode == 'min' and current_score < self.best_score)

        if is_better:
            self.best_score = current_score
            self.best_epoch = trainer.current_epoch

            self._save_best_checkpoint(trainer, pl_module)

            self._export_best_model(pl_module, trainer)
    
    def _get_current_score(self, trainer):
        """Get current monitored metric value"""
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return None
            
        if torch.is_tensor(current_score):
            return current_score.item()
        return current_score
    
    @rank_zero_only
    def _save_last_checkpoint(self, trainer, pl_module):
        """Save latest checkpoint"""
        checkpoint = {
            'epoch': trainer.current_epoch,
            'global_step': trainer.global_step,
            'state_dict': pl_module.state_dict(),
            'optimizer_states': [opt.state_dict() for opt in trainer.optimizers],
            'lr_schedulers': [scheduler.state_dict() for scheduler in trainer.lr_schedulers],
            'callbacks': trainer.checkpoint_connector.dump_checkpoint()['callbacks'],
            'pytorch-lightning_version': pl.__version__,
            'model_hparams': dict(pl_module.hparams),
            'monitor': self.monitor,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch
        }
        torch.save(checkpoint, self.last_ckpt_path)
        print(f"Saved last checkpoint to {self.last_ckpt_path}")
    
    @rank_zero_only
    def _save_best_checkpoint(self, trainer, pl_module):
        """Save best checkpoint"""
        checkpoint = torch.load(self.last_ckpt_path)

        checkpoint['best_score'] = self.best_score
        checkpoint['best_epoch'] = self.best_epoch

        torch.save(checkpoint, self.best_ckpt_path)
        print(f"Saved best checkpoint to {self.best_ckpt_path} (score={self.best_score:.4f})")
    
    @rank_zero_only
    def _export_best_model(self, pl_module, trainer):
        """Export best TorchScript model"""
        try:
            original_device = pl_module.device

            pl_module.to(self.export_device)
            model = pl_module.model
            model.eval()

            example_input = self._get_example_input(pl_module, trainer).to(self.export_device)

            with torch.no_grad():
                traced_model = torch.jit.trace(model, example_input)
                traced_model.save(self.best_pt_path)

            pl_module.to(original_device)
            
            print(f"Exported best TorchScript model to {self.best_pt_path}")
        except Exception as e:
            print(f"Error exporting model: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_example_input(self, pl_module, trainer):
        """Get example input"""
        try:
            if hasattr(trainer, 'datamodule') and hasattr(trainer.datamodule, 'val_dataloader'):
                val_loader = trainer.datamodule.val_dataloader()
                batch = next(iter(val_loader))
                return batch[0] if isinstance(batch, (list, tuple)) else batch
        except:
            pass

        try:
            if hasattr(pl_module, 'example_input_array') and pl_module.example_input_array is not None:
                return pl_module.example_input_array
        except:
            pass

        return torch.randn(1, 3, 224, 224)