import os
import matplotlib

matplotlib.use('Agg')
import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning import Callback
from pytorch_lightning import loggers
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import wandb
import matplotlib.cm

class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            loggers.WandbLogger: self._wandb,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        # raise ValueError("No way wandb")
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)

            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.model.eval()

            mini_batch = {k: v[:self.max_images] for k, v in batch.items()}
            with torch.no_grad():
                images = pl_module.log_images(mini_batch, split=split, pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]

                if k == "indices" or k.startswith("indices_"):
                    idx_numpy = images[k].detach().cpu().numpy()
                    if idx_numpy.ndim == 4:
                        idx_numpy = idx_numpy.squeeze(1)
                    norm_indices = idx_numpy / pl_module.K
                    cmap = matplotlib.cm.get_cmap('nipy_spectral')
                    colored_numpy = cmap(norm_indices)
                    rgb_numpy = colored_numpy[..., :3]
                    colored_tensor = torch.from_numpy(rgb_numpy).permute(0, 3, 1, 2).float()
                    images[k] = colored_tensor * 2.0 - 1.0

                elif isinstance(images[k], torch.Tensor) and (k != "indices" or not k.startswith("indices_")):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.trainer.default_root_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.model.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        mini_batch = {k: v[:self.max_images] for k, v in batch.items()}
        self.log_img(pl_module, mini_batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        mini_batch = {k: v[:self.max_images] for k, v in batch.items()}
        self.log_img(pl_module, mini_batch, batch_idx, split="val")
