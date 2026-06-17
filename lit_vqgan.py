import numpy as np
import pytorch_lightning as pl
import torch
from timm.scheduler import CosineLRScheduler

from torchmetrics.image import StructuralSimilarityIndexMeasure

from src.model.vqgan.vqperceptual import VQLPIPSWithDiscriminator
from src.utils.utils import log_parameters
from src.vqgan import VQGANConfig, DDConfig, VQGAN, EMAVQGAN
import wandb


class LitVQGAN(pl.LightningModule):
    def __init__(
            self,
            vqgan_config,
            loss_config,
            train_config,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.register_buffer("ae_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("disc_step", torch.tensor(0, dtype=torch.long))

        self.model = EMAVQGAN(config=vqgan_config, )

        self.loss = VQLPIPSWithDiscriminator(**loss_config)

        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=2.0, sigma=(0.5, 0.5), kernel_size=(3, 3))

        self.save_hyperparameters()
        self.K = self.model.quantize.n_e

        self.gradient_clip_val = getattr(self.hparams.train_config, 'gradient_clip_val', None)
        self.accumulate_grad_batches = getattr(self.hparams.train_config, 'accumulate_grad_batches', 1)

        log_parameters(self.model)
        log_parameters(self.loss)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        accumulation_done = (
                (batch_idx + 1) % self.accumulate_grad_batches == 0
                or (batch_idx + 1) == len(self.trainer.train_dataloader)
        )
        opt_ae, opt_disc = self.optimizers()
        sched_ae, sched_disc = self.lr_schedulers()

        x = batch["img"]
        out = self(x)
        xrec, qloss, indices = out["reconstruction"], out["loss"], out["indices"]

        # ============================
        #   Generator / Autoencoder
        # ============================
        aeloss, log_dict_ae = self.loss(
            qloss, x, xrec, 0, self.ae_step.item(), last_layer=self.model.get_last_layer(), split="train"
        )
        self.manual_backward(aeloss)

        if accumulation_done:
            if self.gradient_clip_val:
                self.clip_gradients(opt_ae, gradient_clip_val=self.gradient_clip_val)
            opt_ae.step()
            sched_ae.step_update(self.ae_step)
            opt_ae.zero_grad()
            self.ae_step += 1

        # ======================
        #     Discriminator
        # ======================
        discloss, log_dict_disc = self.loss(
            qloss, x, xrec, 1, self.ae_step.item(), last_layer=self.model.get_last_layer(), split="train"
        )
        self.manual_backward(discloss)

        if accumulation_done:
            if self.gradient_clip_val:
                self.clip_gradients(opt_disc, gradient_clip_val=self.gradient_clip_val)
            opt_disc.step()
            sched_disc.step_update(self.disc_step)
            opt_disc.zero_grad()
            if self.ae_step.item() > self.loss.discriminator_iter_start:
                self.disc_step += 1

        # Codebook utilization
        flat = indices.long().view(-1)  # Get 1 sample's indices
        counts = torch.bincount(flat, minlength=self.K)
        utilization = (counts > 0).float().mean()

        # Logging
        self.log("train/codebook_util", utilization, prog_bar=False, on_step=True, on_epoch=False)
        self.log("train/codebook_used", (counts > 0).sum(), prog_bar=False, on_step=True, on_epoch=False)
        self.log("train/perplexity", out["perplexity"], prog_bar=False, on_step=True, on_epoch=False)
        if self.logger and "wandb" in str(
                type(
                    self.logger.experiment)) and self.global_step % self.hparams.train_config.log_hist_every_n_steps == 0:
            indices_for_hist = np.repeat(np.arange(self.K), counts.cpu().numpy())
            wandb.log({"train/codebook_hist": wandb.Histogram(indices_for_hist)})

        # Logging
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("ae_step", self.ae_step, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("disc_step", self.disc_step, prog_bar=False, logger=True, on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        x = batch["img"]

        out = self(x)
        xrec, qloss, _ = out["reconstruction"], out["loss"], out["indices"]
        aeloss, log_dict_ae = self.loss(
            qloss, x, xrec, 0, self.ae_step.item(), last_layer=self.model.get_last_layer(), split="val"
        )

        discloss, log_dict_disc = self.loss(
            qloss, x, xrec, 1, self.ae_step.item(), last_layer=self.model.get_last_layer(), split="val"
        )

        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        self.val_ssim(xrec, x)
        self.log("val_ssim", self.val_ssim, prog_bar=True, logger=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        opt_ae = torch.optim.Adam((p for p in self.model.parameters() if p.requires_grad),
                                  lr=self.hparams.train_config.learning_rate, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=self.hparams.train_config.dis_learning_rate, betas=(0.5, 0.9))
        scheduler_ae = CosineLRScheduler(opt_ae, **self.hparams.train_config.cosine_scheduler)
        scheduler_disc = CosineLRScheduler(opt_disc, **self.hparams.train_config.dis_cosine_scheduler)
        return (
            [opt_ae, opt_disc],
            [
                {
                    'scheduler': scheduler_ae,
                    'interval': self.hparams.train_config.interval,
                    'frequency': 1,
                },
                {
                    'scheduler': scheduler_disc,
                    'interval': self.hparams.train_config.interval,
                    'frequency': 1,
                }
            ]
        )

    def log_images(self, batch, **kwargs):
        log = dict()
        x = batch["img"]
        x = x.to(self.device)
        out = self(x)
        xrec, qloss, indices = out["reconstruction"], out["loss"], out["indices"]
        log["inputs"] = x
        log["reconstructions"] = xrec
        log["indices"] = indices
        return log
