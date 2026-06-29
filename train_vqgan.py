import argparse

from pytorch_lightning.strategies import SingleDeviceStrategy

from src.utils.callback import ImageLogger
from src.datamodule.datamodule import CelebADatamodule
from lit_vqgan import LitVQGAN
from sconf import Config
import pytorch_lightning as pl
import torch
import warnings
import setproctitle

from src.vqgan import VQGANConfig, DDConfig

setproctitle.setproctitle("Train VQGAN")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def train(config: Config):
    pl.seed_everything(config.seed_everything, workers=True)

    vqgan_config = VQGANConfig(
        n_embed=256,
        embed_dim=32,
        dd_config=DDConfig(
            in_channels=3,
            out_ch=3,
            ch=32,
            z_channels=32,
            num_res_blocks=2,
            ch_mult=[1, 2, 2, 4],  # num_down = len(ch_mult)-1
            resolution=128,
            attn_resolutions=[],
            double_z=False,
        ),
    )
    model_module = LitVQGAN(**config.model, vqgan_config=vqgan_config)

    data_module = CelebADatamodule(**config.data)


    lr_callback = pl.callbacks.LearningRateMonitor(**config.callbacks.lr_monitor)

    # checkpoint_callback = pl.callbacks.ModelCheckpoint(**config.callbacks.ckpt_callback)
    # last_checkpoint_callback = pl.callbacks.ModelCheckpoint(filename="last", save_top_k=1, monitor=None)
    # logger = WandbLogger(**config.wandb)
    # logger.watch(model_module, "all", log_freq=2000)
    image_logger = ImageLogger(**config.callbacks.image_logger)

    trainer = pl.Trainer(
        **config.trainer,
        strategy=SingleDeviceStrategy(device="cuda:0"),
        # logger=logger,
        callbacks=[
            lr_callback,
            # checkpoint_callback,
            # last_checkpoint_callback,
            image_logger,
        ],
    )
    trainer.fit(model_module, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default="config/vqgan_config.yaml",
                        help="Path to the config file.")
    args = parser.parse_args()
    config = Config(args.config)
    train(config)
