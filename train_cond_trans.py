import argparse

import torch

from src.utils.callback import ImageLogger
from src.datamodule.datamodule import CelebADatamodule
from src.TextArtT5.TextArtT5 import TextArtT5Config
from lit_cond_trans import LitAutoRegression
from sconf import Config
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy, SingleDeviceStrategy
from pytorch_lightning.loggers.wandb import WandbLogger
import os
import warnings
import setproctitle

setproctitle.setproctitle("Train HMEG Transformer")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(config: Config):
    pl.seed_everything(config.seed_everything, workers=True)
    codebook_size = 1024
    config_net = TextArtT5Config(
        attn_implementation="sdpa",
        encoder_vocab_size=151665,  # text size
        vocab_size=codebook_size + 4,  # codebook size=256
        codebook_size=codebook_size,
        max_position_embeddings=4096,

        encoder_layers=10,
        encoder_ffn_dim=2048,
        encoder_attention_heads=8,

        decoder_layers=20,
        decoder_ffn_dim=4096,
        decoder_attention_heads=8,

        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        activation_function="gelu",
        d_model=16,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        classifier_dropout=0.0,
        scale_embedding=False,
        use_cache=True,

        encoder_pad_token_id=151643,
        encoder_unk_token_id=1,
        encoder_cls_token_id=2,
        encoder_sep_token_id=3,

        pad_token_id=codebook_size,
        unk_token_id=codebook_size + 1,
        bos_token_id=codebook_size + 2,
        eos_token_id=codebook_size + 3,
        decoder_start_token_id=codebook_size + 2,
        forced_eos_token_id=codebook_size + 3,
    )
    model_module = LitAutoRegression(**config.model, scribenet_config=config_net)
    # model_module = LitNonRegression(**config.model)

    # model_module = LatentDiffusion(**config.model)

    # data_module = VQGANDatamodule(**config.data)
    data_module = CelebADatamodule(**config.data)

    lr_callback = pl.callbacks.LearningRateMonitor(**config.callbacks.lr_monitor)

    # checkpoint_callback = pl.callbacks.ModelCheckpoint(**config.callbacks.ckpt_callback)
    # last_checkpoint_callback = pl.callbacks.ModelCheckpoint(filename="last", save_top_k=1, monitor=None)
    # logger = WandbLogger(**config.wandb)
    # logger.watch(model_module, "all", log_freq=50)
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
    parser.add_argument("--config", type=str, required=False, default="config/ar_config.yaml")
    args = parser.parse_args()
    config = Config(args.config)
    train(config)
