import torch

from src.model.vqgan.vq_gan import Encoder, Decoder
from src.model.vqgan.quantize import VectorQuantizer2 as VectorQuantizer, EMAVectorQuantizer, GumbelQuantize

from transformers import PretrainedConfig, PreTrainedModel
from transformers import PretrainedConfig


class DDConfig(PretrainedConfig):
    model_type = "dd_config"

    def __init__(
            self,
            ch=32,
            in_channels=1,
            out_ch=1,
            ch_mult=[1, 2, 4],
            num_res_blocks=2,
            attn_resolutions=[],
            resolution=64,
            z_channels=32,
            double_z=False,
            dropout=0.0,
            resamp_with_conv=True,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.ch = ch
        self.out_ch = out_ch
        self.ch_mult = list(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = list(attn_resolutions)
        self.in_channels = in_channels
        self.resolution = resolution
        self.z_channels = z_channels
        self.double_z = double_z

        # optional
        self.dropout = dropout
        self.resamp_with_conv = resamp_with_conv


class VQGANConfig(PretrainedConfig):
    model_type = "vqgan"

    def __init__(
            self,
            dd_config: DDConfig = DDConfig(),
            n_embed: int = 256,
            embed_dim: int = 32,
            remap=None,
            sane_index_shape=True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dd_config = dd_config
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.remap = remap
        self.sane_index_shape = sane_index_shape


class VQGAN(PreTrainedModel):
    config_class = VQGANConfig

    def __init__(
            self,
            config: VQGANConfig
    ):
        super().__init__(config)

        dd = config.dd_config
        if isinstance(dd, dict):
            dd = DDConfig(**dd)
        dd_params = {
            k: v for k, v in dd.__dict__.items()
            if not k.startswith("_") and k != "model_type"
        }

        self.encoder = Encoder(**dd_params)
        self.decoder = Decoder(**dd_params)
        self.quantize = VectorQuantizer(n_e=self.config.n_embed, e_dim=self.config.embed_dim, beta=0.25,
                                        remap=config.remap, sane_index_shape=self.config.sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(dd.z_channels, self.config.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(self.config.embed_dim, dd.z_channels, 1)
        self.post_init()

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, x):
        quant, diff, [perplexity, _, indices] = self.encode(x)
        dec = self.decode(quant)
        return {
            "reconstruction": dec,
            "loss": diff,
            "indices": indices,
            "perplexity": perplexity
        }

    def get_last_layer(self):
        return self.decoder.conv_out.weight


class EMAVQGAN(VQGAN):
    def __init__(self, config: VQGANConfig):
        super().__init__(config)
        self.quantize = EMAVectorQuantizer(n_e=self.config.n_embed, e_dim=self.config.embed_dim,
                                           beta=0.25, decay=0.99, eps=1e-5,
                                           remap=config.remap, sane_index_shape=self.config.sane_index_shape)
        self.post_init()
