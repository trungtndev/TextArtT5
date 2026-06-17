from typing import Tuple

import pytorch_lightning as pl
import torch
from einops import rearrange
from timm.scheduler import CosineLRScheduler

from src.TextArtT5.TextArtT5 import TextArtT5Config, TextArtT5ForConditionalGeneration
from src.TextArtT5.processor import TextArtT5Processor

from src.utils.utils import log_parameters
from src.vqgan import EMAVQGAN, VQGAN, VQGANConfig, DDConfig


def create_stage_bundle(seed, device='cuda'):
    with torch.random.fork_rng(devices=[device] if torch.cuda.is_available() else []):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        stage = {
            'seed': seed,
            'cpu_state': torch.get_rng_state(),
            'gpu_state': torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None
        }
    return stage


def apply_stage(stage, device='cuda'):
    torch.set_rng_state(stage['cpu_state'])
    if stage['gpu_state'] is not None:
        torch.cuda.set_rng_state(stage['gpu_state'], device)


class LitAutoRegression(pl.LightningModule):
    def __init__(
            self,
            train_config,
            scribenet_config: TextArtT5Config,
            image_tokenizer_path_or_id: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # self.tokenizer = EMAVQGAN.from_pretrained(image_tokenizer_path_or_id).eval()

        self.tokenizer = EMAVQGAN(
            config=VQGANConfig(
                n_embed=1024,
                embed_dim=32,
                dd_config=DDConfig(
                    in_channels=3,
                    out_ch=3,
                    ch=32,
                    z_channels=32,
                    num_res_blocks=2,
                    ch_mult=[1, 1, 2, 2, 4],  # num_down = len(ch_mult)-1
                    resolution=256,
                    attn_resolutions=[],
                    double_z=False,
                ),
            )).eval()
        for param in self.tokenizer.parameters():
            param.requires_grad = False

        self.codebook_size = self.tokenizer.quantize.n_e
        self.codebook_dim = self.tokenizer.quantize.e_dim

        self.K = self.codebook_size

        self.processer = TextArtT5Processor.from_pretrained("./textart_t5", codebook_size=self.codebook_size)

        self.model = TextArtT5ForConditionalGeneration(scribenet_config)
        # self.val_ssim = StructuralSimilarityIndexMeasure(data_range=2.0, sigma=(0.6, 0.6), kernel_size=(7, 7))
        # log_parameters(self.tokenizer)
        log_parameters(self.model.model)

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels=None, **kwargs, ):
        if labels is not None:
            decoder_input_ids = decoder_input_ids[:, :-1]
            decoder_attention_mask = decoder_attention_mask[:, :-1]

        return self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask,
            labels=labels, **kwargs
        )

    def _create_labels(self, input_ids):
        pad_token_id = self.model.config.pad_token_id
        labels = input_ids.clone()[:, 1:]
        labels[labels == pad_token_id] = -100
        return labels

    def training_step(self, batch, batch_idx):
        text_cond = batch["txt"]

        if "inds_tensor" in batch:
            z_indices = batch["inds_tensor"]
        else:
            x = batch["img"]
            with torch.no_grad():
                _, _, [_, _, z_indices] = self.tokenizer.encode(x)
                z_indices = z_indices.detach()

        z_indices = rearrange(z_indices, "b h w -> b (h w)")

        inputs = self.processer(text=text_cond, codebook=z_indices).to(self.device)

        labels = self._create_labels(inputs["decoder_input_ids"])

        if self.hparams.train_config.perturb_prob > 0.0:
            decoder_ids = inputs["decoder_input_ids"]

            mask = torch.rand_like(decoder_ids, dtype=torch.float32) < self.hparams.train_config.perturb_prob

            pad_id = self.model.config.pad_token_id
            eos_id = self.model.config.eos_token_id

            mask &= (decoder_ids != pad_id)
            mask &= (decoder_ids != eos_id)
            mask &= (decoder_ids != 151644)  # bos ~ <|im_start|>

            r_indices = torch.randint(
                low=0,
                high=self.codebook_size,
                size=decoder_ids.shape,
                device=decoder_ids.device,
                dtype=decoder_ids.dtype
            )

            inputs["decoder_input_ids"] = torch.where(mask, r_indices, decoder_ids)

        outputs = self(**inputs, labels=labels)
        loss = outputs.loss

        self.log("train/loss", loss, prog_bar=True, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        text_cond = batch["txt"]

        if "inds_tensor" in batch:
            z_indices = batch["inds_tensor"]
        else:
            x = batch["img"]
            with torch.no_grad():
                _, _, [_, _, z_indices] = self.tokenizer.encode(x)
                z_indices = z_indices.detach()

        z_indices = rearrange(z_indices, "b h w -> b (h w)")

        inputs = self.processer(text=text_cond, codebook=z_indices).to(self.device)

        labels = self._create_labels(inputs["decoder_input_ids"])
        outputs = self(**inputs, labels=labels)
        loss = outputs.loss

        self.log("val/loss", loss, prog_bar=True, on_step=True, logger=True)
        return loss

    def lr_scheduler_step(self, scheduler, optimizer_idx):
        scheduler.step_update(self.global_step)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.train_config.learning_rate
        )
        scheduler_optim = CosineLRScheduler(optim, **self.hparams.train_config.cosine_scheduler)
        scheduler = {
            'scheduler': scheduler_optim,
            'interval': self.hparams.train_config.interval,
            'frequency': 1,
        }
        return [optim], [scheduler]

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = {}

        x = batch["img"]
        text_cond = batch["txt"]
        if "inds_tensor" in batch:
            z_indices = batch["inds_tensor"]
        else:
            with torch.no_grad():
                _, _, [_, _, z_indices] = self.tokenizer.encode(x)
                z_indices = z_indices.detach()

        b, h, w = z_indices.size(0), z_indices.size(1), z_indices.size(2)
        bhwc = (b, h, w, self.codebook_dim)
        l = h * w
        z_indices_flat = rearrange(z_indices, "b h w -> b (h w)")

        cond_inputs = self.processer(text=text_cond, ).to(self.device)

        # --- 1. RECONSTRUCTION (Kiểm tra chất lượng tối đa của VQGAN) ---
        x_rec = self.decode_to_img(z_indices_flat.to(self.device), bhwc)

        # --- 2. NOPIX SAMPLED (Vẽ từ đầu dựa trên LaTeX - Sampled) ---
        out_nopix = self.model.generate(
            **cond_inputs,
            min_new_tokens=l + 1,
            max_new_tokens=l + 1,
            do_sample=True,
            top_k=10,
            temperature=1.0,
            # eos_token_id=self.processer.eos_token_id,
            forced_eos_token_id=None,
            use_cache=True,
        )

        out_nopix = out_nopix[:, 1: -1]
        out_nopix = torch.clamp(out_nopix, min=0, max=self.K - 1)
        x_sample_nopix = self.decode_to_img(out_nopix, bhwc)

        # --- 3. NOPIX DETERMINISTIC ---
        out_det = self.model.generate(
            **cond_inputs,
            min_new_tokens=l + 1,
            max_new_tokens=l + 1,
            do_sample=False,
            # eos_token_id=self.processer.eos_token_id,
            forced_eos_token_id=None,
            use_cache=True,
        )

        out_det = out_det[:, 1: -1]
        out_det = torch.clamp(out_det, min=0, max=self.K - 1)
        x_sample_det = self.decode_to_img(out_det, bhwc)

        # --- 4. HALF ---
        z_half = z_indices_flat[:, : l // 2]
        half_inputs = self.processer(
            text=text_cond,
            codebook=z_half,
            pre_codebook=True
        ).to(self.device)

        out_half = self.model.generate(
            **half_inputs,
            min_new_tokens=(l - (l // 2)) + 1,
            max_new_tokens=(l - (l // 2)) + 1,
            do_sample=True,
            top_k=10,
            temperature=1.0,
            # eos_token_id=self.processer.codebook_tokenizer.eos_token_id,
                forced_eos_token_id=None,
            use_cache=True,
        )
        out_half = out_half[:, 1: -1]
        out_half = torch.clamp(out_half, min=0, max=self.K - 1)
        x_sample_half = self.decode_to_img(out_half, bhwc)

        # --- ĐÓNG GÓI LOG ---

        log["inputs"] = x
        log["reconstructions"] = x_rec
        log["samples_half"] = x_sample_half
        log["samples_nopix"] = x_sample_nopix
        log["samples_det"] = x_sample_det

        # Log indices để soi nếu ảnh bị lỗi
        log["indices_input"] = z_indices
        log["indices_nopix"] = out_nopix.reshape(b, h, w)
        log["indices_det"] = out_det.reshape(b, h, w)
        log["indices_half"] = out_half.reshape(b, h, w)
        return log

    def decode_to_img(self, z_indices: torch.LongTensor, bhwc: Tuple[int, int, int, int]) -> torch.FloatTensor:
        quant_z = self.tokenizer.quantize.get_codebook_entry(
            z_indices.reshape(-1),
            bhwc
        )
        xrec = self.tokenizer.decode(quant_z)
        return xrec
