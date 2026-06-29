import math
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn
from transformers.utils import logging

from vllm.config import CacheConfig, VllmConfig
from vllm.config.lora import LoRAConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn

from vllm.v1.attention.backend import AttentionType
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.cross_attention import CrossAttention
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.multimodal.processing.dummy_inputs import BaseDummyInputsBuilder

from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsQuant,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    cast_overflow_tensors,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ModalityData,
    ModalityDataItems,
    ModalityDataParser,
    MultiModalDataItems,
    MultiModalDataParser,
    ProcessorBatchItems,
)
from vllm.multimodal.processing import (
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptUpdate,
)
from vllm.inputs import MultiModalDataDict

from vllm.sequence import IntermediateTensors
from vllm.utils.collection_utils import is_list_of

from .TextArtT5 import TextArtT5Config

logger = logging.get_logger(__name__)

def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean-ish environment variable.

    Accepted truthy: 1, true, yes, on
    Accepted falsy: 0, false, no, off
    """
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    logger.warning("Unrecognized value for %s=%r; using default=%s", name, val, default)
    return default


def get_bsz_seq_len(input_ids):
    shp = input_ids.shape
    ndim = len(shp)
    if ndim == 1:
        return 1, input_ids.numel()
    else:
        return shp[:2]

class TextArtT5LearnedPositionalEmbedding(VocabParallelEmbedding):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)

    def forward(
            self,
            positions: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(positions)


class TextArtT5EncoderAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            bias: bool = True,
            config: TextArtT5Config | None = None,
            cache_config: CacheConfig | None = None,
            quant_config: QuantizationConfig | None = None,
            prefix: str = "",
    ):
        super().__init__()
        self.d_model = config.d_model
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            self.d_model,
            self.d_model // self.total_num_heads,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
        )

        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        if self.total_num_kv_heads >= tp_world_size:
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.attn = MMEncoderAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        is_2d = q.dim() == 2
        if is_2d:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

        attn_output = self.attn(q, k, v)

        output, _ = self.out_proj(attn_output)
        if is_2d:
            output = output.squeeze(0)
        return output


class TextArtT5DecoderSelfAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            bias: bool = True,
            config: TextArtT5Config | None = None,
            cache_config: CacheConfig | None = None,
            quant_config: QuantizationConfig | None = None,
            prefix: str = "",
    ):
        super().__init__()
        self.d_model = config.d_model
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            self.d_model,
            self.d_model // self.total_num_heads,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
        )

        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        if self.total_num_kv_heads >= tp_world_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.DECODER,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        attn_output = self.attn(q, k, v)

        output, _ = self.out_proj(attn_output)
        return output


class TextArtT5CrossAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            bias: bool = True,
            config: TextArtT5Config | None = None,
            cache_config: CacheConfig | None = None,
            quant_config: QuantizationConfig | None = None,
            prefix: str = "",
    ):
        super().__init__()
        self.prefix = prefix
        self.d_model = config.d_model
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.kv_size = self.total_num_kv_heads * self.head_dim

        # Q_proj for projecting decoder hidden states
        self.q_proj = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )

        # KV_proj for projecting encoder hidden states with no overhead of
        # unused Q_proj by setting total_num_heads to 0
        self.kv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=0,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj",
        )

        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        if self.total_num_kv_heads >= tp_world_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = self.num_heads  # No GQA in bart
        self.attn = CrossAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.ENCODER_DECODER,
        )

    def forward(
            self,
            decoder_hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""

        q, _ = self.q_proj(decoder_hidden_states)

        # Encoder hidden states are only computed once during prefill phase.
        # Afterwards, the keys and values should be available in the kv-cache.
        if encoder_hidden_states is not None:
            kv, _ = self.kv_proj(encoder_hidden_states)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        else:
            k = v = None

        attn_output = self.attn(q, k, v)
        output, _ = self.out_proj(attn_output)
        return output


class TextArtT5MLP(nn.Module):
    def __init__(
            self,
            config: TextArtT5Config,
            quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()

        hidden_size = config.d_model
        intermediate_size = config.encoder_ffn_dim
        ffn_has_bias = True
        self.fc1 = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=ffn_has_bias,
            quant_config=quant_config,
        )

        self.activation = get_act_fn(config.activation_function)

        self.fc2 = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=ffn_has_bias,
            quant_config=quant_config,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class TextArtT5EncoderLayer(nn.Module):
    def __init__(
            self,
            config: TextArtT5Config,
            cache_config: CacheConfig | None = None,
            quant_config: QuantizationConfig | None = None,
            prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = TextArtT5EncoderAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = get_act_fn(config.activation_function)

        self.mlp = TextArtT5MLP(
            config=config,
            quant_config=quant_config,
        )

        self.ffn_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class TextArtT5DecoderLayer(nn.Module):
    def __init__(
            self,
            config: TextArtT5Config,
            cache_config: CacheConfig | None = None,
            quant_config: QuantizationConfig | None = None,
            prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = TextArtT5DecoderSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        """
        afeldman-nm: personally I would call this "cross-attention",
        however I left the name as "encoder_attn" to maintain consistency
        with the name of the pretrained weights.
        """
        self.encoder_attn = TextArtT5CrossAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            config=config,
            prefix=f"{prefix}.encoder_attn",
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.mlp = TextArtT5MLP(
            config=config,
            quant_config=quant_config,
        )

        self.ffn_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            decoder_hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = decoder_hidden_states
        hidden_states = self.self_attn_layer_norm(decoder_hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn(
            decoder_hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class TextArtT5Encoder(nn.Module):

    def __init__(
            self,
            config: TextArtT5Config,
            cache_config: CacheConfig | None = None,
            quant_config: QuantizationConfig | None = None,
            lora_config: LoRAConfig | None = None,
            embed_tokens: nn.Embedding | None = None,
            prefix: str = "",
    ):
        super().__init__()

        self.cache_config = cache_config
        self.quant_config = quant_config
        self.lora_config = lora_config
        embed_dim = config.d_model
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = VocabParallelEmbedding(
            config.encoder_vocab_size, embed_dim,
        )

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = TextArtT5LearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList(
            [
                TextArtT5EncoderLayer(
                    config,
                    cache_config,
                    quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.encoder_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # retrieve input_ids and inputs_embeds
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        embed_pos = self.embed_positions(positions)
        embed_pos = embed_pos.to(inputs_embeds.device)
        hidden_states = inputs_embeds + embed_pos

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states=hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class TextArtT5CodeBookDecoder(nn.Module):

    def __init__(
            self,
            config: TextArtT5Config,
            cache_config: CacheConfig | None = None,
            quant_config: QuantizationConfig | None = None,
            lora_config: LoRAConfig | None = None,
            embed_tokens: nn.Embedding | None = None,
            prefix: str = "",
    ):
        super().__init__()
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.lora_config = lora_config
        self.max_target_positions = config.max_position_embeddings

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.d_model
        )

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = TextArtT5LearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )

        self.layers = nn.ModuleList([
                TextArtT5DecoderLayer(
                    config,
                    cache_config,
                    quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                ) for layer_idx in range(config.decoder_layers)
            ])

        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def forward(
            self,
            decoder_input_ids: torch.Tensor,
            decoder_positions: torch.Tensor,
            encoder_hidden_states: torch.Tensor | None,
            inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert decoder_input_ids is not None
            inputs_embeds = self.embed_input_ids(decoder_input_ids)

        embed_pos = self.embed_positions(decoder_positions)
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                decoder_hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class TextArtT5Model(nn.Module, SupportsQuant):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config

        lora_vocab = (
            (lora_config.lora_extra_vocab_size * (lora_config.max_loras or 1))
            if lora_config
            else 0
        )
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.encoder = TextArtT5Encoder(
            config, cache_config, quant_config=quant_config, prefix=f"{prefix}.encoder"
        )
        self.decoder = TextArtT5CodeBookDecoder(
            config, cache_config, quant_config=quant_config, prefix=f"{prefix}.decoder"
        )

    def forward(
            self,
            input_ids: torch.Tensor | None,
            positions: torch.Tensor,
            inputs_embeds: torch.Tensor | None,
            encoder_outputs: list[torch.Tensor],
    ) -> torch.Tensor:

        decoder_outputs = self.decoder(
            decoder_input_ids=input_ids,
            decoder_positions=positions,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_outputs,
        )

        return decoder_outputs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        # Unify kv only for cross-attention, while keeping q separate
        cross_attn_stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("kv_proj", "k_proj", "k"),
            ("kv_proj", "v_proj", "v"),
        ]

        other_weights = []
        loaded_stacked_params = []
        model_params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in cross_attn_stacked_params_mapping:
                if weight_name not in name or "encoder_attn" not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in model_params_dict:
                    continue
                param = model_params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_stacked_params.append(name)
                break
            else:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name or "encoder_attn" in name:
                        # Also skip q_proj in cross_attn which
                        # can be loaded normally
                        continue
                    name = name.replace(weight_name, param_name)
                    if name not in model_params_dict:
                        continue
                    param = model_params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_stacked_params.append(name)
                    break
                else:
                    if name in model_params_dict:
                        other_weights.append((name, loaded_weight))

        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(other_weights)
        loaded_params.update(loaded_stacked_params)
        return loaded_params


class TextArtT5ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> Any:
        return self.ctx.model_config.hf_config

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"text": 1}

    def get_mm_max_tokens_per_item(
            self,
            seq_len: int,
            mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        config = self.get_hf_config()
        return {"text": config.max_position_embeddings}

    def get_data_parser(self) -> MultiModalDataParser:
        return TextArtT5DataParser()


if hasattr(BaseProcessingInfo, "get_default_tok_params"):
    def _textart_get_default_tok_params(self):
        return super(TextArtT5ProcessingInfo, self).get_default_tok_params() \
            .with_kwargs()


    TextArtT5ProcessingInfo.get_default_tok_params = _textart_get_default_tok_params  # type: ignore[attr-defined]

class TextArtT5MultiModalProcessor(EncDecMultiModalProcessor[TextArtT5ProcessingInfo]):
    def create_encoder_prompt(self, prompt: str | list[int], mm_items: MultiModalDataItems) -> str | list[int]:
        # if isinstance(prompt, str) and prompt:
        #     tokenizer = self.info.get_tokenizer()
        #     tokens = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"].flatten()
        #     return tokens.tolist()
        # return [0]
        if isinstance(prompt, str) and prompt:
            tokenizer = self.info.get_tokenizer()
            token_ids = tokenizer.apply_chat_template([{"caption": prompt}], tokenize=True, return_tensors="pt")["input_ids"].flatten()
            return token_ids.tolist()
        return [0]


    def create_decoder_prompt(self, prompt: str | list[int], mm_items: MultiModalDataItems) -> str | list[int]:
        return prompt

    def _call_hf_processor(
        self, prompt: str, mm_data: Mapping[str, object], mm_kwargs: Mapping[str, object], tok_kwargs: Mapping[str, object]
    ):
        from transformers.feature_extraction_utils import BatchFeature
        tokenizer = self.info.get_tokenizer()
        has_encoder_data = mm_data is not None and "texts" in mm_data
        result = {}

        if has_encoder_data:
            encoder_texts = mm_data["texts"]
            encoder_text = encoder_texts[0] if encoder_texts else ""
            # encoder_tokenized = tokenizer(encoder_text, return_tensors="pt", add_special_tokens=True)
            encoder_tokenized = tokenizer.apply_chat_template([{"caption": encoder_text}], tokenize=True, return_tensors="pt")
            result["encoder_input_ids"] = encoder_tokenized["input_ids"]

        if isinstance(prompt, (list, tuple)) and len(prompt) > 0 and isinstance(prompt[0], int):
            result["input_ids"] = torch.tensor([prompt])
        else:
            prompt_tokenized = tokenizer(prompt if prompt else "", return_tensors="pt", **tok_kwargs)
            result["input_ids"] = prompt_tokenized["input_ids"]

        return BatchFeature(result)

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs: Mapping[str, object]) -> Mapping[str, MultiModalFieldConfig]:
        return dict(encoder_input_ids=MultiModalFieldConfig.batched("text"))

    def _get_prompt_updates(
        self, mm_items: MultiModalDataItems, hf_processor_mm_kwargs: Mapping[str, object], out_mm_kwargs: MultiModalKwargsItems
    ) -> Sequence[PromptUpdate]:
        from vllm.multimodal.processing import PromptReplacement
        num_text_items = mm_items.get_count("text", strict=False)
        if num_text_items == 0:
            return []

        text_items = mm_items.get_items("text", TextArtT5ProcessorItems)
        tokenizer = self.info.get_tokenizer()
        text = text_items.get(0)
        # num_tokens = len(tokenizer.encode(text, add_special_tokens=True))
        num_tokens = len(tokenizer.apply_chat_template([{"caption": text}], tokenize=True))

        return [PromptReplacement(modality="text", target=[0], replacement=[0] * num_tokens)]

class TextArtT5DummyInputsBuilder(BaseDummyInputsBuilder[TextArtT5ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(self,
                          seq_len: int, mm_counts: Mapping[str, int],
                          mm_options: Mapping[str, BaseDummyOptions] | None = None,
                          ) -> MultiModalDataDict:
        num_texts = mm_counts.get("text", 0)
        if num_texts == 0:
            return {}
        dummy_text = " ".join(["word"] * (seq_len - 10)) # TODO: change dummy_text when use aply_chat_template
        return {"text": dummy_text}


class TextArtT5ProcessorItems(ProcessorBatchItems[str]):
    def __init__(self, data) -> None:
        if data is None:
            data = [""]
        elif isinstance(data, str):
            data = [data]
        super().__init__(data, "text")


class TextArtT5DataParser(MultiModalDataParser):
    def __init__(self):
        super().__init__()

    def _parse_text_data(self, data: ModalityData[str], ) -> ModalityDataItems[Any, Any] | None:
        if data is None or not len(data):
            return TextArtT5ProcessorItems(None)

        if isinstance(data, str) or is_list_of(data, str):
            return TextArtT5ProcessorItems(data)
        else:
            raise TypeError(
                f"Text data must be a string or list of strings, got {type(data)}"
            )

    def _get_subparsers(self) -> Mapping[str, ModalityDataParser]:
        return {
            "text": self._parse_text_data,
        }


@MULTIMODAL_REGISTRY.register_processor(
    TextArtT5MultiModalProcessor,
    info=TextArtT5ProcessingInfo,
    dummy_inputs=TextArtT5DummyInputsBuilder,
)
class TextArtT5ForConditionalGeneration(nn.Module, SupportsQuant, SupportsMultiModal):
    # Map LayerNorm to layernorm to match vLLM's internal configuration.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "model.",
        },
        orig_to_new_substr={
            "LayerNorm": "layernorm",
        },
    )


    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        lora_config = vllm_config.lora_config

        # Verify the weight-sharing constraint from the original configuration.
        assert config.tie_word_embeddings
        self.config = config

        # Initialize the core TextArtT5Model block.
        self.model = TextArtT5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

        self.lm_head = ParallelLMHead(
            config.vocab_size, config.d_model, bias=True
        )

        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size, config.vocab_size
        )

        # Optimize encoder block processing with a custom environment variable.
        self._encoder_max_seq_padding = _env_flag(
            "VLLM_TEXTART_ENCODER_MAX_SEQ_PADDING", default=False
        )
        self._pad_id = getattr(self.config, "encoder_pad_token_id", None) # NOTE: verify config
        if self._encoder_max_seq_padding and self._pad_id is None:
            self._encoder_max_seq_padding = False

    def get_language_model(self) -> nn.Module:
        return self.model.decoder

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.decoder.embed_tokens(input_ids)

    def embed_multimodal(self, **kwargs) -> MultiModalEmbeddings:
        encoder_input_ids_list = self._parse_and_validate_encoder_input(**kwargs)

        # TODO: For debug
        if not encoder_input_ids_list:
            raise ValueError(
                "encoder_input_ids_list is empty. Please check the pipeline."
            )

        # Run each sequence through the encoder sequentially.
        if not self._encoder_max_seq_padding:
            encoder_outputs: list[torch.Tensor] = []
            for encoder_input_ids in encoder_input_ids_list:
                encoder_positions = torch.arange(
                    encoder_input_ids.size(-1),
                    dtype=torch.long,
                    device=encoder_input_ids.device,
                )

                encoder_output = self.model.encoder(
                    input_ids=encoder_input_ids.squeeze(0),
                    positions=encoder_positions,
                )
                encoder_outputs.append(encoder_output)
        else:
            lengths = [t.numel() for t in encoder_input_ids_list]
            max_len = max(lengths) if lengths else 0
            assert max_len > 0, "Detected empty encoder_input_ids."

            same_len = all(l == max_len for l in lengths)
            if len(encoder_input_ids_list) == 1:
                batch_encoder_input_ids = encoder_input_ids_list[0]
            elif same_len:
                batch_encoder_input_ids = torch.cat(encoder_input_ids_list, dim=0)
            else:
                batch_encoder_input_ids = torch.full(
                    (len(encoder_input_ids_list), max_len),
                    fill_value=self._pad_id,
                    dtype=encoder_input_ids_list[0].dtype,
                    device=encoder_input_ids_list[0].device,
                )
                for i, t in enumerate(encoder_input_ids_list):
                    batch_encoder_input_ids[i, : t.numel()] = t.squeeze()

            batch_encoder_positions = (
                torch.arange(
                    max_len,
                    dtype=torch.long,
                    device=batch_encoder_input_ids.device,
                )
                .unsqueeze(0)
                .expand(batch_encoder_input_ids.size(0), -1)
            )

            batch_encoder_output = self.model.encoder(
                input_ids=batch_encoder_input_ids,
                positions=batch_encoder_positions,
            )

            # Unbind back into a list and trim any extra padding.
            encoder_outputs: list[torch.Tensor] = batch_encoder_output.unbind(dim=0)
            if not same_len:
                encoder_outputs = [
                    out[:l] for out, l in zip(encoder_outputs, lengths, strict=False)
                ]
        return encoder_outputs

    def _parse_and_validate_encoder_input(self, **kwargs: object) -> list[torch.Tensor]:
        encoder_input_ids = kwargs.get("encoder_input_ids", kwargs.get("input_ids"))

        if encoder_input_ids is None:
            return []

        if not isinstance(encoder_input_ids, (torch.Tensor, list)):
            raise ValueError(
                f"Invalid encoder input_ids type: {type(encoder_input_ids)}"
            )

        if isinstance(encoder_input_ids, list):
            result = []
            for item in encoder_input_ids:
                if isinstance(item, torch.Tensor):
                    if item.dim() == 0:
                        item = item.unsqueeze(0)
                    result.append(item)
                else:
                    result.append(item)
            return result
        else:
            return encoder_input_ids.unsqueeze(1).unbind(dim=0)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: IntermediateTensors | None = None,
            inputs_embeds: torch.Tensor | None = None,
            encoder_outputs: torch.Tensor | None = None,
            **kwargs,
    ) -> torch.Tensor:
        if encoder_outputs is not None:
            encoder_outputs = torch.cat(encoder_outputs, dim=0)
        print("encoder_outputsencoder_outputs", encoder_outputs)
        return self.model(
            input_ids, positions, inputs_embeds, encoder_outputs=encoder_outputs
        )

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits


    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights_tuple_list = list(weights)

        shared_decoder_embedding_weight = None
        for name, loaded_weight in weights_tuple_list:
            if (
                    "model.decoder.embed_tokens.weight" in name
                    or "lm_head.weight" in name
            ):
                if shared_decoder_embedding_weight is not None:
                    continue
                shared_decoder_embedding_weight = loaded_weight

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["cls.", "pooler."]),
        )
        loaded_params = loader.load_weights(
            weights_tuple_list, mapper=self.hf_to_vllm_mapper
        )

        if shared_decoder_embedding_weight is not None:
            weight_loader = getattr(
                self.lm_head.weight, "weight_loader", default_weight_loader
            )
            weight_loader(self.lm_head.weight, shared_decoder_embedding_weight)

            self.model.decoder.embed_tokens.weight = self.lm_head.weight
            loaded_params.update(
                {
                    "lm_head.weight",
                    "model.decoder.embed_tokens.weight",
                }
            )

        return loaded_params