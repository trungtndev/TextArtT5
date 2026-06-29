import math
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedConfig, PreTrainedModel, GenerationMixin, M2M100ForConditionalGeneration, BartModel, \
    Qwen3VLForConditionalGeneration, BertModel, CLIPTextModel, BartForConditionalGeneration, T5ForConditionalGeneration, \
    T5EncoderModel, T5Config, GPT2PreTrainedModel

from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.masking_utils import create_bidirectional_mask, create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, is_torchdynamo_compiling, logging, \
    torch_compilable_check
from transformers.utils.output_capturing import OutputRecorder

logger = logging.get_logger(__name__)


class TextArtT5Config(PreTrainedConfig):
    model_type = "textart_t5"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
            self,
            encoder_vocab_size=115,  # text size
            vocab_size=260,  # codebook size=256
            codebook_size=256,
            max_position_embeddings=2048,

            encoder_model_id_or_path="aieng-lab/math_pretrained_bert_mamut",
            encoder_layers=2,
            encoder_ffn_dim=32,
            encoder_attention_heads=2,

            decoder_layers=2,
            decoder_ffn_dim=32,
            decoder_attention_heads=2,

            encoder_layerdrop=0.0,
            decoder_layerdrop=0.0,
            activation_function="gelu",
            d_model=16,
            dropout=0.1,
            attention_dropout=0.0,
            activation_dropout=0.0,
            init_std=0.02,
            use_cache=True,

            encoder_pad_token_id=0,
            encoder_unk_token_id=1,
            encoder_cls_token_id=2,
            encoder_sep_token_id=3,

            pad_token_id=256,
            unk_token_id=257,
            bos_token_id=258,
            eos_token_id=259,

            decoder_start_token_id=258,
            forced_eos_token_id=259,
            is_encoder_decoder=True,
            is_decoder=False,
            tie_word_embeddings=True,
            **kwargs,
    ):
        self.is_decoder = is_decoder
        self.tie_word_embeddings = tie_word_embeddings
        self.encoder_vocab_size = encoder_vocab_size
        self.vocab_size = vocab_size
        self.codebook_size = codebook_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers

        self.encoder_pad_token_id = encoder_pad_token_id
        self.encoder_unk_token_id = encoder_unk_token_id
        self.encoder_cls_token_id = encoder_cls_token_id
        self.encoder_sep_token_id = encoder_sep_token_id

        assert codebook_size == pad_token_id
        assert codebook_size + 1 == unk_token_id
        assert codebook_size + 2 == bos_token_id
        assert codebook_size + 3 == eos_token_id
        assert codebook_size + 2 == decoder_start_token_id
        assert decoder_start_token_id == bos_token_id
        assert forced_eos_token_id == eos_token_id
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.forced_eos_token_id = forced_eos_token_id
        self.encoder_model_id_or_path = encoder_model_id_or_path
        super().__init__(
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )


class TextArtT5LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)

    def forward(
            self, input_ids: torch.Tensor, past_key_values_length: int = 0, position_ids: torch.Tensor | None = None
    ):
        """`input_ids' shape is expected to be [bsz x seqlen]."""

        if position_ids is None:
            bsz, seq_len = input_ids.shape[:2]
            position_ids = torch.arange(
                past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
            ).expand(bsz, -1)
        else:
            position_ids = position_ids.unsqueeze(0)

        return super().forward(position_ids)


# Copied from transformers.models.bert.modeling_bert.eager_attention_forward
def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float | None = None,
        dropout: float = 0.0,
        **kwargs: Unpack[TransformersKwargs],
):
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class TextArtT5Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
            is_causal: bool = False,
            config: TextArtT5Config | None = None,
            layer_idx: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal
        self.layer_idx = layer_idx
        if layer_idx is None and self.is_decoder:
            logger.warning_once(
                f"Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not recommended and "
                "will lead to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: torch.Tensor | None = None,
            past_key_values: Cache | None = None,
            attention_mask: torch.Tensor | None = None,
            # TODO: we need a refactor so that the different attention modules can get their specific kwargs
            # ATM, we have mixed things encoder, decoder, and encoder-decoder attn
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        # determine input shapes
        input_shape = hidden_states.shape[:-1]

        hidden_shape = (*input_shape, -1, self.head_dim)

        # get query proj
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        is_updated = False
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_states from cache
                    curr_past_key_values = past_key_values.cross_attention_cache
                else:
                    curr_past_key_values = past_key_values.self_attention_cache
            else:
                curr_past_key_values = past_key_values

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_values is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_values.layers[self.layer_idx].keys
            value_states = curr_past_key_values.layers[self.layer_idx].values
        else:
            key_states = self.k_proj(current_states)
            value_states = self.v_proj(current_states)
            kv_shape = (*current_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(kv_shape).transpose(1, 2)
            value_states = value_states.view(kv_shape).transpose(1, 2)

            if past_key_values is not None:
                key_states, value_states = curr_past_key_values.update(key_states, value_states, self.layer_idx)
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention and isinstance(past_key_values, EncoderDecoderCache):
                    past_key_values.is_updated[self.layer_idx] = True
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class TextArtT5MLP(nn.Module):
    """Multi-Layer Perceptron (FFN) module"""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            activation_function: str = "gelu",
            dropout: float = 0.0,
            activation_dropout: float = 0.0,
            bias: bool = True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.activation_fn = ACT2FN[activation_function]
        self.dropout = dropout
        self.activation_dropout = activation_dropout

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states


class TextArtT5EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: TextArtT5Config, layer_idx: int | None = None):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = TextArtT5Attention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
            layer_idx=layer_idx,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.mlp = TextArtT5MLP(
            input_dim=self.embed_dim,
            hidden_dim=config.encoder_ffn_dim,
            output_dim=self.embed_dim,
            activation_function=config.activation_function,
            dropout=config.dropout,
            activation_dropout=config.activation_dropout,
        )
        self.ffn_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: torch.FloatTensor,
            **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # Pre-norm: apply LayerNorm before attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Pre-norm: apply LayerNorm before FFN
        residual = hidden_states
        hidden_states = self.ffn_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states


class TextArtT5DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: TextArtT5Config, layer_idx: int | None = None):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = TextArtT5Attention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
            layer_idx=layer_idx,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = TextArtT5Attention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
            layer_idx=layer_idx,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.mlp = TextArtT5MLP(
            input_dim=self.embed_dim,
            hidden_dim=config.decoder_ffn_dim,
            output_dim=self.embed_dim,
            activation_function=config.activation_function,
            dropout=config.dropout,
            activation_dropout=config.activation_dropout,
        )
        self.ffn_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            encoder_hidden_states: torch.Tensor | None = None,
            encoder_attention_mask: torch.Tensor | None = None,
            past_key_values: Cache | None = None,
            use_cache: bool | None = True,
            **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # Pre-norm: Self Attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            hidden_states, _ = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

        # Pre-norm: Fully Connected
        residual = hidden_states
        hidden_states = self.ffn_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        return hidden_states


class TextArtT5PreTrainedModel(PreTrainedModel):
    config: TextArtT5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = ["encoder.version", "decoder.version"]
    _no_split_modules = [r"TextArtT5EncoderLayer", r"TextArtT5DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True

    def _init_weights(self, module):
        super()._init_weights(module)

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


class TextArtT5WrapperTextEncoder(TextArtT5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = CLIPTextModel.from_pretrained(config.encoder_model_id_or_path)
        # self.proj = nn.Linear(self.model.config.hidden_size, config.hidden_size)
        # self.norm = nn.LayerNorm(config.hidden_size)

        # for p in self.model.parameters():
        #     p.requires_grad = False
        # self.model.eval()

    # def train(self, mode: bool = True):
    #     super().train(mode)
    #     self.model.train(False)
    #     return self

    def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            # token_type_ids: torch.Tensor | None = None,
            return_dict: bool | None = None,
            **kwargs,
    ) -> tuple | BaseModelOutput:
        # with torch.no_grad():
        #     outputs = self.model(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         # token_type_ids=token_type_ids,
        #         # **kwargs,
        #     )
        #     last_hidden_state = outputs.last_hidden_state
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = outputs.last_hidden_state
        # last_hidden_state = self.proj(last_hidden_state)
        # last_hidden_state = self.norm(last_hidden_state)

        if not return_dict:
            return tuple(v for v in [last_hidden_state, None, None] if v is not None)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state, hidden_states=None, attentions=None
        )


class TextArtT5TextEncoder(TextArtT5PreTrainedModel):
    _can_record_outputs = {
        "hidden_states": TextArtT5EncoderLayer,
        "attentions": TextArtT5Attention,
    }
    def __init__(self, config: TextArtT5Config):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.encoder_pad_token_id
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = nn.Embedding(config.encoder_vocab_size, embed_dim,)
        self.embed_positions = TextArtT5LearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )

        self.layers = nn.ModuleList(
            [TextArtT5EncoderLayer(config, layer_idx=i) for i in range(config.encoder_layers)])
        self.final_layer_norm = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            **kwargs,
    ) -> BaseModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        embed_pos = self.embed_positions(inputs_embeds)
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        for idx, encoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if not to_drop:
                hidden_states = encoder_layer(
                    hidden_states,
                    attention_mask,
                    **kwargs,
                )

        hidden_states = self.final_layer_norm(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )

class TextArtT5CodeBookDecoder(TextArtT5PreTrainedModel):

    _can_record_outputs = {
        "hidden_states": TextArtT5DecoderLayer,
        "attentions": OutputRecorder(TextArtT5Attention, index=1, layer_name="self_attn"),
        "cross_attentions": OutputRecorder(TextArtT5Attention, index=1, layer_name="encoder_attn"),
    }

    def __init__(self, config: TextArtT5Config):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model,)
        self.embed_positions = TextArtT5LearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )

        self.layers = nn.ModuleList(
            [TextArtT5DecoderLayer(config, layer_idx=i) for i in range(config.decoder_layers)])

        self.final_layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            encoder_hidden_states: torch.FloatTensor | None = None,
            encoder_attention_mask: torch.LongTensor | None = None,
            past_key_values: Cache | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            use_cache: bool | None = None,
            **kwargs,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # initialize `past_key_values`
        if use_cache and past_key_values is None:
            past_key_values = (
                EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))
                if encoder_hidden_states is not None or self.config.is_encoder_decoder
                else DynamicCache(config=self.config)
            )

        batch_size, seq_length = inputs_embeds.size()[:-1]
        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(seq_length, device=inputs_embeds.device) + past_key_values_length

        if attention_mask is None and not is_torchdynamo_compiling():
            # required mask seq length can be calculated via length of past cache
            mask_seq_length = past_key_values_length + seq_length
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        self_attn_cache = (
            past_key_values.self_attention_cache
            if isinstance(past_key_values, EncoderDecoderCache)
            else past_key_values
        )

        attention_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=self_attn_cache,
        )
        encoder_attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
        )

        # embed positions
        positions = self.embed_positions(input_ids, past_key_values_length, position_ids=position_ids)
        positions = positions.to(inputs_embeds.device)

        hidden_states = inputs_embeds + positions

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        hidden_states = self.final_layer_norm(hidden_states)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class TextArtT5Model(TextArtT5PreTrainedModel):

    def __init__(self, config: TextArtT5Config):
        super().__init__(config)

        self.encoder = TextArtT5TextEncoder(config)
        # self.encoder = TextArtT5WrapperTextEncoder(config)
        self.decoder = TextArtT5CodeBookDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            decoder_input_ids: torch.LongTensor | None = None,
            decoder_attention_mask: torch.LongTensor | None = None,
            encoder_outputs: list[torch.FloatTensor] | None = None,
            past_key_values: Cache | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            decoder_inputs_embeds: torch.FloatTensor | None = None,
            use_cache: bool | None = None,
            **kwargs,
    ) -> tuple | Seq2SeqModelOutput:
        if encoder_outputs is None:
            encoder_outputs: BaseModelOutput = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        elif not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        print("encoder_outputs.last_hidden_state", encoder_outputs.last_hidden_state)
        decoder_outputs: BaseModelOutputWithPastAndCrossAttentions = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class TextArtT5ForConditionalGeneration(TextArtT5PreTrainedModel, GenerationMixin):
    base_model_prefix = "model"
    _tied_weights_keys = {
        "lm_head.weight": "model.decoder.embed_tokens.weight",
    }

    def __init__(self, config: TextArtT5Config):
        super().__init__(config)
        self.model = TextArtT5Model(config)
        self.lm_head = nn.Linear(config.d_model, self.config.vocab_size, bias=True)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            decoder_input_ids: torch.LongTensor | None = None,
            decoder_attention_mask: torch.LongTensor | None = None,
            encoder_outputs: list[torch.FloatTensor] | None = None,
            past_key_values: Cache | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            decoder_inputs_embeds: torch.FloatTensor | None = None,
            labels: torch.LongTensor | None = None,
            use_cache: bool | None = None,
            **kwargs,
    ) -> tuple | Seq2SeqLMOutput:

        outputs: Seq2SeqModelOutput = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            # **kwargs,
        )

        lm_logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size).float(), labels.reshape(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


__all__ = [
    "TextArtT5ForConditionalGeneration",
    "TextArtT5Model",
    "TextArtT5PreTrainedModel",
]
