from typing import List

from transformers import BatchEncoding, BatchFeature
from transformers.processing_utils import ProcessorMixin
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast
import torch
import torch.nn.functional as F


class TextArtT5Processor(ProcessorMixin):
    attributes = ["tokenizer", "codebook_tokenizer"]

    def __init__(self,
                 tokenizer,
                 codebook_tokenizer=None,
                 codebook_size=None,
                 # image_processor=None,
                 **kwargs):
        if codebook_size is not None:
            print(f"Prioritizing NEW Codebook Tokenizer build (Size: {codebook_size})")
            codebook_tokenizer = self.build_codebook_tokenizer(codebook_size)
        elif codebook_tokenizer is not None:
            print("'codebook_size' not provided, loading existing Tokenizer...")
        else:
            raise ValueError("ScribeNet Crash: You forgot to provide both 'codebook_size' and 'codebook_tokenizer'!")
        super().__init__(
            tokenizer=tokenizer,
            codebook_tokenizer=codebook_tokenizer,
            # image_processor=image_processor,
            **kwargs
        )

        self.pad_token_id = self.codebook_tokenizer.pad_token_id
        self.bos_token_id = self.codebook_tokenizer.bos_token_id
        self.eos_token_id = self.codebook_tokenizer.eos_token_id

    def __call__(
            self,
            text: List[str] | torch.LongTensor | None = None,
            codebook: torch.LongTensor = None,
            images=None,
            return_tensors="pt",
            padding_side="right",
            padding=True,
            pre_codebook: bool = False,
            **kwargs
    ):
        if images is not None:
            raise NotImplementedError("Image processing is not implemented yet.")
        inputs = {}
        if text is not None:
            chat_format = [
                [{"caption": t}]
                for t in text
            ]
            text_inputs = self.tokenizer.apply_chat_template(
                chat_format,
                return_tensors=return_tensors,
                padding_side=padding_side,
                padding=padding,
                **kwargs
            )
            inputs["input_ids"] = text_inputs["input_ids"]
            inputs["attention_mask"] = text_inputs["attention_mask"]
        if codebook is not None:
            if not isinstance(codebook, torch.Tensor):
                raise TypeError("codebook must be torch.Tensor")

            if codebook.dim() != 2:
                raise ValueError("codebook must have shape [B, L]")

            B, L = codebook.shape
            device = codebook.device

            # add BOS/EOS
            bos = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)
            tensors_to_cat = [bos, codebook]

            if not pre_codebook:
                eos = torch.full((B, 1), self.eos_token_id, dtype=torch.long, device=device)
                tensors_to_cat.append(eos)

            codebook_input_ids = torch.cat(tensors_to_cat, dim=1)

            attention_mask = torch.ones_like(codebook_input_ids, dtype=torch.long)

            inputs["decoder_input_ids"] = codebook_input_ids
            inputs["decoder_attention_mask"] = attention_mask

        return BatchFeature(inputs)

    @staticmethod
    def build_codebook_tokenizer(codebook_size):

        vocab = {str(i): i for i in range(codebook_size)}

        special_tokens = ["[PAD]", "[UNK]", "[CODE_START]", "[CODE_END]"]
        for i, token in enumerate(special_tokens):
            vocab[token] = codebook_size + i

        base_tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        codebook_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=base_tokenizer,
            pad_token="[PAD]",
            bos_token="[CODE_START]",
            eos_token="[CODE_END]",
            unk_token="[UNK]",
        )

        return codebook_tokenizer
