# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torch import nn

from ...masking_utils import create_bidirectional_mask
from ...modeling_layers import GenericForSequenceClassification, GenericForTokenClassification
from ...modeling_outputs import BaseModelOutputWithPast, MaskedLMOutput
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ..bamba.modeling_bamba import apply_mask_to_padding_states
from ..lfm2.modeling_lfm2 import (
    Lfm2Attention,
    Lfm2DecoderLayer,
    Lfm2Model,
    Lfm2PreTrainedModel,
)
from .configuration_lfm2_bidirectional import Lfm2BidirectionalConfig


class Lfm2BidirectionalShortConv(nn.Module):
    def __init__(self, config: Lfm2BidirectionalConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.L_cache = config.conv_L_cache
        self.bias = config.conv_bias

        # centered (non-causal) padding so each position mixes its left and right neighbors
        self.conv = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=self.L_cache,
            groups=config.hidden_size,
            bias=self.bias,
            padding=self.L_cache // 2,
        )
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=self.bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=self.bias)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Zero out padding positions before the (non-causal) conv so pads do not leak into neighboring
        # real tokens; this makes padded batches match the unpadded forward. Disabled for checkpoints
        # trained without it (see `Lfm2BidirectionalConfig.conv_zero_padding`).
        if self.config.conv_zero_padding:
            hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        seqlen = hidden_states.shape[1]
        BCx = self.in_proj(hidden_states).transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)
        Bx = B * x

        conv_out = self.conv(Bx)[..., :seqlen]

        y = C * conv_out
        y = y.transpose(-1, -2).contiguous()
        return self.out_proj(y)


class Lfm2BidirectionalAttention(Lfm2Attention):
    def __init__(self, config: Lfm2BidirectionalConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.is_causal = False


class Lfm2BidirectionalDecoderLayer(Lfm2DecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if self.is_attention_layer:
            hidden_states, _ = self.self_attn(
                hidden_states=self.operator_norm(hidden_states),
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )
        else:
            hidden_states = self.conv(self.operator_norm(hidden_states), attention_mask=attention_mask)
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states


class Lfm2BidirectionalPreTrainedModel(Lfm2PreTrainedModel):
    config: Lfm2BidirectionalConfig
    _no_split_modules = ["Lfm2BidirectionalDecoderLayer"]
    # flash_attention_2 support is deferred; eager / sdpa only for now.
    _supports_flash_attn = False
    _can_record_outputs = {
        "hidden_states": Lfm2BidirectionalDecoderLayer,
        "attentions": Lfm2BidirectionalAttention,
    }


class Lfm2BidirectionalModel(Lfm2Model):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        bidirectional_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        # Conv layers consume the raw 2D padding mask (to optionally zero pad states); attention layers
        # consume the 4D bidirectional mask. Skip the conv mask in the single-token case (compile-friendly).
        conv_mask = attention_mask if inputs_embeds.shape[1] != 1 else None

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_mask = bidirectional_mask if self.config.layer_types[i] == "full_attention" else conv_mask
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                **kwargs,
            )

        hidden_states = self.embedding_norm(hidden_states)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class Lfm2BidirectionalForMaskedLM(Lfm2BidirectionalPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Lfm2BidirectionalConfig):
        super().__init__(config)
        self.model = Lfm2BidirectionalModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MaskedLMOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked);
            the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size, **kwargs)

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Lfm2BidirectionalForSequenceClassification(GenericForSequenceClassification, Lfm2BidirectionalPreTrainedModel):
    pass


class Lfm2BidirectionalForTokenClassification(GenericForTokenClassification, Lfm2BidirectionalPreTrainedModel):
    pass


__all__ = [
    "Lfm2BidirectionalForMaskedLM",
    "Lfm2BidirectionalForSequenceClassification",
    "Lfm2BidirectionalForTokenClassification",
    "Lfm2BidirectionalModel",
    "Lfm2BidirectionalPreTrainedModel",
]
