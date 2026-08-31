import math
from typing import Optional

import torch
from torch import nn
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2Model


from abundance_aware.src.AbundanceAttention.Models.AbundanceBias import AbundanceEncoder, HeadWiseAbundanceBias
from abundance_aware.src.AbundanceAttention.Models.GatedFusionModule import GatedAbundanceFusion
from abundance_aware.src.models.AbundanceAwareModel import AbundanceAwareGPT2LMHeadModel


class AbundanceAwareGPT2Attention(GPT2Attention):

    def __init__(self, config, layer_idx=None,is_cross_attention=False, gated=False):
        super().__init__(config)
        self.num_heads = config.num_attention_heads
        self.rank = config.rank
        self.gated=gated
        self.abundance_embedding = AbundanceEncoder(config.n_embd)
        self.abundance_bias = HeadWiseAbundanceBias(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            rank=16
        )
        self.abundance_scale = nn.Parameter(
            torch.tensor(0.0)
        )
        if gated:
            self.gated_fusion = GatedAbundanceFusion(hidden_size=config.hidden_size)
        # nn.init.zeros_(model.abundance_embedding.mlp[-1].weight)
        # nn.init.zeros_(model.abundance_embedding.mlp[-1].bias)

    def compute_attention(
            self,
            query,
            key,
            abundance_embeddings=None,
            fused_embeddings=None,
    ):
        if self.gated:
            if fused_embeddings is not None:
                query = fused_embeddings * query
                key = fused_embeddings * key

        attn_weights = torch.matmul(
            query,
            key.transpose(-1, -2)
        )

        attn_weights /= math.sqrt(query.size(-1))

        if abundance_embeddings is not None:
            bias = self.abundance_bias(
                abundance_embeddings
            )

            attn_weights += (
                    self.abundance_scale
                    * bias
            )

        return attn_weights
    def forward(self,
                hidden_states,
                layer_past=None,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                use_cache=None,
                output_attentions=False,
                **kwargs):
        if hidden_states is None:
            raise ValueError("input_embeddings is required")
        if encoder_hidden_states is not None:
            #cross attention
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "Cross-attention weights are missing."
                )
            query_states = self.q_attn(
                hidden_states
            )
            key_states, value_states = self.c_attn(
                encoder_hidden_states
            ).split(
                self.split_size,
                dim=2,
            )

            attention_mask = (
                encoder_attention_mask
            )

        else:
            #self attention
            query_states, key_states, value_states = self.c_attn(
                hidden_states
            ).split(
                self.split_size,
                dim=2,
            )

        #split into attention heads
        query_states = self._split_heads(
            query_states,
            self.num_heads,
            self.head_dim,
        )

        key_states = self._split_heads(
            key_states,
            self.num_heads,
            self.head_dim,
        )

        value_states = self._split_heads(
            value_states,
            self.num_heads,
            self.head_dim,
        )

        #handle cached
        if layer_past is not None:
            past_key, past_value = layer_past

            key_states = torch.cat(
                (past_key, key_states),
                dim=-2,
            )
            value_states = torch.cat(
                (past_value, value_states),
                dim=-2,
            )
        if use_cache:
            present = (
                key_states,
                value_states,
            )
        else:
            present = None
        abundances = kwargs.get("abundances", None)
        abundance_embeddings = None
        if abundances is not None:
            abundance_embeddings = self.abundance_embedding(abundances)

        #gate here
        fused_embeddings = None
        if self.gated:
            fused_embeddings, gates = self.gated_fusion(hidden_states, abundance_embeddings)
            value_states = value_states * fused_embeddings

        attn_weights = self.compute_attention(query=query_states, key=key_states, abundance_embeddings=abundance_embeddings, fused_embeddings=fused_embeddings)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        #softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout( attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value_states)

        #merge heads
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        attn_output = self.c_proj(attn_output)

        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output,present)
        if output_attentions:
            outputs += (
                attn_weights,
            )

        return outputs

class AbundanceAwareBlock(GPT2Block):
    def __init__(
            self,
            config,
            layer_idx=None,
    ):
        super().__init__(
            config,
            layer_idx=layer_idx,
        )

        # Replace standard attention

        old_attention = self.attn

        self.attn = (
            AbundanceAwareGPT2Attention(
                config,
                is_cross_attention=False,
                layer_idx=layer_idx,
            )
        )

        # Copy pretrained attention parameters
        #
        # strict=False is important because the new
        # abundance modules don't exist in the old checkpoint.

        self.attn.load_state_dict(
            old_attention.state_dict(),
            strict=False,
        )
    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        abundance=None,
        **kwargs
    ):

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        #attention
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            abundance=abundance,
        )
        attn_output = attn_outputs[0]
        #residual connection
        hidden_states = (attn_output + residual)

        #MLP
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = ( residual + feed_forward_hidden_states )

        outputs = (hidden_states,)

        if use_cache:
            outputs += ( attn_output[1],)
        if output_attentions:
            outputs += (
                attn_outputs[2 if use_cache else 1],
            )
        return outputs
class AbundanceAwareGPT2Model(GPT2Model):

    def __init__(self, config):

        super().__init__(config)

        old_blocks = self.h

        new_blocks = []

        for i, old_block in enumerate(
            old_blocks
        ):

            new_block = (
                AbundanceAwareBlock(
                    config,
                    layer_idx=i,
                )
            )
            new_block.load_state_dict(
                old_block.state_dict(),
                strict=False,
            )

            new_blocks.append(
                new_block
            )

        self.h = nn.ModuleList(
            new_blocks
        )

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            abundance=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        use_cache = (
            use_cache
            if use_cache is not None
            else self.config.use_cache
        )

        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids "
                "and inputs_embeds"
            )
        if input_ids is not None:

            input_shape = input_ids.size()

            input_ids = input_ids.view(
                -1,
                input_shape[-1],
            )

            batch_size = input_ids.shape[0]

        elif inputs_embeds is not None:

            input_shape = inputs_embeds.size()[:-1]

            batch_size = inputs_embeds.shape[0]

        else:

            raise ValueError(
                "You have to specify either "
                "input_ids or inputs_embeds"
            )

        #validate abundance
        if abundance is not None:

            if abundance.dim() != 2:
                raise ValueError(
                    "abundance must have shape "
                    "[batch_size, sequence_length]"
                )

            if abundance.shape[0] != batch_size:
                raise ValueError(
                    "Abundance batch dimension does "
                    "not match input_ids."
                )

            if abundance.shape[1] != input_shape[1]:
                raise ValueError(
                    "Abundance sequence length does "
                    "not match input_ids."
                )

            abundance = abundance.to(
                device=input_ids.device
                if input_ids is not None
                else inputs_embeds.device,
                dtype=torch.float32,
            )
        #embeddings
        if inputs_embeds is None:
            inputs_embeds = self.wte(
                input_ids
            )
        #position ids
        if position_ids is None:
            position_ids = torch.arange(
                0,
                input_shape[-1],
                dtype=torch.long,
                device=inputs_embeds.device,
            )

            position_ids = position_ids.unsqueeze(
                0
            )

        position_embeds = self.wpe(
            position_ids
        )
        #combine embeddings
        hidden_states = (
                inputs_embeds
                + position_embeds
        )
        #token type embeddings
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(
                -1,
                input_shape[-1],
            )

            token_type_embeds = self.wte(
                token_type_ids
            )

            hidden_states = (
                    hidden_states
                    + token_type_embeds
            )
        hidden_states = self.drop(
            hidden_states
        )
        head_mask = self.get_head_mask(
            head_mask,
            self.config.n_layer,
        )
        #attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.view(
                batch_size,
                -1,
            )

            attention_mask = attention_mask[:, None, None, :]

            attention_mask = attention_mask.to(
                dtype=hidden_states.dtype
            )

            attention_mask = (
                                     1.0 - attention_mask
                             ) * torch.finfo(
                hidden_states.dtype
            ).min
        #Encoder attention mask
        if (
                encoder_hidden_states is not None
                and encoder_attention_mask is not None
        ):
            encoder_attention_mask = (
                encoder_attention_mask.view(
                    batch_size,
                    -1,
                )
            )

            encoder_attention_mask = (
                encoder_attention_mask[:, None, None, :]
            )

            encoder_attention_mask = (
                                             1.0 - encoder_attention_mask
                                     ) * torch.finfo(
                hidden_states.dtype
            ).min
        if past_key_values is None:
            past_key_values = [
                                  None
                              ] * len(self.h)
        #transformer blocks
        presents = () if use_cache else None

        all_hidden_states = (
            ()
            if output_hidden_states
            else None
        )

        all_attentions = (
            ()
            if output_attentions
            else None
        )

        for i, (block, layer_past) in enumerate(
                zip(self.h, past_key_values)
        ):
            if output_hidden_states:
                all_hidden_states += (
                    hidden_states,
                )
            outputs = block(
                hidden_states,
                layer_past = layer_past,
                attention_mask = attention_mask,
                head_mask = head_mask[i],
                encoder_hidden_states = encoder_hidden_states,
                encoder_attention_mask = encoder_attention_mask,
                use_cache = use_cache,
                output_attentions = output_attentions,
                abundance = abundance,
            )
            hidden_states = outputs[0]
            if use_cache:
                presents += (
                    outputs[1],
                )
            if output_attentions:
                all_attentions += (
                    outputs[2 if use_cache else 1],
                )
        #final layer norm
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(
                *input_shape,
                -1,
            )
        if output_hidden_states:
            all_hidden_states += (
                    hidden_states,
                )
        #return
        if not return_dict:
            return tuple(
                v for v in [ hidden_states,
                             presents,
                             all_hidden_states,
                             all_attentions] if v is not None
            )
        from transformers.modeling_outputs import (
           BaseModelOutputWithPastAndCrossAttentions
        )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

class AbundanceAwareAttentionModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        old_transformer = self.transformer
        new_transformer = (
            AbundanceAwareGPT2Model(config)
        )
        #copy pretrained
        new_transformer.load_state_dict(
            old_transformer.state_dict(),
            strict=False,
        )
        self.transformer = new_transformer

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        nn.init.zeros_(model.transformer.model.abundance_embedding.mlp[-1].weight)
        nn.init.zeros_(model.transformer.model.abundance_embedding.mlp[-1].bias)
        return model

    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                abundance=None,
                **kwargs,
                ):
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            abundance=abundance,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = transformer_outputs[0]

        logits = self.lm_head(
            hidden_states
        )

        loss = None
        if labels is not None:
            shift_logits = logits[
                ...,
                :-1,
                :,
            ].contiguous()

            shift_labels = labels[
                ...,
                1:,
            ].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss()

            loss = loss_fct(
                shift_logits.view(
                    -1,
                    shift_logits.size(-1),
                ),
                shift_labels.view(-1),
            )
        if not return_dict:
            output = (
                         logits,
                     ) + transformer_outputs[1:]

            return (
                (loss,) + output
                if loss is not None
                else output
            )

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=(
                transformer_outputs.past_key_values
            ),
            hidden_states=(
                transformer_outputs.hidden_states
            ),
            attentions=(
                transformer_outputs.attentions
            ),
            cross_attentions=(
                transformer_outputs.cross_attentions
                if hasattr(
                    transformer_outputs,
                    "cross_attentions"
                )
                else None
            ),
        )



