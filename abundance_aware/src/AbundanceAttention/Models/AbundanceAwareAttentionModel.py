import math
from typing import Optional

import torch
from torch import nn
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2Model

from abundance_aware.src.common.ComputeBias import BIAS_REGISTRY, AttentionBias
from abundance_aware.src.models.AbundanceEncoder import AbundanceEncoder

class AbundanceAwareGPT2IndividualAttentionWeightsAttention(GPT2Attention):

    def __init__(self, config, layer_idx=None, is_cross_attention=False, biases=None):
        super().__init__(config)
        self.num_heads = config.num_attention_heads
        self.is_cross_attention = is_cross_attention
        self.layer_idx = layer_idx
        self.biases = nn.ModuleDict()
        self.bias_scales = nn.ParameterDict()
        self._bias_input_keys = {}

        for name, cfg in (biases or {}).items():
            if name not in BIAS_REGISTRY:
                raise KeyError(f"No bias registered under '{name}'. Available: {list(BIAS_REGISTRY)}")
            cfg = dict(cfg)
            init_scale = cfg.pop("init_scale", 0.0)
            module = BIAS_REGISTRY[name](**cfg)
            self.biases[name] = module
            self.bias_scales[name] = nn.Parameter(torch.tensor(float(init_scale)))
            self._bias_input_keys[name] = module.input_key

    def compute_attention(self, query, key, gated_contributions=None, **raw_inputs):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(query.size(-1))

        if gated_contributions:
            for name, contrib in gated_contributions.items():
                q_k = contrib * query
                k_k = contrib * key
                branch = torch.matmul(q_k, k_k.transpose(-1, -2)) / math.sqrt(query.size(-1))
                attn_weights = attn_weights + self.bias_scales[name] * branch

        for name, bias_module in self.biases.items():
            raw = raw_inputs.get(bias_module.input_key)
            if raw is None:
                continue
            attn_weights = attn_weights + self.bias_scales[name] * bias_module.compute(raw)

        return attn_weights

    def forward(self, hidden_states,
                layer_past=None,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None, use_cache=None,
                output_attentions=False, **kwargs):

        if hidden_states is None:
            raise ValueError("input_embeddings is required")

        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError("Cross-attention weights are missing.")
            query_states = self.q_attn(hidden_states)
            key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)

        raw_inputs = {k: kwargs.get(k) for k in self._bias_input_keys.values()}

        fused_stream = hidden_states
        gated_contributions = {}
        self.last_gates = {}
        any_fusion = False
        for name, bias_module in self.biases.items():
            raw = raw_inputs.get(bias_module.input_key)
            if raw is None or bias_module.fusion is None:
                continue
            fused, gate = bias_module.fuse(hidden_states, raw)  # independent: always hidden_states, not fused_stream
            gated_contributions[name] = fused - hidden_states  # isolate this bias's own term
            self.last_gates[name] = gate
            any_fusion = True

        if any_fusion:
            value_states = value_states * (hidden_states + sum(gated_contributions.values()))

        query_states = self._split_heads(query_states, self.num_heads, self.head_dim)
        key_states = self._split_heads(key_states, self.num_heads, self.head_dim)
        if any_fusion:
            query_states = query_states * (hidden_states + sum(gated_contributions.values()))
            key_states = key_states * (hidden_states + sum(gated_contributions.values()))
        value_states = self._split_heads(value_states, self.num_heads, self.head_dim)

        # contributions are hidden_size-dim (pre-head-split), same as Q/K/V before splitting — split them too
        gated_contributions = {
            name: self._split_heads(c, self.num_heads, self.head_dim)
            for name, c in gated_contributions.items()
        }

        if layer_past is not None:
            past_key, past_value = layer_past
            key_states = torch.cat((past_key, key_states), dim=-2)
            value_states = torch.cat((past_value, value_states), dim=-2)
        present = (key_states, value_states) if use_cache else None

        attn_weights = self.compute_attention(
            query=query_states, key=key_states,
            gated_contributions=gated_contributions, **raw_inputs
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class AbundanceAwareGPT2Attention(GPT2Attention):

    def __init__(self, config, layer_idx=None, is_cross_attention=False, biases=None):
        super().__init__(config)
        self.num_heads = config.num_attention_heads
        self.is_cross_attention = is_cross_attention
        self.layer_idx = layer_idx
        self.biases = nn.ModuleDict()
        self.bias_scales = nn.ParameterDict()
        self._bias_input_keys = {}

        for name, cfg in (biases or {}).items():
            if name not in BIAS_REGISTRY:
                raise KeyError(f"No bias registered under '{name}'. Available: {list(BIAS_REGISTRY)}")
            cfg = dict(cfg)
            init_scale = cfg.pop("init_scale", 0.0)
            module = BIAS_REGISTRY[name](**cfg)
            self.biases[name] = module
            self.bias_scales[name] = nn.Parameter(torch.tensor(float(init_scale)))
            self._bias_input_keys[name] = module.input_key

    def compute_attention(self, query, key, **raw_inputs):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(query.size(-1))

        for name, bias_module in self.biases.items():
            raw = raw_inputs.get(bias_module.input_key)
            if raw is None:
                continue
            attn_weights = attn_weights + self.bias_scales[name] * bias_module.compute(raw)

        return attn_weights

    def forward(self, hidden_states,
                layer_past=None,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None, use_cache=None,
                output_attentions=False, **kwargs):

        if hidden_states is None:
            raise ValueError("input_embeddings is required")

        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError("Cross-attention weights are missing.")
            query_states = self.q_attn(hidden_states)
            key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)

        raw_inputs = {k: kwargs.get(k) for k in self._bias_input_keys.values()}
        #encoded_inputs = {k: kwargs.get(k) for k in self._bias_input_keys.values()}

        # --- Gated fusion: pre-head-split, hidden_size-dim, one or more
        # biases chained sequentially into the value stream.
        # fused_stream = hidden_states
        # self.last_gates = {}
        # any_fusion = False
        # for name, bias_module in self.biases.items():
        #     raw = raw_inputs.get(bias_module.input_key)
        #     if raw is None or bias_module.fusion is None:
        #         continue
        #     fused_stream, gate = bias_module.fuse(fused_stream, raw)
        #     self.last_gates[name] = gate
        #     any_fusion = True
        #
        # if any_fusion:
        #     value_states = value_states * fused_stream

        #to make fusing independent
        gated_contributions = {}
        self.last_gates = {}
        any_fusion = False
        for name, bias_module in self.biases.items():
            raw = raw_inputs.get(bias_module.input_key)
            if raw is None or bias_module.fusion is None:
                continue
            fused, gate = bias_module.fuse(hidden_states, raw)
            gated_contributions[name] = fused - hidden_states
            self.last_gates[name] = gate
            any_fusion = True

        if any_fusion:
            value_states = value_states * (hidden_states + sum(gated_contributions.values()))


        query_states = self._split_heads(query_states, self.num_heads, self.head_dim)
        key_states = self._split_heads(key_states, self.num_heads, self.head_dim)
        if any_fusion:
            query_states = query_states * (hidden_states + sum(gated_contributions.values()))
            key_states = key_states * (hidden_states + sum(gated_contributions.values()))
        value_states = self._split_heads(value_states, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key_states = torch.cat((past_key, key_states), dim=-2)
            value_states = torch.cat((past_value, value_states), dim=-2)
        present = (key_states, value_states) if use_cache else None

        attn_weights = self.compute_attention(query=query_states, key=key_states, **raw_inputs)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
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

        self.attn = AbundanceAwareGPT2Attention(
            config,
            biases={
                "abundance": dict(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    rank=16,
                    encoder=AbundanceEncoder(config.n_embd),
                    use_gated_fusion=True,
                    init_scale=0.0,
                ),

            },
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


