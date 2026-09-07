import torch
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import  GPT2Model

from abundance_aware.src.MultiBiasArchitecture.AbundanceAttention.BIas.AbundanceBias import ABUNDANCE_BIAS
from abundance_aware.src.MultiBiasArchitecture.HierarchyAttention.utils.Utils import load_taxonomy_by_token
from abundance_aware.src.MultiBiasArchitecture.Models.MultiBiasBlock import MultiBiasBlock


class MultiBiasGPT2Model(GPT2Model):

    def __init__(self, config):

        super().__init__(config)

        old_blocks = self.h

        new_blocks = []

        for i, old_block in enumerate(
            old_blocks
        ):

            new_block = (
                MultiBiasBlock(
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
            **kwargs
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
        abundance = kwargs.get(ABUNDANCE_BIAS, None)
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
        #taxonomy_hierarchy
        tax_by_token = load_taxonomy_by_token()
        taxonomy_ids = tax_by_token[input_ids]
        if taxonomy_ids is not None:
            pass
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




