from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from abundance_aware.src.MultiBiasArchitecture.Models.MultiBiasAttention import MultiBiasGPT2Attention


class MultiBiasBlock(GPT2Block):
    def __init__(
            self,
            config,
            bias_specs,
            layer_idx=None,
    ):
        super().__init__(
            config,
            layer_idx=layer_idx,
        )

        # Replace standard attention

        old_attention = self.attn
        self.attn = MultiBiasGPT2Attention(config, biases=bias_specs)
        self.attn.load_state_dict(old_attention.state_dict(), strict=False)

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
            **kwargs
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