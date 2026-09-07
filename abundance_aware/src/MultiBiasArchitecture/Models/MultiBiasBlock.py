from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from abundance_aware.src.MultiBiasArchitecture.AbundanceAttention.BIas.AbundanceBias import ABUNDANCE_BIAS
from abundance_aware.src.MultiBiasArchitecture.HierarchyAttention.Bias.TaxonomyHierarchyBias import HIERARCHY_BIAS
from abundance_aware.src.MultiBiasArchitecture.Models.MultiBiasAttention import MultiBiasGPT2Attention
from abundance_aware.src.Abundance.models.AbundanceEncoder import AbundanceEncoderWithFiLM


class MultiBiasBlock(GPT2Block):
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

        self.attn = MultiBiasGPT2Attention(
            config,
            biases={
                ABUNDANCE_BIAS: dict(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    rank=16,
                    #encoder=AbundanceEncoder(config.n_embd),
                    encoder=AbundanceEncoderWithFiLM(config.n_embd),
                    use_gated_fusion=True,
                    init_scale=0.0,
                ),

            },
        )
        # Copy pretrained attention parameters
        #
        # strict=False is important because the new
        # modules don't exist in the old checkpoint.

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