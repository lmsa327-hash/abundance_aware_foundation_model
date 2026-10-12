import math

import torch
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from abundance_aware.src.MultiBiasArchitecture.common.ComputeBias import BIAS_REGISTRY
from abundance_aware.src.MultiBiasArchitecture.common.GatedFusion import GatedFusion


class _MultiBiasGatingMixin:
    """
    Post-softmax multiplicative gating, factored out so it's a composable
    toggle on top of EITHER attention-combination method below, rather
    than duplicated (and liable to drift out of sync) inside each one.
    Identical mechanism to the original MultiBiasGPT2Attention: a base
    gate conditioned on hidden_states alone, multiplied by one gate per
    active bias (each conditioned on hidden_states + that bias's encoded
    signal), applied to attn_output AFTER the attention/value matmul --
    this is a "how much to trust what was attended to" decision,
    independent of "who gets attended to" (that's compute_attention's job).
    """

    def _init_gating(self, hidden_size):
        self.base_post_gate = GatedFusion(hidden_size, uses_aux=False)
        self.last_gates = {}

    def _compute_gate(self, hidden_states, encoded_inputs):
        gate = self.base_post_gate(hidden_states)
        self.last_gates = {"base": gate.detach()}
        for name, encoded in encoded_inputs.items():
            _, g_k = self.biases[name].fuse_from_encoded(hidden_states, aux=encoded)
            gate = gate * g_k
            self.last_gates[name] = g_k.detach()
        return gate


class MultiBiasGPT2CombinedLogitAttention(GPT2Attention, _MultiBiasGatingMixin):

    def __init__(self, config, layer_idx=None, is_cross_attention=False, biases=None, use_gating=False):
        super().__init__(config)
        self.num_heads = config.num_attention_heads
        self.is_cross_attention = is_cross_attention
        self.layer_idx = layer_idx
        self.biases = nn.ModuleDict()
        self.bias_scales = nn.ParameterDict()
        self._bias_input_keys = {}
        #self.split_size = {}
        self.last_gates = {}
        self.use_gating = use_gating

        for name, cfg in (biases or {}).items():
            if name not in BIAS_REGISTRY:
                raise KeyError(f"No bias registered under '{name}'. Available: {list(BIAS_REGISTRY)}")
            cfg = dict(cfg)
            init_scale = cfg.pop("init_scale", 0.0)
            module = BIAS_REGISTRY[name](**cfg)
            self.biases[name] = module
            self.bias_scales[name] = nn.Parameter(torch.tensor(float(init_scale)))
            self._bias_input_keys[name] = module.input_key

        if self.use_gating:
            self._init_gating(config.hidden_size)

    def compute_attention(self, query, key,  **encoded_inputs):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(query.size(-1))

        for name, encoded in encoded_inputs.items():
            attn_weights = attn_weights + self.bias_scales[name] * self.biases[name].compute_from_encoded(encoded)

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

        encoded_inputs = {}
        for name, bias_module in self.biases.items():
            raw = raw_inputs.get(bias_module.input_key)
            #raw = raw_inputs.get(name)
            if raw is None:
                continue
            encoded = bias_module.encode(raw)
            encoded_inputs[name] = encoded

        gate = self._compute_gate(hidden_states, encoded_inputs) if self.use_gating else None

        query_states = self._split_heads(query_states, self.num_heads, self.head_dim)
        key_states = self._split_heads(key_states, self.num_heads, self.head_dim)
        value_states = self._split_heads(value_states, self.num_heads, self.head_dim)

        if gate is not None:
            gate = self._split_heads(gate, self.num_heads, self.head_dim)


        if layer_past is not None:
            past_key, past_value = layer_past
            key_states = torch.cat((past_key, key_states), dim=-2)
            value_states = torch.cat((past_value, value_states), dim=-2)
        present = (key_states, value_states) if use_cache else None

        attn_weights = self.compute_attention(
            query=query_states, key=key_states, **encoded_inputs
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value_states)
        if gate is not None:
            attn_output = attn_output * gate

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class MultiBiasGPT2IndependentSoftmaxAttention(GPT2Attention, _MultiBiasGatingMixin):
    """Independent softmax per bias, blended via a learned convex
    combination, with an optional post-softmax multiplicative gate on top."""

    def __init__(self, config, layer_idx=None, is_cross_attention=False,
                 biases=None, mixture_init_bias=-6.0, use_gating=True):
        super().__init__(config)
        self.num_heads = config.num_attention_heads
        self.is_cross_attention = is_cross_attention
        self.layer_idx = layer_idx
        self.use_gating = use_gating
        self.biases = nn.ModuleDict()
        self._bias_input_keys = {}

        for name, cfg in (biases or {}).items():
            if name not in BIAS_REGISTRY:
                raise KeyError(f"No bias registered under '{name}'. Available: {list(BIAS_REGISTRY)}")
            cfg = dict(cfg)
            cfg.pop("init_scale", None)
            module = BIAS_REGISTRY[name](**cfg)
            self.biases[name] = module
            self._bias_input_keys[name] = module.input_key

        n_biases = len(self.biases)
        init_logits = torch.full((1 + n_biases,), mixture_init_bias)
        init_logits[0] = 0.0
        self.mixture_logits = nn.Parameter(init_logits)
        self._bias_order = list(self.biases.keys())

        if self.use_gating:
            self._init_gating(config.hidden_size)

    def compute_attention(self, query, key, attention_mask=None, **encoded_inputs):
        base_logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.size(-1))
        if attention_mask is not None:
            base_logits = base_logits + attention_mask
        attn_0 = nn.functional.softmax(base_logits, dim=-1)

        per_bias_attn = [attn_0]
        for name in self._bias_order:
            encoded = encoded_inputs.get(name)
            if encoded is None:
                per_bias_attn.append(attn_0)
                continue
            bias_logits = self.biases[name].compute_from_encoded(encoded)
            if attention_mask is not None:
                bias_logits = bias_logits + attention_mask
            per_bias_attn.append(nn.functional.softmax(bias_logits, dim=-1))

        mixture_weights = nn.functional.softmax(self.mixture_logits, dim=0)
        stacked = torch.stack(per_bias_attn, dim=0)
        combined = (mixture_weights.view(-1, 1, 1, 1, 1) * stacked).sum(dim=0)
        return combined

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None, use_cache=None,
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
        encoded_inputs = {}
        for name, bias_module in self.biases.items():
            raw = raw_inputs.get(bias_module.input_key)
            if raw is None:
                continue
            encoded_inputs[name] = bias_module.encode(raw)

        gate = self._compute_gate(hidden_states, encoded_inputs) if self.use_gating else None

        query_states = self._split_heads(query_states, self.num_heads, self.head_dim)
        key_states = self._split_heads(key_states, self.num_heads, self.head_dim)
        value_states = self._split_heads(value_states, self.num_heads, self.head_dim)
        if gate is not None:
            gate = self._split_heads(gate, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key_states = torch.cat((past_key, key_states), dim=-2)
            value_states = torch.cat((past_value, value_states), dim=-2)
        present = (key_states, value_states) if use_cache else None

        attn_weights = self.compute_attention(
            query=query_states, key=key_states, attention_mask=attention_mask, **encoded_inputs
        )
        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value_states)
        if gate is not None:
            attn_output = attn_output * gate

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs
