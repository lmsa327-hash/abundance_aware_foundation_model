import math

import torch
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from abundance_aware.src.MultiBiasArchitecture.common.ComputeBias import BIAS_REGISTRY
from abundance_aware.src.MultiBiasArchitecture.common.GatedFusion import GatedFusion


class MultiBiasGPT2IndividualAttentionWeightsAttention(GPT2Attention):

    def __init__(self, config, layer_idx=None, is_cross_attention=False, biases=None):
        super().__init__(config)
        self.num_heads = config.num_attention_heads
        self.is_cross_attention = is_cross_attention
        self.layer_idx = layer_idx
        self.biases = nn.ModuleDict()
        self.bias_scales = nn.ParameterDict()
        self._bias_input_keys = {}
        self.split_size = {}
        self.last_gates = {}

        for name, cfg in (biases or {}).items():
            if name not in BIAS_REGISTRY:
                raise KeyError(f"No bias registered under '{name}'. Available: {list(BIAS_REGISTRY)}")
            cfg = dict(cfg)
            init_scale = cfg.pop("init_scale", 0.0)
            module = BIAS_REGISTRY[name](**cfg)
            self.biases[name] = module
            self.bias_scales[name] = nn.Parameter(torch.tensor(float(init_scale)))
            self._bias_input_keys[name] = module.input_key
        hidden_size = config.hidden_size

    def compute_attention(self, query, key,  **encoded_inputs):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(query.size(-1))

        for name, bias_module in self.biases.items():
            encoded = encoded_inputs.get(bias_module.input_key)
            if encoded is None:
                continue
            attn_weights = attn_weights + self.bias_scales[name] * bias_module.compute_from_encoded(encoded)

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
            # raw = raw_inputs.get(bias_module.input_key)
            raw = raw_inputs.get(name)
            if raw is None:
                continue
            encoded = bias_module.encode(raw)
            encoded_inputs[name] = encoded
            _, g_k = bias_module.fuse_from_encoded(hidden_states, aux=encoded)
            self.last_gates[name] = g_k.detach()

        query_states = self._split_heads(query_states, self.num_heads, self.head_dim)
        key_states = self._split_heads(key_states, self.num_heads, self.head_dim)
        value_states = self._split_heads(value_states, self.num_heads, self.head_dim)


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

class MultiBiasGPT2Attention(GPT2Attention):

    def __init__(self, config, layer_idx=None, is_cross_attention=False, biases=None):
        super().__init__(config)
        self.num_heads = config.num_attention_heads
        self.is_cross_attention = is_cross_attention
        self.layer_idx = layer_idx
        #self.biases = nn.ModuleDict()
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

            hidden_size = config.hidden_size
            self.last_gates = {}
            self.base_post_gate = GatedFusion(hidden_size, uses_aux=False)

    def compute_attention(self, query, key, **encoded_inputs):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(query.size(-1))

        for name, bias_module in self.biases.items():
            encoded = encoded_inputs.get(bias_module.input_key)
            if encoded is None:
                continue
            attn_weights = attn_weights + self.bias_scales[name] * bias_module.compute_from_encoded(encoded)

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
        # --- Post-SDPA multiplicative gate: combine base + per-bias gates
        # element-wise, independently (each conditioned on hidden_states + its
        # own encoded signal, not on each other).
        gate = self.base_post_gate(hidden_states)
        self.last_gates = {"base": gate.detach()}
        encoded_inputs = {}
        for name, bias_module in self.biases.items():
            #raw = raw_inputs.get(bias_module.input_key)
            raw = raw_inputs.get(name)
            if raw is None:
                continue
            encoded = bias_module.encode(raw)
            encoded_inputs[name] = encoded
            _, g_k = bias_module.fuse_from_encoded(hidden_states, aux=encoded)
            gate = gate * g_k
            self.last_gates[name] = g_k.detach()

        query_states = self._split_heads(query_states, self.num_heads, self.head_dim)
        key_states = self._split_heads(key_states, self.num_heads, self.head_dim)
        value_states = self._split_heads(value_states, self.num_heads, self.head_dim)
        gate = self._split_heads(gate, self.num_heads, self.head_dim)  # per-head gate

        if layer_past is not None:
            past_key, past_value = layer_past
            key_states = torch.cat((past_key, key_states), dim=-2)
            value_states = torch.cat((past_value, value_states), dim=-2)
        present = (key_states, value_states) if use_cache else None

        attn_weights = self.compute_attention(query=query_states, key=key_states, **encoded_inputs)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output * gate  # <-- post-SDPA gate applied here, per head, element-wise

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (
                attn_output,) if False else outputs  # unchanged from original if output_attentions branch existed
        if output_attentions:
            outputs += (attn_weights,)
        return outputs
