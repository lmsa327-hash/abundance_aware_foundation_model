import math
from typing import Optional
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2ForSequenceClassification

from abundance_aware.src.models.AbundanceEncoder import AbundanceEncoder, AbundanceEncoderWithFiLM
from abundance_aware.src.models.GatedAbundance import GatedAbundanceFusion


class AbundanceAwareGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config, withFilm = False, gated = False):
        super().__init__(config)
        self.film = withFilm
        self.gated = gated
        if self.film:
            self.abundance_embedding = AbundanceEncoderWithFiLM(config.n_embd)
        else:
            self.abundance_embedding = AbundanceEncoder(config.n_embd)
        if gated:
            self.abundance_fusion = GatedAbundanceFusion(
                config.n_embd
            )

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # HF's from_pretrained re-runs the model's default weight
        # initialization on any parameter missing from the checkpoint
        # (i.e. abundance_embedding, since the original MGM checkpoint
        # has no such module) AFTER __init__ runs -- silently overwriting
        # the zero-init set inside AbundanceEncoder.__init__. Re-zero
        # explicitly, post-load, so the loaded model is guaranteed
        # numerically identical to plain MGM regardless of HF's internal
        # re-initialization order.
        model = super().from_pretrained(*args, **kwargs)
        nn.init.zeros_(model.abundance_embedding.mlp[-1].weight)
        nn.init.zeros_(model.abundance_embedding.mlp[-1].bias)
        return model
    def condition_on_abundance(self, abundances, embeddings):
        if self.film:
            gamma, beta = self.abundance_embedding(abundances)
            embeddings = gamma * embeddings + beta
        else:
            abundance_embeddings = self.abundance_embedding(abundances)
            embeddings = embeddings + abundance_embeddings
            #Does dynamically gating abundance information improve microbiome representation learning over direct abundance embedding?
            if self.gated:
                embeddings, gate = self.abundance_fusion(
                    embeddings,
                    abundance_embeddings
                )
        return embeddings
    def forward(self, input_ids=None, abundances: Optional[torch.Tensor] = None,
                attention_mask=None, labels=None, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids is required")
        inputs_embeds = self.transformer.wte(input_ids)
        if abundances is not None:
           inputs_embeds = self.condition_on_abundance(abundances, inputs_embeds)

        return super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels, **kwargs,
        )


class AbundanceAwareGPT2ForSequenceClassification(GPT2ForSequenceClassification):
    def __init__(self, config, withFilm = False):
        super().__init__(config)
        self.abundance_embedding = AbundanceEncoder(config.n_embd)
        self.film = withFilm
        if self.film:
            self.abundance_embedding = AbundanceEncoderWithFiLM(config.n_embd)
        else:
            self.abundance_embedding = AbundanceEncoder(config.n_embd)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)  # same re-init issue as above
        nn.init.zeros_(model.abundance_embedding.mlp[-1].weight)
        nn.init.zeros_(model.abundance_embedding.mlp[-1].bias)
        return model

    def condition_on_abundance(self, abundances, embeddings):
        if self.film:
            gamma, beta = self.abundance_embedding(abundances)
            embeddings = gamma * embeddings + beta
        else:
            embeddings = embeddings + self.abundance_embedding(abundances)
        return embeddings
    def forward(self, input_ids=None, abundances: Optional[torch.Tensor] = None,
                attention_mask=None, labels=None, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids is required")
        inputs_embeds = self.transformer.wte(input_ids)
        if abundances is not None:
            inputs_embeds = self.condition_on_abundance(abundances, inputs_embeds)
        return super().forward(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            labels=labels, **kwargs,
        )