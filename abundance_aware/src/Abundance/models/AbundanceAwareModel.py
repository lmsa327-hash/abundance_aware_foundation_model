from typing import Optional
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2ForSequenceClassification

from abundance_aware.src.Abundance.models.AbundanceEncoder import AbundanceEncoder, AbundanceEncoderWithFiLM
from abundance_aware.src.Abundance.models.GatedAbundance import GatedAbundanceFusion, GatedHierarchyFusion
from abundance_aware.src.MultiBiasArchitecture.HierarchyAttention.Bias.TaxonomyHierarchyEncoder import \
    TaxonomyHierarchyEncoder
from abundance_aware.src.MultiBiasArchitecture.HierarchyAttention.utils.Utils import load_taxonomy_by_token


class AbundanceAwareGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config,
                 hierarchy = False,
                 withFilm = False,
                 gated = False):
        super().__init__(config)
        self.hierarchy = hierarchy
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
            self.hierarchy_fusion = GatedHierarchyFusion(config.n_embd)
        if hierarchy:
            self.hierarchy_encoder = TaxonomyHierarchyEncoder(config.n_embd)

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
    def condition_on_hierarchy(self, embeddings, input_ids):
        # taxonomy_hierarchy
        tax_by_token = load_taxonomy_by_token()
        taxonomy_ids = tax_by_token[input_ids]
        hierarchy_embeddings = self.hierarchy_encoder(taxonomy_ids)
        embeddings = embeddings + hierarchy_embeddings
        if self.gated:
            embeddings, gate = self.hierarchy_fusion(
                embeddings,
                hierarchy_embeddings
            )
        return embeddings
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                **kwargs):
        if input_ids is None:
            raise ValueError("input_ids is required")
        inputs_embeds = self.transformer.wte(input_ids)
        abundances = kwargs.pop("abundances", None)
        if abundances is not None:
            inputs_embeds = self.condition_on_abundance(abundances, inputs_embeds)
        if self.hierarchy:
            inputs_embeds = self.condition_on_hierarchy(inputs_embeds, input_ids)

        return super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels, **kwargs,
        )


class AbundanceAwareGPT2ForSequenceClassification(GPT2ForSequenceClassification):
    def __init__(self, config, withFilm = False, hierarchy = False, gated=False):
        super().__init__(config)
        self.abundance_embedding = AbundanceEncoder(config.n_embd)
        self.film = withFilm
        self.hierarchy = hierarchy
        if self.film:
            self.abundance_embedding = AbundanceEncoderWithFiLM(config.n_embd)
        else:
            self.abundance_embedding = AbundanceEncoder(config.n_embd)
        if gated:
            self.abundance_fusion = GatedAbundanceFusion(
                config.n_embd
            )
            self.hierarchy_fusion = GatedHierarchyFusion(config.n_embd)
        if hierarchy:
            self.hierarchy_encoder = TaxonomyHierarchyEncoder(config.n_embd)

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
            abundance_embeddings = self.abundance_embedding(abundances)
            embeddings = embeddings + abundance_embeddings
            # Does dynamically gating abundance information improve microbiome representation learning over direct abundance embedding?
            if self.gated:
                embeddings, gate = self.abundance_fusion(
                    embeddings,
                    abundance_embeddings
                )
        return embeddings

    def condition_on_hierarchy(self, embeddings, input_ids):
        # taxonomy_hierarchy
        tax_by_token = load_taxonomy_by_token()
        taxonomy_ids = tax_by_token[input_ids]
        hierarchy_embeddings = self.hierarchy_encoder(taxonomy_ids)
        embeddings = embeddings + hierarchy_embeddings
        if self.gated:
            embeddings, gate = self.hierarchy_fusion(
                embeddings,
                hierarchy_embeddings
            )
        return embeddings
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                **kwargs):
        if input_ids is None:
            raise ValueError("input_ids is required")
        inputs_embeds = self.transformer.wte(input_ids)
        abundances = kwargs.get("abundances", None)
        if abundances is not None:
            inputs_embeds = self.condition_on_abundance(abundances, inputs_embeds)
        if self.hierarchy:
            inputs_embeds = self.condition_on_hierarchy(inputs_embeds, input_ids)

        return super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels, **kwargs,
        )