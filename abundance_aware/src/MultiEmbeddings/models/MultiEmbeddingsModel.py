
from transformers import GPT2LMHeadModel, GPT2ForSequenceClassification
from abundance_aware.src.MultiEmbeddings.models.EmbeddingConditioner import ConditionedEmbeddingsMixin

# --------------------------------------------------------------------------
# The two model classes collapse to this — no more duplication
# --------------------------------------------------------------------------
class MultiEmbeddingsGPT2LMHeadModel(ConditionedEmbeddingsMixin, GPT2LMHeadModel):
    def __init__(self, config, conditioners=None):
        super().__init__(config)
        conditioners = conditioners or getattr(config, "conditioners_spec", None) or {} #(conditioner spec via ini )
        config.conditioners_spec = conditioners  # plain dict -> round-trips via config.json
        self._init_conditioners(conditioners, config.n_embd)

    @classmethod
    def from_pretrained(cls, *args, conditioners=None, **kwargs):
        # explicit conditioners= wins; otherwise pulled from the saved config automatically
        model = super().from_pretrained(*args, conditioners=conditioners, **kwargs)
        model.zero_init_conditioners_on_load()
        return model

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids is required")
        inputs_embeds = self.transformer.wte(input_ids)
        # pop every consumed raw input so it isn't also forwarded to the HF
        # base forward() as an unrecognized kwarg
        consumed = {c.input_key for c in self.conditioners.values() if c.input_key != "input_ids"}
        for key in consumed:
            kwargs.pop(key, None)
        inputs_embeds = self.apply_conditioning(inputs_embeds, input_ids,
                                                 **{k: v for k, v in list(kwargs.items())})
        return super().forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                labels=labels, **kwargs)


class MultiEmbeddingsGPT2ForSequenceClassification(ConditionedEmbeddingsMixin, GPT2ForSequenceClassification):
    def __init__(self, config, conditioners=None):
        super().__init__(config)
        self._init_conditioners(conditioners, config.n_embd)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        model.zero_init_conditioners_on_load()
        return model

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids is required")
        inputs_embeds = self.transformer.wte(input_ids)
        raw_for_conditioning = {k: kwargs[k] for k in
                                 (c.input_key for c in self.conditioners.values()) if k in kwargs}
        for key in raw_for_conditioning:
            kwargs.pop(key, None)
        inputs_embeds = self.apply_conditioning(inputs_embeds, input_ids, **raw_for_conditioning)
        return super().forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                labels=labels, **kwargs)

    # model = MultiEmbeddingsGPT2LMHeadModel(config, conditioners={
    #     "abundance": dict(use_gated_fusion=True, fusion_init_bias=-6.0),
    #     "hierarchy": dict(use_gated_fusion=True, fusion_init_bias=-6.0),
    # })