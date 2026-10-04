import torch


class MultiEmbeddingsDataCollator:
    def __init__(self, base_collator, conditioners, pad_token_id):
        self.base_collator = base_collator
        self.pad_token_id = pad_token_id
        self.extra_fields = {
            c.input_key: c.collate_strategy
            for c in conditioners.values()
            if c.collate_strategy is not None
        }

    def __call__(self, examples):
        extra = {}
        for key, strategy in self.extra_fields.items():
            values = [ex.pop(key) for ex in examples if key in ex]
            if not values:
                continue
            if strategy == "sequence":
                max_len = max(v.size(0) for v in values)
                padded = torch.zeros(len(values), max_len, *values[0].shape[1:])
                for i, v in enumerate(values):
                    padded[i, :v.size(0)] = v
                extra[key] = padded
            elif strategy == "fixed":
                extra[key] = torch.stack(values)
            else:
                raise ValueError(f"Unknown collate_strategy '{strategy}' for field '{key}'")

        batch = self.base_collator(examples)
        batch.update(extra)
        return batch