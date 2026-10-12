
from transformers import PreTrainedTokenizer
from typing import List

SPECIALS = ["<pad>", "<mask>", "<bos>", "<eos>", "<unk>"]

class MicrobialTokenizer(PreTrainedTokenizer):
    def __init__(self, toks, **kwargs):
        base = [t for t in dict.fromkeys(toks) if t not in SPECIALS]
        self.toks = base + SPECIALS
        self.vocab = {v: i for i, v in enumerate(self.toks)}
        self.ids_to_tokens = {i: v for i, v in enumerate(self.toks)}
        super().__init__(
            pad_token="<pad>", mask_token="<mask>", bos_token="<bos>",
            eos_token="<eos>", unk_token="<unk>", **kwargs,
        )

    def __setstate__(self, state):
        scratch = MicrobialTokenizer(['<pad>', '<mask>', '<bos>', '<eos>', '<unk>'])
        self.__dict__.update(scratch.__dict__)
        old_toks = state.get('toks', [])
        if old_toks:
            specials = ['<pad>', '<mask>', '<bos>', '<eos>', '<unk>']
            base_toks = [t for t in old_toks if t not in specials]
            full_toks = base_toks + specials

            self.toks = full_toks
            self.vocab = {v: i for i, v in enumerate(full_toks)}
            self.ids_to_tokens = {i: v for i, v in enumerate(full_toks)}

            old_vocab = {v: i for i, v in enumerate(old_toks)}
            assert self.vocab == old_vocab, (
                "Vocab mismatch after rebuilding from pickle — "
                "ids would not match the original tokenizer."
            )

    def _tokenize(self, text):
        return list(text)

    def _add_tokens(self, new_tokens, special_tokens=False):
        added = 0
        for t in new_tokens:
            t = str(t)
            if t not in self.vocab:
                self.vocab[t] = len(self.toks)
                self.ids_to_tokens[self.vocab[t]] = t
                self.toks.append(t)
                added += 1
        return added

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens[index]

    def get_vocab(self):
        return self.vocab

    def get_vocab_size(self):
        return len(self.vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)
