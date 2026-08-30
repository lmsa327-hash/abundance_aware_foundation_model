
from transformers import PreTrainedTokenizer
from typing import Optional, List, Dict, Union, Tuple

class MicrobialTokenizer(PreTrainedTokenizer):
    def __init__(self, toks, **kwargs):

        # self.unk_token =

        self.toks = toks
        self.vocab = {v: i for i, v in enumerate(self.toks)}
        self.ids_to_tokens = {i: v for i, v in enumerate(self.toks)}
        super(MicrobialTokenizer, self).__init__(**kwargs)
        self.add_special_tokens({'pad_token': '<pad>', 'mask_token': '<mask>', 'bos_token': '<bos>', 'eos_token': '<eos>', 'unk_token':'<unk>'})

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

    def _add_tokens(self, new_tokens: List[str], special_tokens: bool = False) -> int:
        self.toks.extend(new_tokens)
        self.vocab = {v: i for i, v in enumerate(self.toks)}
        self.ids_to_tokens = {i: v for i, v in enumerate(self.toks)}

    def _convert_token_to_id(self, token):
        # print(token in self.vocab)
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
