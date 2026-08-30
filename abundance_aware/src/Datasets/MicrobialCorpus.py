from pickle import dump
from typing import Optional

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import pandas as pd
import numpy as np
import tqdm
import torch

from abundance_aware.src.tokenizers.MicrobialTokenizer import MicrobialTokenizer
from abundance_aware.src.utils.Utils import find_pkg_resource


class MicrobialCorpus(Dataset):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 data_path: Optional[str] = None,
                 abu: Optional[pd.DataFrame] = None,
                 phylogeny_path=None,
                 key='genus',
                 max_len=512,
                 preprocess=True):
        if phylogeny_path is None:
            phylogeny_path = find_pkg_resource("resources/phylogeny.csv")
        if data_path:
            file_type = data_path.split('.')[-1]
            if file_type not in ['h5', 'csv', 'tsv', 'txt']:
                raise ValueError('File type not supported.'
                                 'Please provide h5, csv, tsv or txt file.')
            if file_type == 'h5':
                self.data = pd.read_hdf(data_path, key=key).T
            else:
                sep = ',' if file_type == 'csv' else '\t'
                self.data = pd.read_csv(data_path, sep=sep, index_col=0).T
        elif abu is not None:
            self.data = abu
        else:
            raise ValueError('Please provide data_path or abu.')
        self.tokenizer = tokenizer
        self.phylogeny = pd.read_csv(phylogeny_path, index_col=0)
        self.max_len = max_len

        # add all zero row to save zero values
        self.data, self.zero_values = self._preprocess(self.data, preprocess)

        # convert to token
        tokens_list = []
        length_list = []

        for sample in tqdm(self.data.index):
            tokens, length = self._convert_to_token(self.data.loc[sample])
            tokens_list.append(tokens)
            length_list.append(length)

        # del self.data   # for saving memory
        print(f'Total {len(tokens_list)} samples.\n\
            Max length is {max(length_list)}.\n\
            Average length is {np.mean(length_list)}.\n\
            Min length is {min(length_list)}.')
        self.abundances = self.data
        self.tokens = torch.LongTensor(tokens_list)

    def __getitem__(self, index):
        attention_mask = torch.ones(self.tokens[index].shape)
        attention_mask[self.tokens[index] == self.tokenizer.pad_token_id] = 0
        tokens = self.tokens[index].clone()
        abundance =self.abundances[index].clone()
        return {'input_ids': torch.tensor(tokens),
                'attention_mask': attention_mask,
                "abundances": torch.tensor(abundance),
                }

    def __len__(self):
        return len(self.tokens)

    def _convert_to_token(self, sample):
        # set zero values to zero
        sample = sample[sample > self.zero_values]
        sample = sample.sort_values(ascending=False)
        sent = sample.index.tolist()
        # add bos
        sent = ['<bos>'] + sent
        # add eos
        sent = sent + ['<eos>']

        # convert to token
        tokens = self.tokenizer.encode(sent)
        length = len(tokens)

        # padding and truncate
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens.extend([self.tokenizer.pad_token_id] * (self.max_len - len(tokens)))

        return tokens, length

    def _preprocess(self, data, preprocess):
        # data.columns = data.columns.str.replace('; ', ';', regex=False) # remove space after ;
        # data.columns = data.columns.str.replace(';s__.*', '', regex=True) # drop species level
        # data.columns = data.columns.str.replace('^k__', 'sk__', regex=True) # if start with k__, replace with sk__
        # extract 'g__XXX' in the column names
        data.columns = data.columns.str.extract(r'(g__[A-Za-z0-9_]+)').squeeze()
        data = data.groupby(data.columns, axis=1).sum()
        before = data.shape[0]
        # only keep genus in phylogeny
        target_df = pd.DataFrame(index=self.phylogeny.index)
        data = target_df.merge(data.T, left_index=True, right_index=True, how='left').fillna(0).T
        # drop all zero rows
        data = data.loc[(data != 0).any(axis=1)]
        print(f'{before - data.shape[0]} samples are dropped for all zeroes')
        if not preprocess:
            return data, data.min(0)
        #optional log

        # relative abundance
        data = data.div(data.sum(axis=1), axis=0)
        # normalize
        data.loc['zero'] = 0  # save zero values
        data = (data - self.phylogeny['mean']) / self.phylogeny['std']
        zero_values = data.loc['zero']
        data = data.drop('zero')
        return data, zero_values



class SequenceClassificationDataset(Dataset):
    def __init__(self, seq, mask, abundances, labels):
        self.seq = seq
        self.mask = mask
        self.labels = labels
        self.abundances = abundances

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.seq[idx]),
            "attention_mask": torch.tensor(self.mask[idx]),
            "abundances": torch.tensor(self.abundances[idx]),
            "labels": torch.tensor(self.labels[idx])
        }


class MicrobialCorpusWithLabelTokens(Dataset):
    def __init__(self, tokens, abundances, labels, tokenizer):
        self.tokens = tokens
        self.tokenizer = tokenizer
        self.abundances = abundances
        self.labels = torch.tensor(self.tokenizer.encode(labels)).view(-1, 1)
        # insert label tokens after <bos>
        self.tokens = torch.cat((self.tokens[:, :1], self.labels, self.tokens[:, 1:-1]), dim=1)
        label_abundance = torch.zeros(
            self.abundances.size(0),
            1,
            device=self.abundances.device,
            dtype=self.abundances.dtype,
        )
        self.abundances = torch.cat(
            (
                self.abundances[:, :1],  # BOS abundance
                label_abundance,  # label abundance
                self.abundances[:, 1:-1],  # microbial abundances
            ),
            dim=1,
        )

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        attention_mask = torch.ones(self.tokens[idx].shape)
        attention_mask[self.tokens[idx] == self.tokenizer.pad_token_id] = 0
        tokens = self.tokens[idx].clone()
        abundance =self.abundances[idx].clone()

        return {'input_ids': torch.tensor(tokens),
                'attention_mask': attention_mask,
                'abundances': torch.tensor(abundance)}


if __name__ == '__main__':
    # create MicroCorpus using MGnify data
    special_toks = ['<pad>', '<mask>']
    abu = pd.read_hdf('data/abu_processed.h5', 'genus')
    genus_toks = abu.columns.tolist()
    toks = special_toks + genus_toks
    tokenizer = MicrobialTokenizer(toks)
    dump(tokenizer, open('MicrobialTokenizer.pkl', 'wb'))

    corpus = MicrobialCorpus(abu=abu, tokenizer=tokenizer, preprocess=False)

    dump(corpus, open('corpus/MicroCorpus_general_512.pkl', 'wb'))

    # human corpus
    meta = pd.read_csv('~/data5/download/MGnify/metadata.csv', index_col=0)
    meta = meta['Env'].str.split(':', expand=True)[1]
    meta = meta[meta == 'Host-associated']
    human_abu = abu.loc[abu.index.isin(meta.index)]
    human_corpus = MicrobialCorpus(abu=human_abu, tokenizer=tokenizer, preprocess=False)
    dump(human_corpus, open('corpus/MicroCorpus_human_512.pkl', 'wb'))

    # microbes = abu.columns.tolist()
    # key_list.extend(microbes)

    # # build token dict
    # token_dict = {}
    # for i, key in enumerate(key_list):
    #     token_dict[key] = i
    # dump(token_dict, open('token_dict.pkl', 'wb'))

    # # calculate none zero median value of each microbe
    # median_dict = {}
    # for microbe in microbes:
    #     median_dict[microbe] = abu[microbe].replace(0, np.nan).median()
    # dump(median_dict, open('median_dict.pkl', 'wb'))