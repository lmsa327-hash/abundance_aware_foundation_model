import pickle
import torch
import torch.nn as nn
from abundance_aware.src.tokenizers import MicrobialTokenizer


class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'MicrobialTokenizer':
            return MicrobialTokenizer
        return super().find_class(module, name)