import pickle
from abundance_aware.src.tokenizers.MicrobialTokenizer import MicrobialTokenizer


class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'MicrobialTokenizer':
            return MicrobialTokenizer
        return super().find_class(module, name)