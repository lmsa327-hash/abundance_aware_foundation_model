
from pickle import dump
import sys

from abundance_aware.src.Datasets.CustomUnPickler import CustomUnpickler
from abundance_aware.src.Datasets.MicrobialCorpus import MicrobialCorpus
from abundance_aware.src.utils.Utils import find_pkg_resource

def construct(cfg, args):
    with find_pkg_resource("resources/MicrobialTokenizer.pkl").open("rb") as f:
        unpickler = CustomUnpickler(f)
        tokenizer = unpickler.load()
        
    if not args.no_normalize:
        print("Your data will be normalized with the phylogeny mean and std. If you wish to use your own normalization, please use --no-normalize.")
    corpus = MicrobialCorpus(
        data_path=args.input,
        tokenizer=tokenizer,
        key=args.key,
        max_len=cfg.getint("construct", "max_len"),
        preprocess=not args.no_normalize,
    )
    dump(corpus, open(args.output, "wb"))
