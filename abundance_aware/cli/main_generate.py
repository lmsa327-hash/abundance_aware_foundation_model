import  pandas as pd
import torch
from pickle import load, dump

from abundance_aware.src.Datasets.CustomUnPickler import CustomUnpickler
from abundance_aware.src.Abundance.models.AbundanceAwareModel import AbundanceAwareGPT2LMHeadModel
from abundance_aware.src.utils.TrainingUtils import gen_num_sent
from abundance_aware.src.utils.Utils import find_pkg_resource


def generate(cfg, args):
    with find_pkg_resource("resources/MicrobialTokenizer.pkl").open("rb") as f:
        unpickler = CustomUnpickler(f)
        tokenizer = unpickler.load()
    extended_tokenizer = load(open(f"{args.model}/tokenizer.pkl", "rb"))
    model = AbundanceAwareGPT2LMHeadModel.from_pretrained(args.model)
    bad_words = set(extended_tokenizer.vocab.values()) - set(tokenizer.vocab.values())
    bad_words = [[word] for word in bad_words]
    
    if args.prompt is not None:
        prompt = pd.read_csv(args.prompt, sep='\t', header=None).values
        sent = [['<bos>', disease[0]] for disease in prompt]
    else:
        print("No prompt provided. Generating random sentences.")
        sent = [['<bos>']]
        
    start = [extended_tokenizer.encode(sent, return_tensors='pt') for sent in sent]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    gen_sent = gen_num_sent(start,
                            model,
                            num_sent=args.num_samples,
                            tokenizer=extended_tokenizer,
                            bad_words=bad_words) 
    
    dump(gen_sent, open(args.output, "wb"))
