from argparse import ArgumentParser
from pathlib import Path
from pickle import dump, load
import pandas as pd

from abundance_aware.src.tokenizers.MicrobialTokenizer import MicrobialTokenizer, SPECIALS


def main():
    parser = ArgumentParser(description="Build a MicrobialTokenizer from abundance data.")
    parser.add_argument("input", type=Path, help="Path to the abundance HDF5 file")
    parser.add_argument("key", type=str, nargs="?", default="genus",
                        help="Key for abundance h5 (default: genus)")
    args = parser.parse_args()

    abu = pd.read_hdf(args.input, args.key)

    # genus tokens only; the class owns the special tokens
    genus_toks = [t for t in abu.columns.tolist() if t not in SPECIALS]
    assert len(genus_toks) == len(set(genus_toks)), "Duplicate genus names in columns"

    tokenizer = MicrobialTokenizer(genus_toks)

    # sanity checks before saving
    n = len(genus_toks)
    assert tokenizer.toks[:n] == genus_toks
    assert tokenizer.toks[n:] == SPECIALS
    assert tokenizer.vocab_size == n + len(SPECIALS)
    assert tokenizer.pad_token_id == n  # pad is first special, right after genera

    resources_dir = Path("abundance_aware/resources")
    resources_dir.mkdir(parents=True, exist_ok=True)
    out = resources_dir / "MicrobialTokenizer.pkl"
    with out.open("wb") as f:
        dump(tokenizer, f)

    # round-trip check
    with out.open("rb") as f:
        reloaded = load(f)
    assert reloaded.vocab == tokenizer.vocab, "Vocab changed after pickle round trip"

    print(f"Tokenizer created: {out}  (vocab size {tokenizer.vocab_size}, pad id {tokenizer.pad_token_id})")


if __name__ == "__main__":
    main()