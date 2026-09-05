from argparse import ArgumentParser
from pathlib import Path
from pickle import dump
import shutil

import pandas as pd

from abundance_aware.src.tokenizers.MicrobialTokenizer import MicrobialTokenizer


def main():
    parser = ArgumentParser(description="Build a MicrobialTokenizer from abundance data.")
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the abundance HDF5 file, e.g. data/abundance_processed.h5",
    )

    parser.add_argument(
        "key",
        type=str,
        default="genus",
        help="Key for for abundance h5",
    )

    args = parser.parse_args()

    special_toks = ["<pad>", "<mask>"]

    abu = pd.read_hdf(args.input, args.key)
    genus_toks = abu.columns.tolist()

    toks = special_toks + genus_toks

    tokenizer = MicrobialTokenizer(toks)

    # Save to a temporary/local location
    output_file = Path("MicrobialTokenizer.pkl")
    with output_file.open("wb") as f:
        dump(tokenizer, f)

    # Copy into resources/
    resources_dir = Path("abundance_aware/resources")
    resources_dir.mkdir(parents=True, exist_ok=True)

    resources_file = resources_dir / "MicrobialTokenizer.pkl"
    shutil.copy2(output_file, resources_file)

    print(f"Tokenizer created: {resources_file}")


if __name__ == "__main__":
    main()

