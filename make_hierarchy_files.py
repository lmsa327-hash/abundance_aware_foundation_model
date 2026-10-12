import argparse
import json
import shutil
from pathlib import Path
from pickle import dump

import pandas as pd
import torch

from abundance_aware.src.Datasets.CustomUnPickler import CustomUnpickler
from abundance_aware.src.tokenizers.MicrobialTokenizer import MicrobialTokenizer
from abundance_aware.src.utils.Utils import find_pkg_resource, get_pkg_resource_path

TAXONOMIC_LEVELS = [
    "Kingdom",
    "Phylum",
    "Class",
    "Order",
    "Family",
    "Genus",
]
COLUMN_MAP = {
    "sk": "Kingdom",
    "p": "Phylum",
    "c": "Class",
    "o": "Order",
    "f": "Family",
    "g": "Genus",
}

def get_genus_hierarchy_from_file(taxonomy_file: Path, resources_dir: Path):
    df = pd.read_csv(
        taxonomy_file,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[""],          # only truly empty cells become NaN
    )
    df = df.rename(columns=COLUMN_MAP)
    df = df[TAXONOMIC_LEVELS].copy()
    #normalize
    for level in TAXONOMIC_LEVELS:
        df[level] = df[level].fillna("<UNK>").astype(str).str.strip()

    taxonomy_vocab = {}
    for level in TAXONOMIC_LEVELS:
        unique_values = sorted(
            df[level].unique().tolist()
        )
        # ID 0 is reserved for unknown
        vocab = {
            "<UNK>": 0
        }
        next_id = 1
        for value in unique_values:
            if value == "<UNK>":
                continue
            vocab[value] = next_id
            next_id += 1
        taxonomy_vocab[level] = vocab
    with open(
            resources_dir / "taxonomy_vocab.json",
            "w",
    ) as f:

        json.dump(
            taxonomy_vocab,
            f,
            indent=2,
        )
    genus_hierarchy = {}
    for genus, group in df.groupby("Genus"):
        if genus == "<UNK>":
            continue
        lineage_counts = (
            group[TAXONOMIC_LEVELS[:-1]]
            .value_counts()
        )

        most_common_lineage = lineage_counts.index[0]
        hierarchy = []
        for level, value in zip(
                TAXONOMIC_LEVELS[:-1],
                most_common_lineage,
        ):
            hierarchy.append(
                taxonomy_vocab[level][value]
            )

        # Add genus itself
        hierarchy.append(
            taxonomy_vocab["Genus"][genus]
        )

        genus_hierarchy[genus] = hierarchy
    return genus_hierarchy
def create_taxonomy_by_token_file(genus_hierarchy, tokenizer: MicrobialTokenizer, resources_dir: Path):
    vocab_size = len(tokenizer)

    taxonomy_by_token = torch.zeros(
        (vocab_size, len(TAXONOMIC_LEVELS)),
        dtype=torch.long,
    )

    for genus, hierarchy in genus_hierarchy.items():

        token = f"g__{genus}"

        token_id = tokenizer.convert_tokens_to_ids(token)

        if token_id is None:
            continue

        taxonomy_by_token[token_id] = torch.tensor(
            hierarchy,
            dtype=torch.long,
        )
    torch.save(
        taxonomy_by_token,
        resources_dir / "taxonomy_by_token.pt",
    )
    data = torch.load(resources_dir / "taxonomy_by_token.pt", map_location="cpu")
    print(data)
    return taxonomy_by_token
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=Path,
        help="Taxonomy file path e.g. 16S_asv_tax.csv",
    )
    args = parser.parse_args()
    resource_path = get_pkg_resource_path("resources")
    hierarchy = get_genus_hierarchy_from_file(args.input,resource_path)
    with open(find_pkg_resource("resources/MicrobialTokenizer.pkl"), "rb") as f:
        unpickler = CustomUnpickler(f)
        tokenizer = unpickler.load()
    create_taxonomy_by_token_file(hierarchy, tokenizer, resource_path)

if __name__ == "__main__":
    main()
