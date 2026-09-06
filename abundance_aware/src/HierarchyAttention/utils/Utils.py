import json
from pathlib import Path

import pandas as pd
import torch

from abundance_aware.src.tokenizers.MicrobialTokenizer import MicrobialTokenizer

TAXONOMIC_LEVELS = [
    "Kingdom",
    "Phylum",
    "Class",
    "Order",
    "Family",
    "Genus",
]
def get_genus_hierarchy(taxonomy_file: Path, resources_dir: Path):
    df = pd.read_csv(taxonomy_file)
    df = df[TAXONOMIC_LEVELS].copy()
    # normalize
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


def tokenize_hierarchy(genus_hierarchy: dict, tokenizer: MicrobialTokenizer):
    genus_token_to_id = {}

    for genus in genus_hierarchy:

        token = f"g__{genus}"

        token_id = tokenizer.convert_tokens_to_ids(token)

        if token_id is None:
            raise ValueError(
                f"Genus {genus} is not in the MGM tokenizer vocabulary"
            )

        genus_token_to_id[genus] = token_id
    return genus_token_to_id


def get_taxonomy_by_token(genus_hierarchy, tokenizer: MicrobialTokenizer, resources_dir: Path):
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

    # taxonomy_ids = taxonomy_by_token[input_ids]
    return taxonomy_by_token


def load_taxonomy_by_token(resources_dir: Path):
    taxonomy_by_token = torch.load(
        resources_dir / "taxonomy_by_token.pt"
    )
    return taxonomy_by_token