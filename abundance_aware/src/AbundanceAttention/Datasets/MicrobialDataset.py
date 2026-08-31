"""
Data pipeline for the abundance-aware microbial transformer.

Expected raw sample format (before this module touches it):
    a sample = list of (genus_name: str, relative_abundance: float) pairs,
    relative abundances summing to ~1.0 (or close, after any filtering).

This mirrors how you'd load MGnify/Atlas/Microcorpus after their own
taxonomic-profiling step -- this module does NOT do taxonomic assignment,
it assumes you already have per-sample (genus, abundance) tables (e.g.
from a MetaPhlAn/QIIME2/MGnify pipeline export) and just handles
vocabulary building, ranking, transformation, and batching.

Includes a synthetic data generator so the whole pipeline (tokenize ->
train -> attribute) can be validated on your GPU before you've pulled a
real MGnify/Atlas subset.
"""
import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

PAD_TOKEN = "<pad>"
PAD_IDX = 0


class GenusVocab:
    def __init__(self):
        self.token2id = {PAD_TOKEN: PAD_IDX}
        self.id2token = {PAD_IDX: PAD_TOKEN}

    def build(self, samples: List[List[Tuple[str, float]]]):
        genera = sorted({genus for sample in samples for genus, _ in sample})
        for g in genera:
            if g not in self.token2id:
                idx = len(self.token2id)
                self.token2id[g] = idx
                self.id2token[idx] = g
        return self

    def __len__(self):
        return len(self.token2id)

    def encode(self, genus: str) -> int:
        return self.token2id.get(genus, PAD_IDX)


def transform_abundance(x: float) -> float:
    """log1p transform. Swap for a CLR transform if you're working with
    compositional data that needs it (requires a pseudocount for zeros,
    which log1p sidesteps by construction)."""
    return math.log1p(x)


@dataclass
class MicrobialDataset(Dataset):
    samples: List[List[Tuple[str, float]]]
    vocab: GenusVocab
    max_len: int = 512

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = sorted(self.samples[idx], key=lambda p: p[1], reverse=True)
        sample = sample[: self.max_len]

        taxon_ids = [self.vocab.encode(g) for g, _ in sample]
        abundances = [transform_abundance(a) for _, a in sample]

        pad_len = self.max_len - len(taxon_ids)
        taxon_ids = taxon_ids + [PAD_IDX] * pad_len
        abundances = abundances + [0.0] * pad_len

        taxon_ids = torch.tensor(taxon_ids, dtype=torch.long)
        abundances = torch.tensor(abundances, dtype=torch.float32).unsqueeze(-1)

        # next-token targets for the causal LM objective (shifted by one)
        genus_target = torch.cat([taxon_ids[1:], torch.tensor([PAD_IDX])])
        abundance_target = torch.cat([abundances[1:], torch.zeros(1, 1)])

        return {
            "taxon_ids": taxon_ids,
            "abundance": abundances,
            "genus_target": genus_target,
            "abundance_target": abundance_target,
        }


def generate_synthetic_samples(n_samples: int = 2000, n_genera: int = 300,
                                min_taxa: int = 20, max_taxa: int = 150,
                                seed: int = 0) -> List[List[Tuple[str, float]]]:
    """
    Synthetic (genus, abundance) samples for pipeline validation only --
    NOT a substitute for real MGnify/Atlas data. Draws abundances from a
    Dirichlet so they behave like real compositional data (sum to ~1,
    long-tailed), over a fixed synthetic genus vocabulary.

    Use this to confirm the model trains, loss decreases, and the
    attribution code runs end-to-end before spending time on real data
    ingestion.
    """
    rng = random.Random(seed)
    genus_pool = [f"genus_{i}" for i in range(n_genera)]
    max_taxa = min(max_taxa, n_genera)
    min_taxa = min(min_taxa, max_taxa)

    samples = []
    for _ in range(n_samples):
        k = rng.randint(min_taxa, max_taxa)
        chosen = rng.sample(genus_pool, k)
        # Dirichlet-ish via normalized exponential draws -> long-tailed abundances
        raw = [rng.expovariate(1.0) for _ in chosen]
        total = sum(raw)
        abundances = [r / total for r in raw]
        samples.append(list(zip(chosen, abundances)))
    return samples