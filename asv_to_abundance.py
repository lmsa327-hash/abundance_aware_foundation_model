import argparse

import pandas as pd
from pathlib import Path


MIN_REL_ABUNDANCE = 0.0001   # 0.01%, matches the MGM paper's threshold
MIN_TAXA_PER_SAMPLE = 10  #MGM used this



# 1. Build the taxon label per ASV: genus if present, else family, else drop
def resolve_label(row):
    if pd.notna(row["Genus"]):
        return f"g__{row['Genus']}"
    if pd.notna(row["Family"]):
        return f"f__{row['Family']}"
    return None
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create abundance from ASV and taxonomy"
        )
    )
    parser.add_argument(
        "-tp",
        "--tax-path",
        type=Path,
        default=Path("16S_asv_tax.csv"),
        help="Taxonomy Path (default: 16S_asv_tax.csv)",
    )
    parser.add_argument(
        "-cp",
        "--counts-path",
        type=Path,
        default=Path("16S_asv_counts.csv"),
        help="Counts Path (default: 16S_asv_counts.csv)",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output Dir (default: data)",
    )
    args = parser.parse_args()

    tax = pd.read_csv(args.tax_path, index_col=0)
    counts = pd.read_csv(args.counts_path, index_col=0)

    n_asvs_total = len(tax)
    total_reads = counts.sum().sum()

    tax["__label__"] = tax.apply(resolve_label, axis=1)

    n_genus = (tax["__label__"].str.startswith("g__", na=False)).sum()
    n_family_fallback = (tax["__label__"].str.startswith("f__", na=False)).sum()
    n_dropped = tax["__label__"].isna().sum()

    dropped_asvs = tax.index[tax["__label__"].isna()]
    dropped_asvs_in_counts = counts.index.intersection(dropped_asvs)
    reads_dropped = counts.loc[dropped_asvs_in_counts].sum().sum()

    print(f"ASVs total:              {n_asvs_total:,}")
    print(f"  -> genus-level:        {n_genus:,}")
    print(f"  -> family fallback:    {n_family_fallback:,}")
    print(f"  -> dropped (no fam.):  {n_dropped:,}")
    print(f"Reads dropped:           {reads_dropped:,} ({100*reads_dropped/total_reads:.1f}% of total)")
    print()

    # 2. Map counts to taxon labels, sum per sample
    asv_to_label = tax["__label__"].dropna()
    counts_labeled = counts.loc[counts.index.intersection(asv_to_label.index)].copy()
    counts_labeled["__label__"] = counts_labeled.index.map(asv_to_label)
    taxon_table = counts_labeled.groupby("__label__").sum()

    # 3. Relative abundance per sample
    rel_abundance = taxon_table.div(taxon_table.sum(axis=0), axis=1)

    # 4. Threshold + re-filter
    rel_abundance = rel_abundance.where(rel_abundance >= MIN_REL_ABUNDANCE, 0.0)
    taxa_per_sample = (rel_abundance > 0).sum(axis=0)
    keep_samples = taxa_per_sample[taxa_per_sample >= MIN_TAXA_PER_SAMPLE].index
    dropped_samples = set(rel_abundance.columns) - set(keep_samples)
    rel_abundance = rel_abundance[keep_samples]
    rel_abundance = rel_abundance.loc[rel_abundance.sum(axis=1) > 0]

    print(f"Samples retained: {rel_abundance.shape[1]} / {taxon_table.shape[1]}")
    if dropped_samples:
        print(f"Samples dropped (<{MIN_TAXA_PER_SAMPLE} taxa): {sorted(dropped_samples)}")
    print(f"Taxa retained: {rel_abundance.shape[0]}")
    print()
    print("Taxa-per-sample distribution:")
    print((rel_abundance > 0).sum(axis=0).describe())

    output = args.output_dir / "abundance.csv"
    rel_abundance.to_csv(output, index=False)
    print(f"\nSaved to {output}")

if __name__ == "__main__":
    main()