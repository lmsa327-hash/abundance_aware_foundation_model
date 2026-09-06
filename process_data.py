import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate mean and standard deviation of each taxon "
            "across the MicroCorpus260K abundance matrix."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to the Corpus abundance CSV",
    )

    parser.add_argument(
        "key",
        type=str,
        default="genus",
        help="Key for for abundance h5",
    )

    parser.add_argument(
        "normalize",
        default=True,
        help="Whether to normalize the abundance matrix",
    )

    parser.add_argument(
        "-op",
        "--output-phylogeny",
        type=Path,
        default=Path("abundance_aware/resources/phylogeny.csv"),
        help="Output phylogeny (default: abundance_aware/resources/phylogeny.csv)",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output Dir (default: data)",
    )

    parser.add_argument(
        "--index-col",
        default=None,
        help=(
            "Column containing sample IDs. "
            "If omitted, all columns are treated as taxa."
        ),
    )

    args = parser.parse_args()

    print(f"Reading: {args.input}")

    normalize = args.normalize

    # Read abundance matrix
    abundance = pd.read_csv(args.input)


    print(f"Dataset shape: {abundance.shape}")

    # Remove sample ID column if one exists
    if args.index_col is not None:
        if args.index_col not in abundance.columns:
            raise ValueError(
                f"Sample ID column '{args.index_col}' not found. "
                f"Available columns: {abundance.columns.tolist()}"
            )

        abundance = abundance.drop(columns=[args.index_col])

    # Convert abundance values to numeric
    abundance = abundance.apply(pd.to_numeric, errors="coerce")

    # Calculate statistics across samples
    phylogeny = pd.DataFrame({
        "#SampleID": abundance.columns,
        "mean": abundance.mean(axis=0),
        "std": abundance.std(axis=0),
    })

    # Reset index
    phylogeny = phylogeny.reset_index(drop=True)

    # Write output phylogeny
    #args.output.parent.mkdir(parents=True, exist_ok=True)
    phylogeny.to_csv(args.output_phylogeny, index=False)

    #normalize
    if normalize:
        abundance_norm = (abundance - phylogeny['mean']) / phylogeny['std']
        abundance_norm.to_csv(args.output_dir / "abundance_norm.csv", index=False)
        abundance.to_hdf(args.output_dir / "abundance_processed.h5", key=args.key)


if __name__ == "__main__":
    main()
