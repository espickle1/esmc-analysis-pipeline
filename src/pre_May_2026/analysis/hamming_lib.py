"""
Hamming Distance Analysis Library

Identifies sequence pairs differing by exactly N residues (default: 1),
filtered by temporal ordering. Produces a pairs table for downstream
comparison with embedding/entropy/logits analysis.

Usage:
    from hamming_lib import find_hamming_pairs, save_hamming_results

    # From DataFrames (notebook usage)
    results_df, skipped_df = find_hamming_pairs(sequences_df, metadata_df)
    save_hamming_results(results_df, skipped_df, output_dir="output")

    # From CSV files
    results_df, skipped_df = find_hamming_pairs_from_csv(
        "sequences.csv", "metadata.csv"
    )

    # CLI
    python hamming_lib.py sequences.csv metadata.csv [output_dir]
"""

import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================

DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%Y/%m/%d"]

__all__ = [
    "find_hamming_pairs",
    "find_hamming_pairs_from_csv",
    "save_hamming_results",
    "hamming_distance",
    "parse_date",
]


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def parse_date(date_str: str) -> Optional[datetime.date]:
    """
    Parse a date string trying multiple common formats.

    Args:
        date_str: Date string to parse

    Returns:
        datetime.date object, or None if unparseable

    Example:
        >>> parse_date("2024-01-15")
        datetime.date(2024, 1, 15)
        >>> parse_date("not-a-date") is None
        True
    """
    if not date_str or not isinstance(date_str, str):
        return None

    date_str = date_str.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def hamming_distance(seq_a: str, seq_b: str, max_distance: int = 1) -> int:
    """
    Compute Hamming distance between two equal-length sequences.

    Uses early exit: stops counting once mismatches exceed max_distance.

    Args:
        seq_a: First sequence
        seq_b: Second sequence
        max_distance: Stop counting after this many mismatches

    Returns:
        Number of mismatching positions, or max_distance + 1 if exceeded

    Raises:
        ValueError: If sequences have different lengths

    Example:
        >>> hamming_distance("MKTAYI", "MKTAYL")
        1
        >>> hamming_distance("MKTAYI", "XXXXXX")
        2
    """
    if len(seq_a) != len(seq_b):
        raise ValueError(
            f"Sequences must be equal length: {len(seq_a)} vs {len(seq_b)}"
        )

    mismatches = 0
    for a, b in zip(seq_a, seq_b):
        if a != b:
            mismatches += 1
            if mismatches > max_distance:
                return mismatches
    return mismatches


def find_mutation_position(seq_a: str, seq_b: str) -> Tuple[int, str, str]:
    """
    Find the single differing position between two Hamming-1 sequences.

    Args:
        seq_a: First sequence (earlier in time)
        seq_b: Second sequence (later in time)

    Returns:
        Tuple of (1-indexed position, residue_from, residue_to)

    Example:
        >>> find_mutation_position("MKTAYI", "MKTAYL")
        (6, 'I', 'L')
    """
    for i, (a, b) in enumerate(zip(seq_a, seq_b)):
        if a != b:
            return (i + 1, a, b)
    raise ValueError("Sequences are identical — no mutation found")


def _prepare_data(
    sequences_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge sequences with metadata and parse dates.

    Returns:
        Tuple of (valid_df with parsed dates, skipped_df with reasons)
    """
    skipped_rows = []

    # Deduplicate metadata: keep earliest date per sequence_id
    meta = metadata_df[["sequence_id", "name", "date"]].copy()
    meta["parsed_date"] = meta["date"].astype(str).apply(parse_date)

    # Separate records with no usable date
    no_date = meta[meta["parsed_date"].isna()]
    for _, row in no_date.iterrows():
        raw = str(row["date"])
        reason = "missing_date" if raw in ("", "nan", "None") else f"unparseable_date: {raw}"
        skipped_rows.append({
            "sequence_id": row["sequence_id"],
            "name": row.get("name", ""),
            "date_raw": raw,
            "skip_reason": reason,
        })

    has_date = meta[meta["parsed_date"].notna()].copy()

    # If multiple metadata rows per sequence_id, keep earliest date
    has_date = has_date.sort_values("parsed_date").drop_duplicates(
        subset="sequence_id", keep="first"
    )

    # Merge with sequences
    merged = sequences_df.merge(has_date, on="sequence_id", how="inner")

    skipped_df = pd.DataFrame(
        skipped_rows,
        columns=["sequence_id", "name", "date_raw", "skip_reason"],
    )

    return merged, skipped_df


# =============================================================================
# HIGH-LEVEL API
# =============================================================================

def find_hamming_pairs(
    sequences_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    distance: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find all sequence pairs at exact Hamming distance, with temporal ordering.

    Sequences are grouped by length (Hamming distance is undefined for
    unequal lengths). Within each group, all pairs are compared with
    early-exit optimization.

    Args:
        sequences_df: DataFrame with columns: sequence_id, sequence, length
        metadata_df: DataFrame with columns: sequence_id, name, date
        distance: Target Hamming distance (default: 1)

    Returns:
        Tuple of (results_df, skipped_df)

        results_df columns:
            seq_id_a, name_a, date_a, seq_id_b, name_b, date_b,
            sequence_length, mutation_position (1-indexed),
            residue_from, residue_to, hamming_distance, same_date

        skipped_df columns:
            sequence_id, name, date_raw, skip_reason

    Example:
        >>> results, skipped = find_hamming_pairs(sequences_df, metadata_df)
    """
    merged, skipped_df = _prepare_data(sequences_df, metadata_df)

    if merged.empty:
        empty_results = pd.DataFrame(columns=[
            "seq_id_a", "name_a", "date_a",
            "seq_id_b", "name_b", "date_b",
            "sequence_length", "mutation_position",
            "residue_from", "residue_to",
            "hamming_distance", "same_date",
        ])
        return empty_results, skipped_df

    results = []
    groups = merged.groupby("length")

    for length, group in groups:
        if len(group) < 2:
            continue

        rows = group.to_dict("records")

        for rec_a, rec_b in combinations(rows, 2):
            dist = hamming_distance(
                rec_a["sequence"], rec_b["sequence"], max_distance=distance
            )
            if dist != distance:
                continue

            # Determine temporal order: A is earlier, B is later
            date_a = rec_a["parsed_date"]
            date_b = rec_b["parsed_date"]
            same_date = date_a == date_b

            if date_a > date_b:
                rec_a, rec_b = rec_b, rec_a
                date_a, date_b = date_b, date_a

            # Find mutation details
            if distance == 1:
                pos, res_from, res_to = find_mutation_position(
                    rec_a["sequence"], rec_b["sequence"]
                )
            else:
                pos, res_from, res_to = None, None, None

            results.append({
                "seq_id_a": rec_a["sequence_id"],
                "name_a": rec_a["name"],
                "date_a": str(date_a),
                "seq_id_b": rec_b["sequence_id"],
                "name_b": rec_b["name"],
                "date_b": str(date_b),
                "sequence_length": length,
                "mutation_position": pos,
                "residue_from": res_from,
                "residue_to": res_to,
                "hamming_distance": dist,
                "same_date": same_date,
            })

    results_df = pd.DataFrame(results, columns=[
        "seq_id_a", "name_a", "date_a",
        "seq_id_b", "name_b", "date_b",
        "sequence_length", "mutation_position",
        "residue_from", "residue_to",
        "hamming_distance", "same_date",
    ])

    return results_df, skipped_df


def find_hamming_pairs_from_csv(
    sequences_path: Union[str, Path],
    metadata_path: Union[str, Path],
    distance: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find Hamming pairs from CSV files produced by fasta_cleaner.

    Args:
        sequences_path: Path to sequences.csv
        metadata_path: Path to metadata.csv
        distance: Target Hamming distance (default: 1)

    Returns:
        Tuple of (results_df, skipped_df)

    Example:
        >>> results, skipped = find_hamming_pairs_from_csv(
        ...     "sequences.csv", "metadata.csv"
        ... )
    """
    sequences_df = pd.read_csv(sequences_path)
    metadata_df = pd.read_csv(metadata_path)
    return find_hamming_pairs(sequences_df, metadata_df, distance=distance)


def save_hamming_results(
    results_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    output_dir: Union[str, Path] = ".",
    prefix: str = "",
) -> Tuple[Path, Path]:
    """
    Save Hamming analysis results to CSV files.

    Args:
        results_df: DataFrame of Hamming pairs
        skipped_df: DataFrame of skipped records
        output_dir: Directory to save files (default: current directory)
        prefix: Optional prefix for filenames

    Returns:
        Tuple of (results_path, skipped_path)

    Example:
        >>> save_hamming_results(results_df, skipped_df, output_dir="output")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"{prefix}hamming_results.csv"
    skipped_path = output_dir / f"{prefix}hamming_skipped.csv"

    results_df.to_csv(results_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)

    return results_path, skipped_path


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python hamming_lib.py <sequences.csv> <metadata.csv> [output_dir]")
        print("\nFinds sequence pairs with Hamming distance = 1 and temporal ordering.")
        print("Outputs hamming_results.csv and hamming_skipped.csv")
        sys.exit(1)

    seq_path = sys.argv[1]
    meta_path = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "."

    print(f"Reading sequences from: {seq_path}")
    print(f"Reading metadata from:  {meta_path}")

    results_df, skipped_df = find_hamming_pairs_from_csv(seq_path, meta_path)

    print(f"\nResults:")
    print(f"  {len(results_df)} Hamming-1 pairs found")
    print(f"  {len(skipped_df)} records skipped")

    if not results_df.empty:
        same_count = results_df["same_date"].sum()
        directed_count = len(results_df) - same_count
        print(f"  {directed_count} temporally directed pairs")
        print(f"  {same_count} same-date pairs")

    results_path, skipped_path = save_hamming_results(
        results_df, skipped_df, output_dir=out_dir
    )

    print(f"\nSaved to:")
    print(f"  {results_path}")
    print(f"  {skipped_path}")