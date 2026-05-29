"""
Analysis Package

Functions for entropy calculation, logits analysis, Hamming distance
analysis, and visualization.
"""

from .hamming_lib import (
    find_hamming_pairs,
    find_hamming_pairs_from_csv,
    save_hamming_results,
    hamming_distance,
    parse_date,
)

from .entropy_lib import (
    calculate_entropy,
    calculate_entropy_batched,
    get_constrained_positions,
    get_flexible_positions,
    analyze_entropy,
    entropy_summary,
    save_entropy_results,
)

from .logits_lib import (
    pool_logits,
    extract_amino_acid_probs,
    scale_logits,
    plot_heatmap,
    analyze_residues,
    save_analysis,
    AA_VOCAB,
    ESM_VOCAB_FULL,
)

__all__ = [
    # Hamming
    "find_hamming_pairs",
    "find_hamming_pairs_from_csv",
    "save_hamming_results",
    "hamming_distance",
    "parse_date",
    # Entropy
    "calculate_entropy",
    "calculate_entropy_batched",
    "get_constrained_positions",
    "get_flexible_positions",
    "analyze_entropy",
    "entropy_summary",
    "save_entropy_results",
    # Logits
    "pool_logits",
    "extract_amino_acid_probs",
    "scale_logits",
    "plot_heatmap",
    "analyze_residues",
    "save_analysis",
    "AA_VOCAB",
    "ESM_VOCAB_FULL",
]
