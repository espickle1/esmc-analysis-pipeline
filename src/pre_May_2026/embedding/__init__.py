"""
Embedding Package

Functions for FASTA processing and ESM-C embedding generation.
"""

from .fasta_cleaner import (
    clean_sequence,
    hash_sequence,
    parse_header,
    parse_fasta,
    process_fasta_content,
    process_fasta_files,
    save_results,
)

from .esmc_embed_lib import (
    load_esmc_model,
    embed_single,
    embed_sequences,
    embed_from_csv,
    save_embeddings,
    load_embeddings,
    get_embedding_for_sequence,
    results_to_dataframe,
)

__all__ = [
    # FASTA cleaning
    "clean_sequence",
    "hash_sequence",
    "parse_header",
    "parse_fasta",
    "process_fasta_content",
    "process_fasta_files",
    "save_results",
    # Embedding
    "load_esmc_model",
    "embed_single",
    "embed_sequences",
    "embed_from_csv",
    "save_embeddings",
    "load_embeddings",
    "get_embedding_for_sequence",
    "results_to_dataframe",
]
