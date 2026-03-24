"""
Standalone Entropy Calculator

Calculate Shannon entropy from protein sequence logits.
No pipeline dependencies — just torch.

Usage:
    python entropy_calc.py input.pt
    python entropy_calc.py input.pt --base 2 --output results.csv
    python entropy_calc.py input.pt --batch-size 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import torch


# =============================================================================
# ENTROPY MATH
# =============================================================================

def calculate_entropy(logits: torch.Tensor, base: str = "e") -> torch.Tensor:
    """
    Shannon entropy per position from logits.

    Args:
        logits: (num_residues, vocab_size)
        base: 'e' (nats), '2' (bits), '10' (dits)

    Returns:
        (num_residues,) entropy values
    """
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)

    if base == "2":
        entropy = entropy / torch.log(torch.tensor(2.0))
    elif base == "10":
        entropy = entropy / torch.log(torch.tensor(10.0))

    return entropy


def calculate_entropy_batched(
    logits: torch.Tensor, base: str = "e", batch_size: int = 10000
) -> torch.Tensor:
    """Memory-efficient batched entropy for large tensors."""
    n = logits.shape[0]
    if n <= batch_size:
        return calculate_entropy(logits, base)

    chunks = []
    for i in range(0, n, batch_size):
        chunks.append(calculate_entropy(logits[i : i + batch_size], base))
    return torch.cat(chunks)


# =============================================================================
# LOGIT LOADING
# =============================================================================

def load_logits(path: str) -> List[Tuple[str, torch.Tensor]]:
    """
    Load logits from a .pt file. Handles three formats:

    1. Raw tensor (num_residues, vocab_size) — single sequence
    2. List of tensors — multiple sequences
    3. Dict with 'logits' key — pipeline embeddings.pt format
    """
    data = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(data, torch.Tensor):
        if data.dim() == 2:
            return [("seq_0", data)]
        elif data.dim() == 3:
            return [(f"seq_{i}", data[i]) for i in range(data.shape[0])]
        else:
            raise ValueError(f"Unexpected tensor shape: {data.shape}")

    if isinstance(data, list):
        return [(f"seq_{i}", t) for i, t in enumerate(data)]

    if isinstance(data, dict):
        logits_list = data.get("logits")
        if logits_list is None:
            raise ValueError("Dict has no 'logits' key. Available keys: " + str(list(data.keys())))
        seq_ids = data.get("sequence_id", [f"seq_{i}" for i in range(len(logits_list))])
        return list(zip(seq_ids, logits_list))

    raise ValueError(f"Unsupported data type: {type(data)}")


# =============================================================================
# OUTPUT
# =============================================================================

def print_results(
    seq_id: str, entropy: torch.Tensor, show_positions: bool = True
) -> None:
    """Print entropy summary for one sequence."""
    print(f"\n{'=' * 60}")
    print(f"  {seq_id}  ({len(entropy)} positions)")
    print(f"{'=' * 60}")
    print(f"  Mean:  {entropy.mean().item():.4f}")
    print(f"  Std:   {entropy.std().item():.4f}")
    print(f"  Min:   {entropy.min().item():.4f}  (position {entropy.argmin().item()})")
    print(f"  Max:   {entropy.max().item():.4f}  (position {entropy.argmax().item()})")

    if show_positions and len(entropy) <= 200:
        print(f"\n  Per-position entropy:")
        # Print in rows of 10
        for i in range(0, len(entropy), 10):
            chunk = entropy[i : i + 10]
            vals = "  ".join(f"{v:.3f}" for v in chunk.tolist())
            print(f"    [{i:>4d}-{min(i+9, len(entropy)-1):>4d}]  {vals}")


def save_csv(
    results: List[Tuple[str, torch.Tensor]], output_path: str
) -> None:
    """Save per-position entropy to CSV."""
    lines = ["sequence_id,position,entropy"]
    for seq_id, entropy in results:
        for pos, val in enumerate(entropy.tolist()):
            lines.append(f"{seq_id},{pos},{val:.6f}")
    Path(output_path).write_text("\n".join(lines) + "\n")
    print(f"\nSaved per-position entropy to {output_path}")


def save_summary_csv(
    results: List[Tuple[str, torch.Tensor]], output_path: str
) -> None:
    """Save summary statistics to CSV."""
    lines = ["sequence_id,num_positions,mean,std,min,max"]
    for seq_id, entropy in results:
        lines.append(
            f"{seq_id},{len(entropy)},"
            f"{entropy.mean().item():.6f},{entropy.std().item():.6f},"
            f"{entropy.min().item():.6f},{entropy.max().item():.6f}"
        )
    Path(output_path).write_text("\n".join(lines) + "\n")
    print(f"Saved summary to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate Shannon entropy from logits."
    )
    parser.add_argument("input", help="Path to .pt file containing logits")
    parser.add_argument(
        "--base", choices=["e", "2", "10"], default="e",
        help="Entropy base: e=nats, 2=bits, 10=dits (default: e)"
    )
    parser.add_argument(
        "--output", "-o", help="Save per-position entropy to CSV"
    )
    parser.add_argument(
        "--summary", "-s", help="Save summary statistics to CSV"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10000,
        help="Batch size for large tensors (default: 10000)"
    )
    parser.add_argument(
        "--no-positions", action="store_true",
        help="Skip printing per-position values for short sequences"
    )
    args = parser.parse_args()

    # Load
    print(f"Loading logits from {args.input}...")
    sequences = load_logits(args.input)
    print(f"Found {len(sequences)} sequence(s)")

    # Calculate
    results = []
    for seq_id, logits in sequences:
        entropy = calculate_entropy_batched(logits, base=args.base, batch_size=args.batch_size)
        results.append((seq_id, entropy))
        print_results(seq_id, entropy, show_positions=not args.no_positions)

    # Overall summary across all sequences
    if len(results) > 1:
        all_entropy = torch.cat([e for _, e in results])
        print(f"\n{'=' * 60}")
        print(f"  OVERALL  ({len(results)} sequences, {len(all_entropy)} total positions)")
        print(f"{'=' * 60}")
        print(f"  Mean:  {all_entropy.mean().item():.4f}")
        print(f"  Std:   {all_entropy.std().item():.4f}")
        print(f"  Min:   {all_entropy.min().item():.4f}")
        print(f"  Max:   {all_entropy.max().item():.4f}")

    # Save
    if args.output:
        save_csv(results, args.output)
    if args.summary:
        save_summary_csv(results, args.summary)


if __name__ == "__main__":
    main()
