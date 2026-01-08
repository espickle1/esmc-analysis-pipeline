# 🧬 ESMC Analysis Pipeline

A production-ready Python pipeline for protein sequence analysis using ESM-C (Evolutionary Scale Modeling) embeddings.

## Features

- **FASTA Cleaning** - Parse and validate sequences, extract metadata
- **Embedding Generation** - Generate ESM-C embeddings with logits
- **Entropy Analysis** - Identify conserved and flexible regions via Shannon entropy
- **Logits Analysis** - Analyze amino acid propensities with heatmap visualization

## Quick Start

### Installation

```bash
git clone https://github.com/espickle1/esmc-analysis-pipeline.git
cd esmc-analysis-pipeline
pip install -r requirements.txt
```

### Usage Options

| Environment | Notebook | Description |
|-------------|----------|-------------|
| Google Colab | `pipeline_colab.ipynb` | Interactive widgets, auto-clones repo |
| VM (Azure/AWS) | `pipeline_vm.ipynb` | File paths, tqdm progress |

### Python Library Usage

```python
import sys
sys.path.insert(0, "/path/to/esmc-analysis-pipeline")

from src.embedding import load_esmc_model, embed_from_csv, process_fasta_files
from src.analysis import analyze_entropy, analyze_residues, plot_heatmap

# 1. Clean FASTA
seq_df, meta_df = process_fasta_files("data/sample_data/sample.fasta")
seq_df.to_csv("sequences.csv", index=False)

# 2. Generate embeddings
model = load_esmc_model("your_hf_token", model_name="esmc_600m")
results = embed_from_csv(model, "sequences.csv")

# 3. Analyze entropy
entropy = analyze_entropy(results)

# 4. Analyze residues
logits = analyze_residues(results, residues_of_interest={10: "Pos11", 50: "Pos51"})
plot_heatmap(logits["scaled_logits"], logits["residue_labels"])
```

## Project Structure

```
esmc-analysis-pipeline/
├── data/
│   └── sample_data/
│       └── sample.fasta       # Test sequences
├── src/
│   ├── analysis/
│   │   ├── entropy_lib.py     # Shannon entropy calculation
│   │   └── logits_lib.py      # Logits pooling & visualization
│   └── embedding/
│       ├── esmc_embed_lib.py  # ESM-C embedding generation
│       └── fasta_cleaner.py   # FASTA parsing & cleaning
├── pipeline_colab.ipynb       # Colab notebook
├── pipeline_vm.ipynb          # VM notebook (Azure/AWS)
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- HuggingFace account with ESM-C access

## API Reference

### Embedding Module

| Function | Description |
|----------|-------------|
| `load_esmc_model(token, model_name)` | Load ESM-C model |
| `embed_sequences(model, df)` | Embed sequences from DataFrame |
| `embed_from_csv(model, path)` | Embed sequences from CSV |
| `process_fasta_files(path)` | Clean FASTA, return seq/meta DataFrames |

### Analysis Module

| Function | Description |
|----------|-------------|
| `analyze_entropy(results)` | Calculate Shannon entropy per position |
| `analyze_residues(results, residues)` | Analyze logits at specific positions |
| `plot_heatmap(data, labels)` | Generate amino acid propensity heatmap |

## License

MIT License
