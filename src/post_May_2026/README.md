# ESMC Analysis Notebooks (Post-May 2026)

Four Colab notebooks for protein analysis using ESM (Evolutionary Scale Modeling) from https://github.com/Biohub/esm.

## Notebooks (9 total)

### 1. **ESMC Local** (`1_ESMC_Local.ipynb`)
**Run ESMC locally on Colab GPU**

Local inference using ESMC protein language model with GPU acceleration.

**Features:**
- Model variants: `esmc_600m` (recommended), `esmc_300m`, `esmc_300m_open_v1`
- Extract embeddings from protein sequences
- Compute Shannon entropy (conservation analysis)
- Score mutations using log-likelihood ratios
- Works with Colab A100/T4 GPUs

**Outputs:**
- `{seq_id}_embedding.npy` - Sequence embeddings [seq_len, hidden_size]
- `{seq_id}_entropy.csv` - Position-wise conservation scores
- `entropy_analysis.png` - Visualization of entropy per position

**When to use:** For local GPU-accelerated analysis, custom batch processing, or when you want full control over the model.

---

### 2. **ESMC Biohub** (`2_ESMC_Biohub.ipynb`)
**Run ESMC remotely via Biohub Platform API**

Remote inference using Biohub's Forge API (uses your existing Forge credentials).

**Features:**
- No local GPU required (inference runs on Biohub servers)
- Biohub Platform authentication with Forge credentials
- Batch processing of sequences
- Extract embeddings and logits
- Compute entropy and mutation scores
- Useful for high-throughput analysis

**Outputs:**
- `{seq_id}_embedding.npy` - Sequence embeddings
- `{seq_id}_entropy.csv` - Conservation metrics
- `summary_statistics.csv` - Per-sequence statistics
- `entropy_analysis.png` - Confidence visualization

**When to use:** For remote analysis without local GPU, integration with Biohub infrastructure, or high-throughput screening.

---

### 3. **ESMC Sparse Autoencoders (SAE)** (`3_ESMC_SAE.ipynb`)
**Extract interpretable features using Sparse Autoencoders**

Decompose dense ESMC embeddings into 16,384 interpretable sparse features.

**Key Concept:**
- SAE extracts ~64 active features per residue (sparse representation)
- Features correspond to: binding sites, motifs, biophysical properties, evolutionary signals
- Each feature is interpretable (unlike dense embeddings)

**Features:**
- Feature activation heatmaps
- Top-20 most activated features per sequence
- Sparsity analysis (which features activate where)
- Feature activation distribution plots
- Biological feature interpretation

**Outputs:**
- `{seq_id}_sae_features.npy` - Full sparse feature matrix [seq_len, 16384]
- `{seq_id}_feature_mask.npy` - Binary mask of active features
- `{seq_id}_top_features.csv` - Top-20 activated features
- `{seq_id}_sae_analysis.png` - 4-panel analysis plots
- `{seq_id}_sae_report.md` - Feature interpretation report

**When to use:** For mechanistic understanding of what ESMC learns, interpretability studies, or feature-based classification.

---

### 4. **ESMFold2 Biohub** (`4_ESMFold2_Biohub.ipynb`)
**Predict 3D structures via Biohub Platform API**

Remote structure prediction via Biohub Platform using ESMFold2.

**Features:**
- Single sequences, multi-chain complexes
- RNA/DNA interactions and non-canonical amino acids
- MSA-guided predictions for improved accuracy
- Confidence metrics: pLDDT (per-residue confidence), pAE (predicted aligned error)
- PDB format output
- Interactive 3D visualization

**pLDDT Interpretation:**
- 90-100: Very high confidence (>90% likely correct)
- 70-90: High confidence (backbone correctly predicted)
- 50-70: Medium confidence (correct overall fold)
- <50: Low confidence (alignment unreliable)

**Outputs:**
- `{seq_id}_structure.pdb` - 3D structure in PDB format
- `{seq_id}_metrics.json` - Confidence scores (pLDDT, pAE, mean confidence)
- `confidence_analysis.png` - pLDDT plots per structure
- `prediction_summary.csv` - Summary statistics for all predictions

**When to use:** For 3D structure prediction, structural analysis, docking preparation, or protein design validation.

---

### 5. **ESMFold2 Modal** (`5_ESMFold2_Modal.ipynb`)
**Run ESMFold2 on Modal's serverless GPU infrastructure**

Leverage Modal's H100 GPUs with persistent model caching for high-throughput structure prediction without local GPU requirements.

**Key Concept:**
Modal is a serverless GPU platform where:
- You define Python functions with `@app.cls()` decorators
- Call `.remote()` to execute on Modal's infrastructure
- Persistent volumes cache models across runs
- Pay per GPU-minute used (no idle billing)

**Features:**
- Serverless execution (no GPU setup needed)
- H100 GPU (~20 min timeout per structure)
- Persistent model caching (faster subsequent runs)
- Can scale to 1000s of predictions with `concurrency_limit`
- CIF format output (compatible with PyMOL, NGL Viewer, Molstar)
- Confidence metrics: pLDDT, pTM, ipTM

**Pricing:**
- H100: $1.98/hour (~$0.02-0.03 per sequence)
- A100: $1.45/hour (alternative option)
- A40: $0.60/hour (budget option)

**Outputs:**
- `{seq_id}_structure.cif` - 3D structure (mmCIF format)
- `{seq_id}_metadata.json` - Confidence scores & model info
- `prediction_summary.csv` - Summary table

**When to use:** For high-throughput predictions without local GPU, cost-effective batch structure prediction, or scaling to 1000+ sequences.

**Requirements:**
- Modal account (free tier available at https://modal.com)
- API token (get from https://modal.com/account/tokens)

---

### 6. **ESMC Mutation Scoring - Colab** (`6_ESMC_Mutation_Scoring_Colab.ipynb`)
**Zero-shot mutation scoring with leave-one-out masking**

Score individual amino acid mutations using ESMC to identify which positions tolerate mutations and which are critical.

**Key Concept:**
- Mask each position one at a time, run inference
- Compute Shannon entropy (flexibility of position)
- Compute log-likelihood ratios (effect of each mutation)
- Identify mutation-tolerant sites for protein engineering

**Features:**
- **Entropy analysis**: High entropy = flexible, low entropy = critical
- **Log-likelihood ratios (LLR)**: Score each possible mutation (A42V, A42L, etc.)
- **Deleterious fraction**: % of 20 AAs harmful at each position
- **Heatmap visualization**: 20 AAs × sequence positions
- **Top mutations**: Ranked list of beneficial/deleterious mutations
- **Leave-one-out masking**: Efficient parallel batch processing

**Outputs:**
- `{seq_id}_entropy.png` - Conservation landscape plot
- `{seq_id}_deleterious_fraction.png` - Mutation tolerance per position
- `{seq_id}_llr_heatmap.png` - 20 AAs × positions heatmap
- `{seq_id}_per_position_metrics.csv` - Per-position entropy & tolerability
- `{seq_id}_mutations.csv` - All mutations scored and ranked

**When to use:** For protein engineering, saturation mutagenesis library design, identifying functional positions, or finding mutation hotspots.

---

### 7. **ESMC Mutation Scoring - Modal** (`7_ESMC_Mutation_Scoring_Modal.ipynb`)
**High-throughput mutation scoring on Modal serverless GPU**

Same mutation scoring as notebook 6, but runs remotely on Modal H100 for scaling to 100s of proteins without local GPU.

**Same features as notebook 6, plus:**
- Serverless execution (no GPU setup)
- H100 GPU processing
- Persistent model caching
- Scale to batch 100+ proteins with increased concurrency_limit
- Cost-effective for bulk protein engineering

**Pricing:**
- H100: $1.98/hour (~$0.02-0.04 per protein)
- First run: ~2-3 min (model loading)
- Subsequent: ~1-2 min per protein

**When to use:** For engineering 100+ protein variants, batch mutation scoring, or when you don't have a local GPU but need to score many proteins.

**Requirements:**
- Modal account with API token (https://modal.com)

---

### 8. **ESMC Fine-Tuning - Colab** (`8_ESMC_Finetuning_Colab.ipynb`)
**Fine-tune ESMC with LoRA adapters for custom classification tasks**

Efficiently fine-tune ESMC using parameter-efficient LoRA adapters (only ~0.4% of parameters trainable).

**Key Concept:**
- **LoRA (Low-Rank Adaptation)**: Frozen backbone + small trainable adapters
- **Memory efficient**: Fits on Colab A100 with batch size 8-16
- **Fast**: ~4 minutes for 1000 training steps
- **Task example**: EC enzyme classification (7 classes), but works for any classification task

**Features:**
- Custom training loop with AdamW optimizer
- Validation every 250 steps
- Stratified train/val split
- bfloat16 mixed precision
- Confusion matrix + loss/accuracy tracking
- Save fine-tuned adapters for reuse

**Outputs:**
- `lora_adapters/` - Fine-tuned LoRA weights (can be loaded into base model)
- `train_history.csv` - Per-step training metrics
- `val_history.csv` - Per-eval validation metrics
- `training_metrics.png` - Loss and accuracy plots
- `config.json` - Hyperparameter configuration

**When to use:** Fine-tune ESMC for classification (enzyme type, protein function, localization, stability prediction, etc.) on your own data. Works on Colab with standard GPU.

---

### 9. **ESMC Fine-Tuning - Modal** (`9_ESMC_Finetuning_Modal.ipynb`)
**Fine-tune ESMC on Modal's serverless H100 infrastructure**

Same fine-tuning as notebook 8, but runs remotely on Modal for more reliability and persistent model caching.

**Same features as notebook 8, plus:**
- Serverless execution (no local GPU needed)
- H100 GPU (more powerful, faster)
- Persistent model caching (faster subsequent runs)
- Persistent output volume (retrieve adapters later)
- Better for production workflows
- Scale to multiple concurrent fine-tuning jobs

**Pricing:**
- H100: $1.98/hour (~$0.13 per 1000-step fine-tuning run)
- A100 option available ($1.45/hour)

**Outputs:**
- Saved on Modal persistent volume (can be retrieved later)
- Same as Colab version

**When to use:** For production fine-tuning, reliable long-running jobs, or when you need to fine-tune multiple models in parallel (increase `concurrency_limit`).

**Requirements:**
- Modal account with API token (https://modal.com)

---

## Quick Start

1. **Copy a notebook to Google Colab:**
   - Open [Google Colab](https://colab.research.google.com)
   - File → Upload notebook → Select notebook from this directory
   - Or: `File → Open notebook → GitHub → https://github.com/Biohub/esm` (then navigate)

2. **For Local notebooks (1, 3, 6):**
   - Select GPU runtime: Runtime → Change runtime type → GPU (A100 recommended)
   - Run cells in order
   - Upload FASTA files or paste sequences

3. **For Biohub notebooks (2, 4):**
   - Have Forge credentials ready (username + API key)
   - Enter credentials when prompted
   - API automatically handles remote inference

4. **For Modal notebooks (5, 7):**
   - Sign up at https://modal.com (free tier available)
   - Get API token from https://modal.com/account/tokens
   - Enter token when prompted in notebook

## Dependencies

### Automatically installed:
- **ESMC:** `git+https://github.com/Biohub/esm.git@main`
- **Core ML:** torch, transformers, accelerate, einops
- **Bioinformatics:** biotite, biopython, rdkit
- **Utilities:** numpy, pandas, scikit-learn, matplotlib, tqdm, ipywidgets
- **Visualization:** py3dmol (for structure viewing)

### Python version:
- Python 3.10+ (notebooks auto-install compatible versions)

## Input Formats

**FASTA (all notebooks):**
```
>protein_id
MKVLIVAALLLAVGLAFWECEKRKYQCPEKPQE
>another_protein
MDVFMGVGVVDAKALVDYLVPGQDTAV
```

**Single sequence (optional):**
- Paste directly in "paste mode" cells
- Or upload FASTA via Colab file browser

## Output Organization

Each notebook creates an output directory with:
- `.npy` files (NumPy arrays for embeddings/features)
- `.csv` files (tabular data: entropy, statistics, top features)
- `.json` files (structured data: confidence metrics)
- `.pdb` files (3D structures)
- `.png` files (visualizations)
- `.md` files (reports)

All outputs are downloadable from Colab's file browser.

## Workflow Examples

### Example 1: Conservation Analysis
1. Start with **ESMC Local** (notebook 1)
2. Upload FASTA file
3. View entropy plots (high entropy = flexible, low entropy = constrained)

### Example 2: Feature Interpretation
1. Run **ESMC SAE** (notebook 3)
2. View top-20 activated features
3. Examine which positions activate which features
4. Correlate features with known binding sites or motifs

### Example 3: Structure-to-Function
1. Run **ESMFold2** (notebook 4) for 3D structures
2. Run **ESMC SAE** (notebook 3) to identify functional features
3. Map features onto 3D structure for mechanistic insights

### Example 4: High-Throughput Structure Prediction
1. Prepare FASTA with 100+ sequences
2. Use **ESMFold2 Modal** (notebook 5) for cost-effective parallel predictions
3. Increase `concurrency_limit` to run 10+ structures simultaneously
4. Collect all structures and confidence metrics

### Example 5: Hybrid Analysis Pipeline
1. Run **ESMC Local** (notebook 1) for sequence analysis
2. Run **ESMFold2 Modal** (notebook 5) for structure prediction (cheaper than Biohub)
3. Run **ESMC SAE** (notebook 3) to identify functional features
4. Map features onto Modal-predicted structures

### Example 6: Protein Engineering - Conservative
1. Run **ESMC Mutation Scoring - Colab** (notebook 6) on wild-type
2. Filter for high-entropy positions (entropy > 3.0)
3. Select mutations with LLR > 0.5 (likely beneficial)
4. Validate top 10 mutations experimentally

### Example 7: Protein Engineering - High-Throughput
1. Run **ESMC Mutation Scoring - Modal** (notebook 7) on 100 variants
2. Set `concurrency_limit=5` for parallel scoring
3. Identify universal constraints (low-entropy across all variants)
4. Design combinatorial library targeting high-entropy positions

### Example 8: Fine-Tune for Enzyme Function Prediction
1. Prepare dataset: protein sequences + EC (Enzyme Commission) labels
2. Run **ESMC Fine-Tuning - Colab** (notebook 8) to train classifier
3. Save LoRA adapters (~10MB, 0.4% of model)
4. Use fine-tuned model to classify new enzymes

### Example 9: Fine-Tune for Custom Protein Property
1. Collect labeled data: sequences + property (stability, expression, activity, etc.)
2. Run **ESMC Fine-Tuning - Modal** (notebook 9) for production fine-tuning
3. Save adapters on persistent volume
4. Deploy fine-tuned model for batch prediction on protein library

### Example 10: Complete ML Pipeline
1. Fine-tune on labeled data (notebook 8/9)
2. Apply fine-tuned model to unlabeled variants
3. Score top candidates with mutation scoring (notebook 6/7)
4. Validate experimentally or use SAE features for mechanistic insights (notebook 3)

## Comparison: Execution Methods

### **Structure Prediction (Notebooks 4 & 5)**

| Feature | Biohub (Notebook 4) | Modal (Notebook 5) |
|---------|-------------------|-------------------|
| **Authentication** | Forge credentials | Modal API token |
| **Setup** | Simple (just credentials) | Slightly more complex (app definition) |
| **GPU** | Shared infrastructure | Dedicated H100 |
| **Cost/prediction** | ~$0.05-0.10 | ~$0.02-0.03 (cheaper) |
| **Caching** | Per-session | Persistent across runs |
| **Scaling** | Sequential | Up to 10+ concurrent |
| **Best for** | Single/few structures | High-throughput (100+) |

### **Mutation Scoring (Notebooks 6 & 7)**

| Feature | Colab Local (Notebook 6) | Modal (Notebook 7) |
|---------|-------------------------|-------------------|
| **GPU Required** | Yes (A100/T4) | No (runs on H100) |
| **Setup** | None (just GPU selection) | Modal API token |
| **Speed/protein** | 1-2 min (local GPU) | 1-2 min (H100, similar) |
| **Memory Limit** | Colab GPU memory cap | H100 80GB (no constraints) |
| **Scaling** | ~1-5 proteins max | 100+ proteins easily |
| **Cost** | Free (Colab) | ~$0.02-0.04/protein |
| **Best for** | 1-5 proteins | 100+ proteins |

### **Fine-Tuning (Notebooks 8 & 9)**

| Feature | Colab Local (Notebook 8) | Modal (Notebook 9) |
|---------|-------------------------|-------------------|
| **GPU Required** | Yes (A100) | No (runs on H100) |
| **Setup** | None (GPU selection) | Modal API token |
| **Speed** | ~4 min per 1000 steps | ~4 min per 1000 steps (H100 faster) |
| **Reliability** | Colab session timeout risk | Persistent, reliable |
| **LoRA Rank** | 8 (default), up to 32 | 8 (default), up to 64 |
| **Batch Size** | 8 training, 16 eval | 8 training, 16 eval |
| **Cost** | Free (Colab) | ~$0.13 per 1000 steps |
| **Model Caching** | Per-session | Persistent (faster 2nd run) |
| **Best for** | Experiments, prototyping | Production, reliable fine-tuning |

**Choice guide:**
- **Structures, 1-10**: Use Biohub (simpler setup)
- **Structures, 100+**: Use Modal (cheaper, faster)
- **Mutations, 1-5 proteins**: Use Colab local (free, no setup)
- **Mutations, 100+ proteins**: Use Modal (better scaling)
- **Fine-tune, experimental**: Use Colab (free, quick)
- **Fine-tune, production**: Use Modal (reliable, persistent)
- **Already have Forge creds**: Use Biohub for structures
- **Cost-conscious at scale**: Use Modal with parallelization

---

## Troubleshooting

**"API error" in notebooks 2 or 4:**
- Verify Forge credentials are correct
- Check that API key hasn't expired
- Ensure internet connection is stable

**"CUDA out of memory" in notebooks 1 or 3:**
- Reduce BATCH_SIZE or use smaller model (esmc_300m)
- Use notebook 2 (Biohub) for remote inference instead
- Request A100 GPU in Colab (Runtime → Change runtime type)

**"Module not found" errors:**
- Re-run the pip install cells at the top
- Restart runtime: Runtime → Restart runtime

**Modal-specific issues:**

**"No Modal API token found"** (notebooks 5, 7):
- Sign up at https://modal.com (free tier available)
- Get API token from https://modal.com/account/tokens
- Paste token when prompted

**"ModuleNotFoundError: modal"** (notebooks 5, 7):
- The pip install cell may not have completed
- Re-run: `!pip install -q modal`
- Restart Colab runtime

**"Timeout in Modal execution"**:
- First run takes 2-3 minutes (model loading)
- Subsequent runs are faster
- If timeout persists, sequence may be too long
- Try a shorter test sequence first

**Mutation scoring specific:**

**"CUDA out of memory"** (notebook 6 - Colab):
- Reduce BATCH_SIZE from 4 to 2
- Use smaller model (esmc_300m)
- Or use Modal version (notebook 7) which has more memory

**"Tokenizer pad_id error"** (notebook 7):
- This can happen if ESM version changes
- Check if tokenizer has `pad_id` attribute
- May need to use `pad_token_id` instead (verify in ESM docs)

## Citation

If you use these notebooks, please cite:
- Lin, Z., et al. (2024). ESM3: Language models and tools for protein structure understanding. Preprint available at https://github.com/Biohub/esm

## Notebook Index

| # | Name | Type | GPU | Auth | Best For |
|---|------|------|-----|------|----------|
| 1 | ESMC Local | Embeddings | Local | None | Sequence analysis |
| 2 | ESMC Biohub | Embeddings | Remote | Forge | Remote analysis |
| 3 | ESMC SAE | Interpretability | Local | None | Feature extraction |
| 4 | ESMFold2 Biohub | Structures | Remote | Forge | Structure prediction |
| 5 | ESMFold2 Modal | Structures | H100 | Modal | Batch structures |
| 6 | Mutation Scoring Colab | Engineering | Local | None | Single proteins |
| 7 | Mutation Scoring Modal | Engineering | H100 | Modal | Batch scoring |
| 8 | Fine-Tuning Colab | Fine-Tuning | Local | None | Custom tasks (local) |
| 9 | Fine-Tuning Modal | Fine-Tuning | H100 | Modal | Custom tasks (batch) |

## Further Reading

- **Biohub ESM Repository:** https://github.com/Biohub/esm
- **ESM Cookbook:** https://github.com/Biohub/esm/tree/main/cookbook
- **ESM3 Paper:** https://arxiv.org/abs/2402.12261
- **Mutation Scoring Tutorial:** https://github.com/Biohub/esm/blob/main/cookbook/tutorials/esmc_mutation_scoring.ipynb
- **Fine-Tuning Tutorial:** https://github.com/Biohub/esm/blob/main/cookbook/tutorials/esmc_finetune.ipynb
- **PEFT LoRA Docs:** https://huggingface.co/docs/peft/
- **Modal Docs:** https://modal.com/docs

---

**Created:** June 2026  
**Based on:** Biohub ESM repository (main branch)  
**Total Notebooks:** 9  
**Coverage:**
- Sequence analysis: embeddings, conservation, interpretability
- Structure prediction: local, Biohub API, Modal
- Mutation engineering: mutation scoring (local/Modal)
- Model customization: fine-tuning with LoRA (local/Modal)  
**Scale:** From single proteins (Colab free) to 1000+ proteins (Modal serverless)
