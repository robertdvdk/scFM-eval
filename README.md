# scFM-eval

A benchmark suite for evaluating single-cell foundation models across five bioinformatics tasks:

1. **Zero-shot batch integration** — embed cells, evaluate batch correction without any training
2. **Finetuned batch integration** — finetune FM embeddings with domain-adaptive training, then evaluate
3. **Proteome prediction** — predict protein expression from RNA embeddings
4. **Drug response prediction** — predict drug sensitivity from cell + drug representations
5. **Gene perturbation** — predict gene expression changes after perturbation

The general workflow is: take a ground-truth `.h5ad` file, embed it with a foundation model, place the embeddings in `data/<task>/submission/`, and run the evaluation.

## Setup

**Requirements:** Python 3.13, Docker

The project runs inside a Docker devcontainer. All commands should be executed through:

```bash
./run_in_container.sh <COMMAND>
```

This script finds the running devcontainer and executes the command as `appuser` in `/app`.

### Dependencies

The main environment uses `uv` for dependency management:

```bash
./run_in_container.sh uv sync
```

### Foundation model environments

Foundation models often have conflicting dependencies (different Python versions, different torch versions). Each model gets an isolated virtualenv under `envs/`:

```bash
# Set up all model environments
./run_in_container.sh bash envs/setup_envs.sh

# Set up a single model
./run_in_container.sh bash envs/setup_envs.sh scgpt
```

| Model | Venv | Python | Notes |
|---|---|---|---|
| scGPT | `envs/scgpt/.venv` | 3.12 | Two checkpoints: `scGPT_human`, `scGPT_pancancer` |
| Geneformer | `envs/geneformer/.venv` | 3.12 | V2 (4096 context window) from HuggingFace |
| CancerFoundation | `envs/cancerfoundation/.venv` | default | PyTorch Lightning; source mounted at `/CancerFoundation` |
| scFoundation | `envs/scfoundation/.venv` | default | MAE-autobin; source mounted at `/scFoundation` |

Models are invoked through subprocess isolation (`src/models/subprocess_embed.py`): the main process serializes the AnnData to a temp file, spawns a subprocess using the model's venv, and reads back the embeddings.

## Running Tasks

```bash
./run_in_container.sh python src/main.py task=<TASK_NAME>
```

where `TASK_NAME` is one of: `batch_integration`, `finetuned_batch_integration`, `proteome_prediction`, `drug_response_prediction`, `gene_perturbation`.

### Examples

```bash
# Zero-shot batch integration on default dataset (neftel_ss2)
./run_in_container.sh python src/main.py task=batch_integration

# Batch integration on a different dataset with a specific model config
./run_in_container.sh python src/main.py task=batch_integration task.dataset_name=kim_lung model=scgpt_human

# Finetuned batch integration with custom DAT weight
./run_in_container.sh python src/main.py task=finetuned_batch_integration task.finetune.dat_weight=5.0

# Drug response prediction with 10-fold cross-validation in cold_cell mode
./run_in_container.sh python src/main.py task=drug_response_prediction task.split_mode=cold_cell task.k_fold=10

# Gene perturbation for a specific submission
./run_in_container.sh python src/main.py task=gene_perturbation task.model_name=MyModel
```

## Configuration

Configs use [Hydra](https://hydra.cc/) and are organized under `configs/`:

```
configs/
├── config.yaml          # Root config (defaults, output dir pattern)
├── task/                # Per-task configs
│   ├── batch_integration.yaml
│   ├── finetuned_batch_integration.yaml
│   ├── proteome_prediction.yaml
│   ├── drug_response_prediction.yaml
│   └── gene_perturbation.yaml
└── model/               # Per-foundation-model configs
    ├── scgpt_human.yaml
    ├── scgpt_pancancer.yaml
    ├── geneformer.yaml
    ├── cancerfoundation.yaml
    └── scfoundation.yaml
```

Any config value can be overridden from the CLI with dot notation:

```bash
./run_in_container.sh python src/main.py task=batch_integration task.parameters.seed=123
```

Outputs go to `outputs/{task.name}/{task.run_group}/{task.model_name}/{timestamp}/`.

## Foundation Models

Models are registered in `src/models/__init__.py` via `@register_model(name)`. The registry maps a string name to a wrapper class:

| Name | Class | File | Embedding dim |
|---|---|---|---|
| `scgpt` | `ScGPTWrapper` | `src/models/scgpt.py` | 512 |
| `geneformer` | `GeneformerWrapper` | `src/models/geneformer.py` | varies (from `AutoConfig`) |
| `cancerfoundation` | `CancerFoundationWrapper` | `src/models/cancerfoundation.py` | 512 |
| `scfoundation` | `ScFoundationWrapper` | `src/models/scfoundation.py` | 768 (`max` pool) or 3072 (`all`) |

Each wrapper extends `FoundationModelWrapper` (`src/models/base.py`) and implements:
- `load_pretrained()` — load checkpoint
- `embed(adata, batch_size) -> EmbeddingResult` — produce cell embeddings
- `forward(batch) -> Tensor` — differentiable forward pass (for finetuning)
- `compute_native_loss(batch) -> (Tensor, Tensor)` — pretraining objective (for finetuning)

To add a new model: create a wrapper class in `src/models/`, decorate it with `@register_model("my_model")`, add a config file in `configs/model/`, and optionally set up an isolated env in `envs/`.

## Tasks

### Zero-shot Batch Integration

Evaluates how well FM embeddings separate cell types while correcting for batch effects, without any task-specific training.

**Runner:** `BatchIntegrationRunner` (`src/tasks/batch_integration/runner.py`)

**Datasets:**
| Config name | Dataset | Label key |
|---|---|---|
| `neftel_ss2` (default) | Neftel et al. 2019 Smart-seq2 | `subtype` |
| `kim_lung` | Kim Lung | `celltype` |
| `ji_skin` | Ji Skin | `celltype` |

**Metrics** (via `scib-metrics` `Benchmarker`):
- **Bio conservation:** Leiden NMI, Leiden ARI, isolated label F1, isolated label ASW, cell type ASW
- **Batch correction:** graph connectivity, kBET, batch ASW, iLISI, cLISI
- **Total:** weighted combination of bio conservation and batch correction

**Baselines:** Harmony (HVG selection → PCA → `harmonypy`)

**Data format:**
- Ground truth: `.h5ad` in `data/batch_integration/{dataset}/ground_truth/`
- Submissions: CSV files with cell embeddings in `data/batch_integration/{dataset}/submission/`. Cell IDs must match the ground truth.

**Outputs:** `scib_results.svg` (results table), UMAP plots per embedding colored by batch and cell type.

### Finetuned Batch Integration

Same evaluation as zero-shot, but first finetunes the FM encoder on the target dataset using a combination of losses:
- **MLM** — masked language modeling (the FM's native pretraining objective)
- **DAT** — domain-adaptive training with a WGAN-style batch discriminator
- **ECS** — elastic cell similarity (optional, controlled by `ecs_threshold`)

**Runner:** `FinetunedBatchIntegrationRunner` (`src/tasks/finetuned_batch_integration/runner.py`)

**Key config options** (`task.finetune.*`):
| Option | Default | Description |
|---|---|---|
| `epochs` | 5 | Training epochs |
| `lr` | 1e-4 | Learning rate |
| `do_dat` | true | Enable domain-adaptive training |
| `dat_weight` | 1.0 | Weight for DAT loss |
| `n_critic` | 5 | Discriminator steps per encoder step |
| `ecs_threshold` | 0.0 | ECS threshold (0 = disabled) |
| `mask_ratio` | 0.4 | Fraction of genes masked for MLM |
| `wandb` | true | Log to Weights & Biases |

**Baselines:** scVI, scANVI (using raw counts from `adata.layers['counts']`)

**Finetuning implementations:** `src/models/finetune_cancerfoundation.py`, `src/models/finetune_scgpt.py`

### Proteome Prediction

Trains a small MLP to predict protein surface expression from cell embeddings, using CITE-seq data.

**Runner:** `ProteomePredictionRunner` (`src/tasks/proteome_prediction/runner.py`)

**Dataset:** PBMC CITE-seq from Liu et al. 2025. Train/test split provided as `PBMC_trainset_preprocessed.h5ad` / `PBMC_testset_preprocessed.h5ad` with protein targets in `prot_train.csv` / `prot_test.csv`.

**Model:** `ProteomePredictionModel` (`src/tasks/proteome_prediction/model.py`) — 3-layer MLP: Linear(cell_dim, 512) → BN → ReLU → Dropout(0.2) → Linear(512, 512) → BN → ReLU → Dropout(0.1) → Linear(512, protein_dim) → Sigmoid. Trained for 1 epoch with Adam + MSE loss.

**Metrics** (computed per protein, then aggregated):
- Pearson correlation
- Cosine similarity
- 1-MSE

**Baselines:**
- Mean predictor — predicts training-set mean for each protein
- Linear regression — `sklearn.linear_model.LinearRegression` with `StandardScaler`

**Outputs:** `model_comparison_metrics.png` (boxplot comparison), `predicted_vs_true_scatter_plots.png`

### Drug Response Prediction

Predicts drug sensitivity (IC50) from cell line embeddings and drug representations.

**Runner:** `DrugResponsePredictionRunner` (`src/tasks/drug_response_prediction/runner.py`)

**Dataset:** GDSC2 dose-response data. Drug representations from MolFormer (`drug_embeddings_molformer.csv`) or graph-based `.hkl` files. Cell line metadata with COSMIC IDs and TCGA cancer type labels.

**Model:** `DualStreamModel` (`src/tasks/drug_response_prediction/model.py`) — dual-stream architecture with separate cell encoder (MLP: 512→256) and drug encoder (MLP or UGCNN for graph drugs), fused through a head (512→128→1). Trained with Adam + MSE loss and early stopping.

**Split modes:**
| Mode | Description |
|---|---|
| `random` | Transductive random shuffle |
| `cold_drug` | Inductive: unseen drugs at test time |
| `cold_cell` | Inductive: unseen cell lines at test time |
| `cancer_stratified` | Cold cell, stratified by cancer type |
| `double_cold` | Zero-shot: unseen drugs × unseen cell lines |
| `loto` | Leave-one-tissue-out |

K-fold cross-validation is enabled by setting `task.k_fold` (default: 10).

**Metrics:**
- MAE (mean absolute error)
- Per-drug Pearson correlation (drugs with >5 samples)
- Per-cell Pearson correlation
- Global Pearson correlation

**Baselines:**
- Global mean, per-drug mean, per-cell mean predictors
- Drug-only MLP (`DrugMLP`) — ignores cell embeddings
- HVG+PCA — 1200 HVGs + 256 PCs as cell representation → DualStreamModel
- Raw expression → DualStreamModel (when `task.run_raw_expression_baseline=true`)

**Data format:**
- Cell embeddings: CSV with cell line ID and embedding columns in `data/drug_response_prediction/submission/`
- Drug embeddings: CSV or directory of `.hkl` graph files in `data/drug_response_prediction/ground_truth/`
- Dose-response matrix: CSV with (cell, drug, IC50) format
- Metadata: CSV with `COSMIC_ID` and `TCGA_DESC` columns

### Gene Perturbation

Evaluates predicted gene expression changes after CRISPR perturbation.

**Runner:** `GenePerturbationRunner` (`src/tasks/gene_perturbation/runner.py`)

**Dataset:** Norman et al. 2019 (Perturb-seq). Ground truth from [Zenodo](https://zenodo.org/records/7041849). Includes precomputed gene weights for weighted metrics (`norman19_weights_vsrest.pkl`, `norman19_weights_vscontrol.pkl`).

**Metrics** (computed on mean expression deltas from control):
- MSE
- Weighted MSE (using precomputed gene importance weights)
- R² on deltas
- Weighted R² on deltas

Reported as both mean and median across perturbations.

**Data format:**
- Ground truth: `.h5ad` at `data/gene_perturbation/norman19/norman19_processed.h5ad` with `obs['condition']` column
- Submission: `.h5ad` at `data/gene_perturbation/norman19/submission/{model_name}.h5ad`. Must have exactly matching `obs_names` and `var_names` as the ground truth.

## Data Format

Ground truth `.h5ad` files store data in two layers:

- **`adata.X`**: Normalized and log1p-transformed expression data. Used by methods operating on log-normalized counts (e.g., Harmony for HVG selection and PCA).
- **`adata.layers['counts']`**: Raw (unnormalized) integer counts. Required by methods that perform internal normalization (e.g., scVI, scANVI).

Both layers are needed so all methods start from the same preprocessing.

## Data Preparation

Scripts and notebooks for preparing evaluation datasets are in `data_prep/`:

- `data_prep/batch_integration/data.py` — downloads and processes perturbation datasets (Norman19, Replogle22) from Zenodo/figshare
- `data_prep/proteome_prediction/prot_data.ipynb` — processes PBMC CITE-seq data into train/test splits
- `data_prep/drug_response_prediction/create_gdsc_data.ipynb` — processes GDSC2 dose-response, expression, and drug metadata

## Development

### Code quality

```bash
./run_in_container.sh ruff check --fix   # Lint
./run_in_container.sh ruff format         # Format
./run_in_container.sh pre-commit run --all-files
```

### Testing

```bash
./run_in_container.sh pytest
./run_in_container.sh pytest -v path/to/test_file.py
```
