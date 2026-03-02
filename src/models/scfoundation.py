"""scFoundation model wrapper.

scFoundation uses a Masked AutoEncoder with Auto-Discretization Binning
(MAE-autobin) architecture. The model processes 19,264 genes plus two
sequencing depth tokens (T, S). Cell embeddings are extracted by
concatenating CLS, SEP, max-pool, and mean-pool representations from the
encoder output (4 x 768 = 3072-dim with pool_type='all').

This wrapper adds the scFoundation model directory to sys.path at runtime
to import the model architecture and loading utilities.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset

from . import register_model
from .base import EmbeddingResult, FoundationModelWrapper

log = logging.getLogger(__name__)


def _gather_data(data, labels, pad_token_id):
    """Gather non-zero gene values into a compact tensor with padding.

    Reimplemented from scFoundation's ``load.gatherData`` to avoid
    runtime sys.path dependencies.
    """
    value_nums = labels.sum(1)
    max_num = max(value_nums)

    fake_data = torch.full((data.shape[0], max_num), pad_token_id, device=data.device)
    data = torch.hstack([data, fake_data])

    fake_label = torch.full((labels.shape[0], max_num), 1, device=labels.device)
    none_labels = ~labels
    labels = labels.float()
    labels[none_labels] = torch.tensor(-float("Inf"), device=labels.device)

    tmp_data = torch.tensor(
        [(i + 1) * 20000 for i in range(labels.shape[1], 0, -1)],
        device=labels.device,
    )
    labels += tmp_data
    labels = torch.hstack([labels, fake_label])

    fake_label_gene_idx = labels.topk(max_num).indices
    new_data = torch.gather(data, 1, fake_label_gene_idx)
    padding_labels = new_data == pad_token_id

    return new_data, padding_labels


@register_model("scfoundation")
class ScFoundationWrapper(FoundationModelWrapper):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self._embedding_dim: int = 0
        self._gene_vocab: list[str] = []
        self._model_config: dict | None = None
        self._gene_list: list[str] = []

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_pretrained(self) -> None:
        cfg = self.cfg
        model_dir = Path(cfg.model.pretrained_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"scFoundation model directory not found: {model_dir}")

        # Add model directory to sys.path so we can import architecture code
        model_dir_str = str(model_dir)
        if model_dir_str not in sys.path:
            sys.path.insert(0, model_dir_str)

        from load import convertconfig
        from pretrainmodels import select_model

        # Load gene reference list (19,264 genes)
        gene_index_file = model_dir / "OS_scRNA_gene_index.19264.tsv"
        if not gene_index_file.exists():
            raise FileNotFoundError(f"Gene index file not found: {gene_index_file}")
        gene_list_df = pd.read_csv(gene_index_file, header=0, delimiter="\t")
        self._gene_list = list(gene_list_df["gene_name"])
        self._gene_vocab = self._gene_list.copy()

        # Load checkpoint (without forced .cuda() from load_model_frommmf)
        ckpt_path = cfg.model.get("ckpt_path") or str(model_dir / "models" / "models.ckpt")
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. Download from the link in the scFoundation README."
            )

        key = cfg.model.get("model_key", "cell")
        model_data = torch.load(ckpt_path, map_location="cpu")
        model_data = model_data[key]
        model_data = convertconfig(model_data)
        config = model_data["config"]

        if "ppi_edge" not in config:
            config["ppi_edge"] = None

        model = select_model(config)
        model.load_state_dict(model_data["model_state_dict"])
        model.eval()

        self._model = model
        self._model_config = config

        # Embedding dim depends on pool_type
        encoder_hidden_dim = config["encoder"]["hidden_dim"]
        pool_type = cfg.model.get("pool_type", "all")
        if pool_type == "all":
            self._embedding_dim = encoder_hidden_dim * 4
        else:
            self._embedding_dim = encoder_hidden_dim

        log.info(
            "Loaded scFoundation: encoder_dim=%d, embedding_dim=%d, genes=%d, pool=%s",
            encoder_hidden_dim,
            self._embedding_dim,
            len(self._gene_list),
            pool_type,
        )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, adata, batch_size: int = 64) -> EmbeddingResult:
        import scipy.sparse

        cfg = self.cfg
        gene_col = cfg.model.get("gene_col", "index")
        pool_type = cfg.model.get("pool_type", "all")
        input_type = cfg.model.get("input_type", "singlecell")
        pre_normalized = cfg.model.get("pre_normalized", "F")
        tgthighres = cfg.model.get("tgthighres", "f1")
        pad_token_id = self._model_config["pad_token_id"]

        # Resolve gene names
        genes = adata.var_names.tolist() if gene_col == "index" else adata.var[gene_col].tolist()

        # Convert to DataFrame
        X = adata.X
        if scipy.sparse.issparse(X):
            X = X.toarray()
        X_df = pd.DataFrame(X, index=adata.obs_names.tolist(), columns=genes)

        # Align to 19,264 gene reference list
        to_fill_columns = list(set(self._gene_list) - set(X_df.columns))
        matched = len(self._gene_list) - len(to_fill_columns)
        log.info(
            "Matched %d/%d genes to scFoundation reference (%d padded with zeros)",
            matched,
            len(genes),
            len(to_fill_columns),
        )
        if matched == 0:
            raise ValueError(
                "No genes matched the scFoundation reference list. "
                f"Check that gene names (from adata.var['{gene_col}']) use "
                "standard HGNC symbols."
            )

        if to_fill_columns:
            padding_df = pd.DataFrame(
                np.zeros((X_df.shape[0], len(to_fill_columns))),
                columns=to_fill_columns,
                index=X_df.index,
            )
            X_df = pd.concat([X_df, padding_df], axis=1)
        X_df = X_df[self._gene_list]

        n_cells = X_df.shape[0]
        all_embeddings = []
        n_batches = (n_cells + batch_size - 1) // batch_size

        from tqdm import tqdm

        for start in tqdm(
            range(0, n_cells, batch_size),
            total=n_batches,
            desc="scFoundation embed",
        ):
            end = min(start + batch_size, n_cells)
            batch_df = X_df.iloc[start:end]
            batch_size_actual = batch_df.shape[0]

            with torch.no_grad():
                batch_data = []
                for i in range(batch_size_actual):
                    row = batch_df.iloc[i]

                    if input_type == "singlecell":
                        total = row.sum()
                        if pre_normalized == "F":
                            tmpdata = np.log1p(row / total * 1e4).tolist() if total > 0 else row.tolist()
                        elif pre_normalized == "T":
                            tmpdata = row.tolist()
                        elif pre_normalized == "A":
                            tmpdata = row.iloc[:-1].tolist()
                            total = row.iloc[-1]
                        else:
                            raise ValueError(f"pre_normalized must be T, F, or A, got {pre_normalized}")

                        if pre_normalized == "A":
                            total = row.iloc[-1]

                        log_total = np.log10(max(total, 1))
                        if tgthighres[0] == "f":
                            t_token = np.log10(max(total * float(tgthighres[1:]), 1))
                        elif tgthighres[0] == "a":
                            t_token = log_total + float(tgthighres[1:])
                        elif tgthighres[0] == "t":
                            t_token = float(tgthighres[1:])
                        else:
                            raise ValueError(f"tgthighres must start with f, a, or t, got {tgthighres}")
                        s_token = log_total
                        batch_data.append(tmpdata + [t_token, s_token])

                    elif input_type == "bulk":
                        tmpdata = row.tolist()
                        if pre_normalized == "T":
                            total = row.sum()
                        elif pre_normalized == "F":
                            total = np.log10(max(row.sum(), 1))
                        else:
                            raise ValueError(f"For bulk, pre_normalized must be T or F, got {pre_normalized}")
                        batch_data.append(tmpdata + [total, total])

                pretrain_gene_x = torch.tensor(batch_data, dtype=torch.float32).to(self._device)
                data_gene_ids = torch.arange(19266, device=self._device).repeat(batch_size_actual, 1)

                value_labels = pretrain_gene_x > 0
                x, x_padding = _gather_data(pretrain_gene_x, value_labels, pad_token_id)
                position_gene_ids, _ = _gather_data(data_gene_ids, value_labels, pad_token_id)

                with torch.amp.autocast(self._device.type, dtype=torch.float32):
                    x = self._model.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
                    position_emb = self._model.pos_emb(position_gene_ids)
                    x += position_emb

                    geneemb = self._model.encoder(x, x_padding)

                if pool_type == "all":
                    geneemb1 = geneemb[:, -1, :]  # CLS token
                    geneemb2 = geneemb[:, -2, :]  # SEP token
                    geneemb3, _ = torch.max(geneemb[:, :-2, :], dim=1)  # max pool
                    geneemb4 = torch.mean(geneemb[:, :-2, :], dim=1)  # mean pool
                    embeddings = torch.cat([geneemb1, geneemb2, geneemb3, geneemb4], dim=1)
                elif pool_type == "max":
                    embeddings, _ = torch.max(geneemb, dim=1)
                else:
                    raise ValueError(f"pool_type must be 'all' or 'max', got {pool_type}")

                all_embeddings.append(embeddings.detach().float().cpu().numpy())

        cell_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        log.info(
            "scFoundation embeddings: shape %s, embedding_dim %d",
            cell_embeddings.shape,
            self._embedding_dim,
        )

        return EmbeddingResult(
            cell_embeddings=cell_embeddings,
            gene_embeddings=None,
            embedding_dim=self._embedding_dim,
        )

    # ------------------------------------------------------------------
    # Finetuning methods (not yet implemented)
    # ------------------------------------------------------------------

    def tokenize(self, adata) -> dict[str, Tensor]:
        raise NotImplementedError("ScFoundationWrapper.tokenize() is not implemented for zero-shot evaluation.")

    def create_dataset(self, adata) -> Dataset:
        raise NotImplementedError("ScFoundationWrapper.create_dataset() is not implemented for zero-shot evaluation.")

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("ScFoundationWrapper.forward() is not implemented for zero-shot evaluation.")

    def compute_native_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        raise NotImplementedError(
            "ScFoundationWrapper.compute_native_loss() is not implemented for zero-shot "
            "evaluation. Native objective: masked autoencoder reconstruction."
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "scFoundation"

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def gene_vocabulary(self) -> list[str]:
        return self._gene_vocab
