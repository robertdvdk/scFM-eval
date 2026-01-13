This is a benchmark suite designed to test the performance of single-cell foundation models on a variety of tasks. The tasks currently implemented are:

1. Zero-shot batch integration
2. Proteome prediction
    1. Cite-seq PBMCs dataset
    2. Perturb-CITE-seq dataset. 73,114

Batch integration is implemented using the scib_metrics package, and with the dataset from Neftel et al. (2019) (https://pubmed.ncbi.nlm.nih.gov/31327527/).

Proteome dataset from Liu et al. (2025) (https://doi.org/10.1038/s41551-025-01528-z).

Run the code by calling
```bash
python src/main.py task=[TASK]
```
where TASK is batch_integration, drug_response_prediction, gene_perturbation or proteome_prediction.


# Proteome prediction
data/proteome_prediction/ground_truth contains PBMC.h5ad, a dataset of
