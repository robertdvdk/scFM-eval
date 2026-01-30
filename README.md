This is a benchmark suite designed to test the performance of single-cell foundation models on a variety of tasks. The general setup is that the benchmark provides .h5ad files that the user has to embed using their foundation model of choice. Then, the embeddings are put in a pre-specified folder (data/[TASK]/submission), and then the benchmark suite evaluates the quality of the embeddings on a variety of tasks. The relevant configs are in configs/task/[TASK].yaml.

The tasks currently implemented are:

1. Zero-shot batch integration
2. Proteome prediction
3. Drug response prediction
4. Gene perturbation

The code is designed to work with a devcontainer, but you can also build the Docker container by just using the Dockerfile. Run the code by calling
```bash
python src/main.py task=[TASK]
```
where TASK is batch_integration, drug_response_prediction, gene_perturbation or proteome_prediction.

The scripts to generate the data are in data_prep.

# Zero-shot batch integration
[ TODO ]
Batch integration is implemented using the scib_metrics package, and with the dataset from Neftel et al. (2019) (https://pubmed.ncbi.nlm.nih.gov/31327527/).

# Proteome prediction
[ TODO ]
Proteome dataset from Liu et al. (2025) (https://doi.org/10.1038/s41551-025-01528-z).
data/proteome_prediction/ground_truth contains PBMC.h5ad, a dataset of

# Drug response prediction
[ TODO ]

# Gene perturbation
[ TODO ]
