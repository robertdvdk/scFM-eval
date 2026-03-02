import hydra
from omegaconf import DictConfig, OmegaConf

from tasks import (
    BatchIntegrationRunner,
    DrugResponsePredictionRunner,
    FinetunedBatchIntegrationRunner,
    GenePerturbationRunner,
    ProteomePredictionRunner,
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    task_name = cfg.task.name

    # Auto-set label_key based on dataset_name
    dataset_label_key = {
        "ji_skin": "celltype",
        "kim_lung": "celltype",
        "neftel_ss2": "subtype",
    }
    dataset_name = cfg.task.get("dataset_name", None)
    if dataset_name in dataset_label_key:
        OmegaConf.update(cfg, "task.metadata.label_key", dataset_label_key[dataset_name])

    print(f"=== Starting Evaluation: {task_name} ===")

    # Dispatcher: Decide which runner to use based on the config
    if task_name == "batch_integration":
        runner = BatchIntegrationRunner(cfg)
    elif task_name == "finetuned_batch_integration":
        runner = FinetunedBatchIntegrationRunner(cfg)
    elif task_name == "drug_response_prediction":
        runner = DrugResponsePredictionRunner(cfg)
    elif task_name == "gene_perturbation":
        runner = GenePerturbationRunner(cfg)
    elif task_name == "proteome_prediction":
        runner = ProteomePredictionRunner(cfg)
    else:
        raise ValueError(f"Unknown task: {task_name}")

    results = runner.run()

    print("\n=== Final Results ===")
    print(results)


if __name__ == "__main__":
    main()
