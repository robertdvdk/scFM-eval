import hydra
from omegaconf import DictConfig

from tasks import (
    BatchIntegrationRunner,
    DrugResponsePredictionRunner,
    GenePerturbationRunner,
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    task_name = cfg.task.name
    print(f"=== Starting Evaluation: {task_name} ===")

    # Dispatcher: Decide which runner to use based on the config
    if task_name == "batch_integration":
        runner = BatchIntegrationRunner(cfg)
    elif task_name == "drug_response_prediction":
        runner = DrugResponsePredictionRunner(cfg)
    elif task_name == "gene_perturbation":
        runner = GenePerturbationRunner(cfg)
    else:
        raise ValueError(f"Unknown task: {task_name}")

    results = runner.run()

    print("\n=== Final Results ===")
    print(results)


if __name__ == "__main__":
    main()
