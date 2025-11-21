import hydra
from omegaconf import DictConfig

from tasks.batch_integration import BatchIntegrationRunner


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    task_name = cfg.task.name
    print(f"=== Starting Evaluation: {task_name} ===")

    # Dispatcher: Decide which runner to use based on the config
    if task_name == "batch_integration":
        runner = BatchIntegrationRunner(cfg)
        results = runner.run()
    elif task_name == "perturbation":
        # runner = PerturbationRunner(cfg)
        raise NotImplementedError("Perturbation task not yet implemented")
    else:
        raise ValueError(f"Unknown task: {task_name}")

    print("\n=== Final Results ===")
    print(results)


if __name__ == "__main__":
    main()
