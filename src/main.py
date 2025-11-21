import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(f"Running task: {cfg.task.name}")
    # This is where your evaluation logic will eventually go


if __name__ == "__main__":
    main()
