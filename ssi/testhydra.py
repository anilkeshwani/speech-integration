import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="sft.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()
