import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from torchtune.training.metric_logging import WandBLogger

from ssi.constants import TORCHTUNE_CONFIG_FILENAME


LOGGER = logging.getLogger(__name__)


class WandBLoggerPatched(WandBLogger):
    def log_config(self, config: DictConfig) -> None:
        """PATCHED. Saves the config locally and logs it to W&B.
        Patch: The config is stored in the checkpointer output directory.

        Args:
            config (DictConfig): Configuration to log to disk and W&B.
        """
        if self._wandb.run:
            resolved = OmegaConf.to_container(config, resolve=True)
            self._wandb.config.update(resolved, allow_val_change=self.config_allow_val_change)
            try:
                output_config_fname = Path(config.checkpointer.output_dir, TORCHTUNE_CONFIG_FILENAME)
                OmegaConf.save(config, output_config_fname)

                LOGGER.info(f"Logging {output_config_fname} to W&B under Files")
                self._wandb.save(output_config_fname, base_path=output_config_fname.parent)
            except Exception as e:
                LOGGER.error(
                    f"Error saving file {output_config_fname!s} to W&B. "
                    "Note: The config still will be logged the W&B workspace.\n"
                    f"Exception: \n{e}."
                )
