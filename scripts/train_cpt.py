#!/usr/bin/env python

import hydra

from ssi.trainer import Trainer


@hydra.main(config_path="../conf", config_name="cpt", version_base=None)
def main(cfg):
    trainer = Trainer(cfg)
    trainer.setup()
    trainer.train()
    trainer.cleanup()


if __name__ == "__main__":
    main()
