#!/usr/bin/env python
"""SFT training via the stateful Trainer class.

Drop-in replacement for train_sft.py — same Hydra config interface,
identical training behavior (verified by equivalence tests).
"""

import hydra

from ssi.trainer import Trainer


@hydra.main(config_path="../conf", config_name="sft", version_base=None)
def main(cfg):
    trainer = Trainer(cfg)
    trainer.setup()
    trainer.train()
    trainer.cleanup()


if __name__ == "__main__":
    main()
