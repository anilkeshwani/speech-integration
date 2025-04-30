#!/usr/bin/env python

import logging
import os
import sys

import hydra
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL

from ssi.train import train


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="cpt.yaml", version_base=None)
def main(cfg):
    train(cfg)


if __name__ == "__main__":
    main()
