#!/usr/bin/env python

import hydra

from ssi.train import train


@hydra.main(config_path="../conf", config_name="cpt", version_base=None)
def main(cfg):
    train(cfg)


if __name__ == "__main__":
    main()
