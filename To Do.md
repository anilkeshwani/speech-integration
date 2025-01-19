# To Do

- [ ] Handle interaction between num batches per epoch and gradient accumulation, given that we're modularizing `train_one_epoch`
- [ ] Make sardalign installable without dependency conflicts
    - NOTE: Python versions are aligned at 3.10.6

## Scaling Up

- [ ] Implement support for optimizer in backward
- [ ] Implement support for activation offloading - this should be used with optimizer in backward
