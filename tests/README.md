# "tests"

These are _informal_ ad hoc tests.

# Checking Custom TextCompletionDataset

i.e. the one with online interleaving and de-duplication.

```
export LOG_LEVEL=DEBUG && \
./tests/data_cpt.py \
    checkpointer.checkpoint_dir='/mnt/scratch-artemis/anilkeshwani/models/extended/Llama-3.2-1B-5000-dsus' \
    checkpointer.checkpoint_files='["ft-model-00001-of-00001.safetensors"]' \
    data.train.dataset.debug_mode=1
```
