# Generation

Generation is performed with vLLM using the Hugging Face-compatible model directories that are written during training time (every `save_steps` global steps). 

Example call:

```bash
python scripts/generate.py \
  model=/mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-5000-dsus-sft/lyric-butterfly-478-id_3qj03g5e/checkpoints/epoch_0/global_step_70000
```

## Specifying a Dataset Explicitly

Where `data` is not explicitly specified, the model will generate ASR transcripts for the test split of the training dataset. 

Example call with explicit test dataset:

```bash
python scripts/generate.py \
  model=/mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-5000-dsus-sft/lyric-butterfly-478-id_3qj03g5e/checkpoints/epoch_0/global_step_70000 \
  data=sft/voxpopuli-hubert_large_ll60k-layer_22
```

## Specifying Generation Parameters

Parameters affecting generation, e.g. sampling parameters, are available to override via the CLI with Hydra syntax. 

See the _generation.yaml_ config for the available options.

Example call with an override:

```
python scripts/generate.py \
  model=/mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-5000-dsus-sft/lyric-butterfly-478-id_3qj03g5e/checkpoints/epoch_0/global_step_70000 \
  sampling_params.max_tokens=1024 # override default value
```

### Specifying Model-specific Parameters

Parameters that are fixed for a given model checkpoint are automatically imputed from the training config, which is saved under the _checkpoints/_ subdirectory of a given run with the name _torchtune_config.yaml_. In the absence of a training configuration, these options can still be specified.

```bash
python scripts/generate.py \
    model=/mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-5000-dsus-sft/lyric-butterfly-478-id_3qj03g5e/checkpoints/epoch_0/global_step_70000 \
    speech.deduplicate=false \
    speech.n_dsus=5000 \
    data=sft/mls-hubert_large_ll60k-layer_22
```

## Extra

There is a hacky script, [generation_launcher.sh](/snippets/generation_launcher.sh), to launch generation for all checkpoints in a given training run (for a given epoch) under [snippets](/snippets)
