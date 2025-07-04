## Generation

Example call:

```bash
python ssi/generate.py \
    checkpointer.config_json='/mnt/scratch-artemis/anilkeshwani/models/extended/Llama-3.2-1B-5000-dsus/config.json' \
    checkpointer.checkpoint_dir='/mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-5000-dsus/playful-morning-102-id_rq5tmfca/checkpoints/global-step-053862' \
    checkpointer.checkpoint_files='["hf_model_0001_1.pt"]' \
    output_jsonl='generated-playful-morning-global-step-053862.jsonl'
```

