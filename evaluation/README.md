# Evaluation

## ASR

```bash
./scripts/evaluation/generation.py \
    "/mnt/scratch-artemis/anilkeshwani/tinyllama/experiments/TinyLlama-1.1B-intermediate-step-1431k-3T-extended-sentencepiece-5000/hf/TinyLlama-1.1B-intermediate-step-1431k-3T-extended-sentencepiece-5000-MLS-iter_0010500" \
    --test-jsonl "/tmp/dummy.jsonl" \
    --text-key "text" \
    --output-jsonl '/tmp/dummy_out.jsonl' \
    --prompt-template 'capital_of_france'
```

```bash
./scripts/evaluation/generation.py \
    "/mnt/scratch-artemis/anilkeshwani/tinyllama/experiments/TinyLlama-1.1B-intermediate-step-1431k-3T-extended-sentencepiece-5000/hf/TinyLlama-1.1B-intermediate-step-1431k-3T-extended-sentencepiece-5000-MLS-iter_0010500" \
    --test-jsonl "/tmp/dummy.jsonl" \
    --text-key "text" \
    --output-jsonl '/tmp/dummy_out.jsonl' \
    --prompt-template 'repeat_this_modality_switch'
```

```bash
./scripts/evaluation/generation.py \
    "/mnt/scratch-artemis/anilkeshwani/tinyllama/experiments/TinyLlama-1.1B-intermediate-step-1431k-3T-extended-sentencepiece-5000/hf/TinyLlama-1.1B-intermediate-step-1431k-3T-extended-sentencepiece-5000-MLS-iter_0010500" \
    --test-jsonl "/mnt/scratch-artemis/anilkeshwani/data/voxpopuli_hf/VoxPopuli_uroman_aligned_hubert.jsonl" \
    --text-key "text" \
    --output-jsonl '' \
    --prompt-template 'basic'
```


```bash
./scripts/evaluation/generation.py \
    "/mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-5000-dsus/playful-morning-102-id_rq5tmfca/checkpoints/global-step-000382" \
    --test-jsonl "/tmp/VoxPopuli_uroman_aligned_hubert_head10.jsonl" \
    --text-key "text" \
    --output-jsonl '/tmp/VoxPopuli_uroman_aligned_hubert_head10_generations.jsonl' \
    --prompt-template 'basic'
```
