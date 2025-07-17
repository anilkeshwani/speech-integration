# Evaluate Model Performance

## Word Error Rate (WER)

To compute word error rate, pass the _generations.jsonl_ file containing the output of _generate.py_ as in the following example:

```bash
python scripts/wer.py \
    /mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-5000-dsus-sft/hopeful-sound-525-id_5plc1ikb/generations/epoch_0/global_step_318000/mls-hubert_large_ll60k-layer_22/test/generations.jsonl
```

See `python scripts/wer.py --help` for details.



> [!IMPORTANT] 
> _wer.py_ relies on _generations.jsonl_ being located in nested directories indicating the dataset name and split: `<dataset>/<split>/generations.jsonl`. 
> If this is not the case, pass them in explicitly - see `--help`

_wer.py_ does not use Hydra

---

Help for _wer.py_ at last update:

```bash
usage: wer.py [-h] [--dataset DATASET] [--split SPLIT] [--gt_transcript_colname GT_TRANSCRIPT_COLNAME] [--normalizer {whisper}] generations_jsonl

Calculate Word Error Rate (WER) from model generations.

positional arguments:
  generations_jsonl     Path to the JSON lines file with generations.

options:
  -h, --help            show this help message and exit
  --dataset DATASET     Hugging Face dataset for reference transcripts.
  --split SPLIT         Hugging Face dataset split for reference transcripts.
  --gt_transcript_colname GT_TRANSCRIPT_COLNAME
                        Column name for ground truth transcripts in the dataset.
  --normalizer {whisper}
                        Text normalizer.
```
