# Speech Integration Research Codebase

This is a research codebase empirically investigating the comparative performances of difference approaches to integrating the speech modality into the Llama 3.2 1B foundation language model via discrete speech tokens.

We are investigating the following approaches to extending the modality of the Llama 3.2 1B model to include speech:

1. Continued Pre-Training (CPT) - training a pre-trained, foundation model on interleaved or concatenated speech-text data.
2. Supervised Fine-Tuning (SFT) - training a pre-trained model on speech-text instruction-following data.
3. CPT followed by SFT - training a pre-trained model on interleaved speech-text data, followed by training on speech-text instruction-following data.

The speech-text datasets are obtained using labelled speech datasets and tokenizing the speech audio using a pre-trained speech tokenizer. We compare the following four speech tokenizers:

1. HuBERT
2. SpeechTokenizer
3. Mimi (from Moshi)
4. FocalCodec

Orthogonally, we investigate whether a tokenizer trained with Byte Pair Encoding (BPE) over the tokenized speech data can be leveraged to train a speech LM with compressed speech representations. 

In this initial phase of experiementation, we will only train on the Multilingual LibriSpeech dataset (MLS). 

## Experiements

The above approaches give rise to the following matrix of experiments:

| # | Approach | Tokenizer | Dataset | 
|---|---|---|---|
| 1 | CPT (interleaved) | HuBERT | MLS |
| 2 | CPT (interleaved) | SpeechTokenizer | MLS |
| 3 | CPT (interleaved) | Mimi | MLS |
| 4 | CPT (interleaved) | FocalCodec | MLS |
| 5 | CPT (concatenated) | HuBERT | MLS |
| 6 | CPT (concatenated) | SpeechTokenizer | MLS |
| 7 | CPT (concatenated) | Mimi | MLS |
| 8 | CPT (concatenated) | FocalCodec | MLS |
| 9 | SFT | HuBERT | MLS |
| 10 | SFT | SpeechTokenizer | MLS |
| 11 | SFT | Mimi | MLS |
| 12 | SFT | FocalCodec | MLS |
| 13 | CPT + SFT | HuBERT | MLS |
| 14 | CPT + SFT | SpeechTokenizer | MLS |
| 15 | CPT + SFT | Mimi | MLS |
| 16 | CPT + SFT | FocalCodec | MLS |

We perform the experiments again for the compressed speech representations obtained using BPE.

Note: We only consider the best-performing CPT approach (interleaved or concatenated) for each tokenizer when performing the CPT + SFT experiments.

## Evaluation

We evaluate the performance of the trained models using the following metrics:

1. Word Error Rate (WER) on the following datasets:
    - LibriSpeech (ASR)
    - Common Voice (ASR)
    - FLEURS (ASR)
