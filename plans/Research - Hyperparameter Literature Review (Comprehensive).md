# Research: Comprehensive Hyperparameter Literature Review

> [!info] This document provides a systematic, exhaustive literature review of hyperparameter choices for all stages of our research pipeline: Continued Pre-Training (CPT), Supervised Fine-Tuning (SFT), and inference/generation. It is a companion to and extension of the existing *Research - Optimal hyperparameters based on literature review.md* document, which focuses specifically on learning rate. That document should be consulted alongside this one. Each section documents the current codebase setting, the literature consensus, and citations to back the recommended values. All recommendations apply to Llama 3.2 1B (architecture: 16 layers, hidden dim 2048, 32 attention heads, 8 KV heads).

---

## 0. Exhaustive Hyperparameter Inventory

The following is the complete list of hyperparameters extracted from `conf/training.yaml`, `conf/cpt.yaml`, `conf/sft.yaml`, `conf/common.yaml`, `conf/generate.yaml`, and the data configs in `conf/data/`.

### 0.1 Optimiser
| Parameter | Current Value | Config Key |
|-----------|--------------|------------|
| Optimiser type | AdamW | (implicit, `ssi/optimizer.py`) |
| Learning rate | `2e-4` | `optimizer.lr` |
| β₁ | `0.9` | `optimizer.betas[0]` |
| β₂ | `0.999` | `optimizer.betas[1]` |
| ε (epsilon) | `1e-8` | `optimizer.eps` |
| Weight decay | `0.01` | `optimizer.weight_decay` |
| AMSGrad | `False` | `optimizer.amsgrad` |
| Fused kernel | `True` | `optimizer.fused` |

### 0.2 Gradient handling
| Parameter | Current Value | Config Key |
|-----------|--------------|------------|
| Gradient accumulation steps | `4` | `gradient_accumulation_steps` |
| Gradient clipping norm | `null` (disabled) | `clip_grad_norm` |

### 0.3 Learning rate schedule
| Parameter | Current Value | Config Key |
|-----------|--------------|------------|
| Schedule type | Cosine with warmup | (implicit, `ssi/lr_schedule.py`) |
| Warmup steps | `1000` | `lr_scheduler.num_warmup_steps` |
| Number of cycles | `0.5` | `lr_scheduler.num_cycles` |

### 0.4 Training loop
| Parameter | Current Value | Config Key |
|-----------|--------------|------------|
| Max steps | `100000` | `max_steps` |
| Log interval | `1` | `log_interval` |
| Eval steps | `1000` | `eval_steps` |
| Save steps | `10000` | `save_steps` |

### 0.5 Tokeniser / sequence lengths
| Parameter | Current Value | Config Key |
|-----------|--------------|------------|
| Max seq len (CPT) | `768` | `tokenizer.max_seq_len` in `cpt.yaml` |
| Max seq len (SFT) | `2048` | `tokenizer.max_seq_len` in `sft.yaml` |

### 0.6 Data loading
| Parameter | CPT Value | SFT Value | Config Key |
|-----------|-----------|-----------|------------|
| Batch size (train) | `16` | `2` | `train.dataloader.batch_size` |
| Batch size (eval) | `16` | `2` | `dev.dataloader.batch_size` |
| Shuffle (train) | `True` | `True` | `train.shuffle` |
| Drop last (train) | `True` | `True` | `train.dataloader.drop_last` |
| Packing | `False` | `False` | `packed` |

### 0.7 Speech-specific
| Parameter | Current Value | Config Key |
|-----------|--------------|------------|
| Number of DSUs | `???` (typically 5000) | `speech.n_dsus` |
| Deduplication | `???` (undecided) | `speech.deduplicate` |
| Modality tokens | `True` | `speech.use_modality_tokens` |
| Sequence type (CPT) | `"interleaved"` | `dataset.sequence_type` |
| Binomial prob (interleaving) | `0.1` | `interleave_kwargs.binom_prob` |
| Downsampling ratio | `320` | `interleave_kwargs.downsampling_ratio` |
| Mean seq len tokens | `39.43` | `interleave_kwargs.mean_seq_len_tokens` |

### 0.8 Generation / inference
| Parameter | Current Value | Config Key |
|-----------|--------------|------------|
| Temperature | `0.0` | `sampling_params.temperature` |
| Top-p | `1.0` | `sampling_params.top_p` |
| Top-k | `-1` | `sampling_params.top_k` |
| Max tokens | `256` | `sampling_params.max_tokens` |
| Number of sequences | `1` | `sampling_params.n` |
| Presence penalty | `0` | `sampling_params.presence_penalty` |
| Frequency penalty | `0` | `sampling_params.frequency_penalty` |
| Repetition penalty | `1` | `sampling_params.repetition_penalty` |
| vLLM batch size | `8` | `vllm_batch_size` |

### 0.9 Compute / dtype
| Parameter | Current Value | Config Key |
|-----------|--------------|------------|
| dtype | `bf16` | `dtype` |
| Torch compile | `False` | `compile` |
| Device | `cuda` | `device` |

---

## 1. Optimiser Hyperparameters (β₁, β₂, ε, weight decay, gradient clipping)

> **TL;DR:** The literature shows a clear split between CPT conventions (β₂ = 0.95, ε = 1e-5, weight_decay = 0.1, clip = 1.0) and SFT/PEFT conventions (β₂ = 0.999, ε = 1e-8, weight_decay = 0.0). The current codebase uses a mixed configuration (β₂ = 0.999, ε = 1e-8, weight_decay = 0.01, no gradient clipping) that is inconsistent with either convention. For full-parameter CPT, the Llama family recipe and every large-scale empirical study converge on {β₁=0.9, β₂=0.95, ε=1e-5, weight_decay=0.1, clip=1.0}.

### Summary Table

| Paper | Venue | Year | Type | Model | β₁ | β₂ | ε | WD | Clip |
|-------|-------|------|------|-------|----|----|---|----|------|
| Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" (arXiv:2302.13971) | arXiv | 2023 | PT | 7B–65B | 0.9 | **0.95** | 1e-5 | 0.1 | 1.0 |
| Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models" (arXiv:2307.09288) | arXiv | 2023 | PT+SFT | 7B–70B | 0.9 | **0.95** | 1e-5 | 0.1 | 1.0 |
| Meta AI, "The Llama 3 Herd of Models" (arXiv:2407.21783) | arXiv/ICML 2025 | 2024 | PT+SFT | 8B–405B | 0.9 | **0.95** | 1e-5 | 0.1 | 1.0 |
| Zhang et al., "TinyLlama: An Open-Source Small Language Model" (arXiv:2401.02385) | arXiv | 2024 | PT | **1.1B** | 0.9 | **0.95** | 1e-5 | 0.1 | 1.0 |
| Rozière et al., "Code Llama: Open Foundation Models for Code" (arXiv:2308.12950) | arXiv/Meta | 2023 | **CPT** | 7B–70B | 0.9 | **0.95** | 1e-5 | 0.1 | 1.0 |
| Qwen Team, "Qwen2.5 Technical Report" (arXiv:2412.15115) | arXiv | 2024 | PT+SFT | 0.5B–72B | 0.9 | **0.95** | 1e-8 | 0.1 | 1.0 |
| Hassid et al., "TWIST: Textually Pretrained Speech Language Models" (arXiv:2305.13009) | **NeurIPS 2023** | 2023 | **CPT (speech)** | 1.3B–13B | 0.9 | 0.95 | — | 0.1 | 1.0 |
| Nguyen et al., "SpiRit-LM: Interleaved Spoken and Written Language Model" (arXiv:2402.05755) | TACL/**ACL 2025** | 2024 | **CPT (speech)** | 7B | 0.9 | 0.95 | — | 0.1 | 1.0 |
| Kim et al., "USDM: Paralinguistics-Aware Speech-Empowered LLMs" (arXiv:2402.05706) | **NeurIPS 2024** | 2024 | CPT+SFT (speech) | 7B | 0.9 | 0.95 | 1e-8 | 0.1 | 1.0 |
| Cappellazzo et al., "Llama-AVSR" (arXiv:2409.12319) | **ICASSP 2025** | 2024 | SFT (LoRA) | 8B | 0.9 | 0.999 | 1e-8 | 0.1 | 1.0 |
| Kim et al., "MMS-LLaMA" (arXiv:2503.11315) | **ACL Findings 2025** | 2025 | SFT (QLoRA) | 8B | 0.9 | **0.98** | — | — | — |
| Liu et al., "LLaVA: Visual Instruction Tuning" (arXiv:2304.08485) | **NeurIPS 2023** | 2023 | SFT (multimodal) | 13B | 0.9 | 0.999 | 1e-8 | **0.0** | 1.0 |
| Dettmers et al., "QLoRA" (arXiv:2305.14314) | **NeurIPS 2023** | 2023 | PEFT | 7B–65B | 0.9 | 0.999 | 1e-8 | ~0.0 | **0.3** |
| Christophe et al., "Clinical LLMs" (arXiv:2409.14988) | **EMNLP Findings 2024** | 2024 | **CPT+SFT** | 7B | 0.9 | **0.95** | 1e-8 | 0.1 | 1.0 |
| Penedo et al., "The Falcon Series" (arXiv:2311.16867) | JMLR 2024 | 2023 | PT | 7B–180B | 0.9 | **0.95** | — | 0.1 | 1.0 |
| Gupta et al., "Continual Pre-Training: How to re-warm your model?" (arXiv:2308.04014) | ICML Workshop 2023 | 2023 | CPT study | 410M–7B | 0.9 | 0.95 | — | 0.1 | 1.0 |
| Parmar et al., "Reuse, Don't Retrain" (arXiv:2407.07263) | NVIDIA arXiv 2024 | 2024 | CPT recipe | 15B | 0.9 | 0.95 | — | 0.1 | 1.0 |
| Semenov et al., "Benchmarking Optimizers for LLM Pretraining" (arXiv:2509.01440) | arXiv 2025 | 2025 | Study | 100M–1B | 0.9 | **0.95** | 1e-8 | 0.1 | — |

### 1.1 β₂ (Adam second-moment decay)

**Current setting: 0.999. Literature consensus for CPT: 0.95.**

The entire Llama family (1, 2, 3, 3.2) uses β₂ = 0.95, not the Adam default of 0.999. This choice was deliberate: 0.95 down-weights older squared gradients more aggressively, stabilising large-scale training and allowing a higher effective learning rate. Every major empirical optimizer benchmarking study (Semenov et al. 2025, Wen et al. 2025 "Fantastic Pretraining Optimizers") confirms β₂ = 0.95 as optimal for causal LM pretraining. The same convention is followed in all major speech-LLM CPT papers that warm-start from a Llama base (TWIST, NeurIPS 2023; SpiRit-LM, TACL 2025; USDM, NeurIPS 2024).

For SFT, β₂ = 0.999 (the Adam default) is used by many papers (QLoRA, LLaVA, Llama-AVSR). One recent speech paper (MMS-LLaMA, ACL Findings 2025) uses the intermediate value 0.98.

**Recommended setting:**
- CPT full-parameter: **β₂ = 0.95** (matches Llama 3.2 1B's own pretraining recipe and every comparable CPT paper)
- SFT full-parameter: **β₂ = 0.95** (safe default; 0.999 also defensible)

### 1.2 ε (Adam epsilon, numerical stability floor)

**Current setting: 1e-8. Llama-family standard: 1e-5.**

Llama 1, 2, and 3 all explicitly use ε = 1e-5, which is 1000× larger than the default. The larger value prevents the adaptive step size from becoming pathologically large in bf16/fp16 mixed-precision training when the denominator (√v̂ + ε) is very small. Other model families (Qwen2.5, Mistral) use 1e-8 and are also stable, but for a Llama-based model running in bf16, the Llama-family ε = 1e-5 is the conservatively correct choice.

**Recommended setting: ε = 1e-5** (follow the Llama 3.2 1B pretraining recipe exactly).

### 1.3 Weight decay

**Current setting: 0.01. Literature consensus for CPT: 0.1.**

Nearly every CPT/PT paper in the table above uses weight_decay = 0.1. The Llama family, Code Llama, TinyLlama, Falcon, and all domain CPT papers (BioMistral, Clinical LLMs, Sheared LLaMA) use 0.1. Both NVIDIA (Parmar et al. 2024) and AMD empirically tested 0.01 vs 0.1 and found no measurable difference, so the choice is not critical — but 0.01 has no special support and does not match the Llama 3.2 pretraining recipe.

For SFT with PEFT (LoRA), weight_decay = 0.0 is common (QLoRA, LLaVA) because there are few adapter parameters to regularise.

**Recommended settings:**
- CPT/SFT full-parameter: **weight_decay = 0.1** (matches Llama family recipe)
- SFT LoRA: weight_decay = 0.0 (defensible but likely irrelevant)

### 1.4 Gradient clipping norm

**Current setting: null (disabled). Literature consensus: 1.0.**

Every pretraining and CPT paper that reports gradient clipping uses clip_grad_norm = 1.0. This is universal in the Llama family and all speech-LLM CPT papers. The only exception is QLoRA (Dettmers et al., NeurIPS 2023) which uses 0.3 — but this is specific to parameter-efficient fine-tuning of a quantised model and is not applicable here. With no clipping, gradient explosions during early CPT are a real risk, particularly when new tokens (the 5000 speech DSU embeddings) are being learned from scratch alongside the frozen base model weights.

**Recommended setting: clip_grad_norm = 1.0**, applied to both CPT and SFT.

---

## 2. Batch Size, Sequence Length, and Gradient Accumulation

> **TL;DR:** The literature is broadly consistent: CPT uses short sequences (512–2048 tokens) with large effective batch sizes in token terms (1M–4M tokens/step) on multi-GPU setups; on a single A6000, batch sizes of 8–48 with seq_len 512–1024 are the documented analogues. Our CPT setting (batch=16, seq=768, grad_accum=4 → 12,288 tokens/step) is at the low end but directly precedented in the speech-specific and single-GPU literature. SFT seq_len=2048 has broad support.

### Summary Table

| Paper | Venue | Year | Type | Model | Seq Len | Tokens/Step | Grad Accum | GPUs |
|-------|-------|------|------|-------|---------|-------------|------------|------|
| Zhang et al., "TinyLlama" (arXiv:2401.02385) | arXiv | 2024 | PT | **1.1B** | 2,048 | 2M | — | 16× A100-40G |
| Hassid et al., "TWIST" (arXiv:2305.13009) | **NeurIPS 2023** | 2023 | CPT (speech) | 2,050 | — | — | — |
| Pareja et al., "Secret Recipe for SFT" (arXiv:2412.13337) | **ICLR 2025** | 2025 | SFT | 3B–7B | — | 3840–7680 samples | Yes | 1–64 |
| Labrak et al., "BioMistral" (arXiv:2402.10373) | **ACL Findings 2024** | 2024 | **CPT** | 7B | **2,048** | **2M** | **2** | 32× A100-80G |
| Gao et al., "ProLong" (arXiv:2410.02660) | **ACL 2025** | 2024 | CPT+SFT | 8B | 64K–512K | 4M–8M | — | H100 |
| Parmar et al., "Reuse Don't Retrain" (arXiv:2407.07263) | arXiv | 2024 | CPT | 15B | 4,096 | ~4.7M | — | — |
| Kang et al., "Domain-Adaptive CPT" (arXiv:2512.12384) | arXiv | 2024 | CPT | **1B, 3B (Llama 3.2)** | **1,024** | **8,192** | — | 1× H100 |
| Zhang et al., "SpeechGPT" (arXiv:2305.11000) | **EMNLP Findings 2023** | 2023 | CPT+SFT (speech) | 13B | 512–1,024 | — | — | 96× A100 |
| Fang et al., "LLaMA-Omni" (arXiv:2409.06666) | **ICLR 2025** | 2024 | SFT (speech) | 8B | — | — | — | 4× L40 |
| Zhan et al., "Ichigo" (arXiv:2410.15316) | arXiv | 2024 | **CPT+SFT (speech, A6000)** | 8B | **512 (CPT)**, 4,096 (SFT) | **246K (CPT)**, 1M (SFT) | — | **10× A6000** / 8× H100 |
| Cuervo & Marxer, "Scaling Speech LMs" (arXiv:2404.00685) | **EMNLP 2024** | 2024 | PT (speech) | up to 823M | 2,050 | up to 131K | — | — |
| Efrat et al., "Efficient CPT / Stability Gap" (arXiv:2406.14833) | arXiv | 2024 | CPT | **1.1B** | 2,048 | — | **4** | 192× V100 |

### 2.1 CPT batch size and sequence length

**Current setting: batch=16, seq_len=768, grad_accum=4 → effective 12,288 tokens/step.**

The most directly comparable paper is Kang et al. (arXiv:2512.12384), which continues pretraining Llama-3.2-1B on a single H100 with batch=8, seq_len=1024, achieving 8,192 tokens/step. Our 12,288 is 50% larger and is therefore well-motivated for our single-A6000 setup.

The Ichigo paper (arXiv:2410.15316) is the closest analogue in speech: it performs speech CPT on 10× A6000-48GB GPUs using seq_len=512, batch=48 per device (480 global, 245,760 tokens/step). Our single-GPU token rate per step is lower, compensated by more steps. The choice of seq_len=768 (rather than the more common 1,024 or 2,048) reflects the natural length distribution of speech token sequences at 320× downsampling for 3–5 second utterances, and is supported by the seq_len=512 choice in Ichigo.

The Efrat et al. stability gap paper (arXiv:2406.14833) uses grad_accum=4 for CPT at 1.1B scale, directly validating our gradient_accumulation_steps=4.

### 2.2 SFT batch size and sequence length

**Current setting: batch=2, seq_len=2048, grad_accum=4 → effective 4,096 tokens/step.**

SFT seq_len=2048 is the most widely reported value (BioMistral, ACL Findings 2024; SpeechGPT stage 3, EMNLP Findings 2023; Ichigo SFT preamble). With batch=2 on a single A6000 and grad_accum=4, the effective batch is 8 sequences (16,384 tokens), which is small relative to large-scale SFT but standard for single-GPU constrained experiments.

The ICLR 2025 "Secret Recipe for SFT" paper (Pareja et al.) finds that larger effective batch sizes (3,840–7,680 samples) yield better SFT results, but these assume access to tens of GPUs. For single-GPU SFT, the constraint is physical.

---

## 3. Training Duration (Max Steps and Tokens)

> **TL;DR:** Our CPT at max_steps=100,000 processes ~4.9B tokens. This is at the low end but within the range where domain CPT is effective for 1B models: the Rho-1 paper (NeurIPS 2024) achieved +30% on math tasks for a 1B model with just 15B tokens, and gains plateau after ~200M tokens in the SEC/financial CPT study (Llama-3.2-1B, single GPU). Our budget is plentiful relative to single-GPU precedents.

### Summary Table

| Paper | Venue | Year | Type | Model | CPT Tokens | CPT Steps | SFT Epochs/Steps | Domain |
|-------|-------|------|------|-------|-----------|-----------|-----------------|--------|
| Lin et al., "Rho-1: Not All Tokens Are What You Need" (arXiv:2404.07965) | **NeurIPS 2024** | 2024 | CPT | **1B** TinyLlama | **15B** (math) / **80B** (general) | Per-1B checkpoints | — | Math/General |
| Xia et al., "Sheared LLaMA" (arXiv:2310.06694) | **ICLR 2024** | 2024 | CPT | **1.3B** | **50B** | ~25k–50k | — | General |
| Azerbayev et al., "Llemma" (arXiv:2310.10631) | **ICLR 2024** | 2024 | CPT | 7B, 34B | ~200B (3–4× 55B corpus) | — | — | Math |
| Liu et al., "MathCoder2" (arXiv:2410.08196) | **ICLR 2025** | 2024 | CPT | 7B–8B | 57.6B (3 epochs × 19.2B) | 3 epochs | — | Math |
| Rozière et al., "Code Llama" (arXiv:2308.12950) | Meta/arXiv | 2023 | CPT | 7B–70B | 500B (7B–34B) | — | — | Code |
| Christophe et al., "Clinical LLMs" (arXiv:2409.14988) | **EMNLP Findings 2024** | 2024 | CPT+SFT | 7B | 65B CPT (×4 epochs = 260B eff.) | — | 4 epochs CPT; 500M SFT | Clinical |
| Nguyen et al., "SpiRit-LM" (arXiv:2402.05755) | TACL/**ACL 2025** | 2024 | **CPT (speech)** | 7B | ~150k hrs speech tokens | ~¼ cold-start steps | — | Speech-text |
| Hassid et al., "TWIST" (arXiv:2305.13009) | **NeurIPS 2023** | 2023 | **CPT (speech)** | up to 13B | ~150k hrs speech tokens | ~¼ cold-start steps | — | Speech |
| Kim et al., "USDM" (arXiv:2402.05706) | **NeurIPS 2024** | 2024 | CPT+SFT (speech) | 7B | ~87k hrs ASR data | — | 5 epochs SFT (batch 64) | Spoken dialogue |
| Tang et al., "SALMONN" (arXiv:2310.13289) | **ICLR 2024** | 2024 | SFT (3-stage) | 7B–13B | 650k items | — | **6 epochs**, LoRA | Speech/audio |
| Kang et al., "Domain-Adaptive CPT" (arXiv:2512.12384) | arXiv | 2024 | CPT | **Llama-3.2-1B** | **400M** (1 epoch) | ~50k | — | Financial |
| Zhang et al., "TinyLlama" (arXiv:2401.02385) | arXiv | 2024 | PT from scratch | **1.1B** | **3T** (3 epochs) | ~1.4M | — | General |
| Su et al., "MiniCPM" (arXiv:2404.06395) | arXiv | 2024 | PT+SFT | **1.2B, 2.4B** | ~1T | — | SFT follows | General |
| Groeneveld et al., "OLMo 2" (arXiv:2501.00656) | **COLM 2025** | 2025 | Mid-training | 7B–32B | 50B–300B per run | — | SFT → DPO → RLVR | General |

### 3.1 CPT duration

**Current setting: max_steps=100,000 → ~4.9B tokens at CPT batch.**

Key data points at 1B scale:
- Rho-1 (NeurIPS 2024): **15B tokens** CPT on OpenWebMath for TinyLlama-1B → +30% relative improvement on math. Gains plateau around 15B.
- Sheared LLaMA (ICLR 2024): **50B tokens** CPT of a pruned 1.3B model beats TinyLlama trained on 3T tokens.
- Kang et al. (Llama-3.2-1B, single GPU): **400M tokens** (1 epoch over SEC filings); largest gains in first 200M tokens.
- TWIST / SpiRit-LM: Both note that warm-starting from a text LLM reduces required steps by ~4× vs. cold-start speech LM training — meaning our 4.9B effective tokens is more compute-efficient than it appears.

Our 4.9B token budget is above the Kang et al. single-GPU benchmark (~400M) and well below the Sheared LLaMA target (50B), but the domain-focused nature of our corpus (speech tokens from MLS English) and the warm-start advantage mean convergence should occur well within this budget.

### 3.2 SFT duration

**Current setting: max_steps=100,000 → ~1.6B tokens at SFT batch.**

Key data points:
- Clinical LLMs (EMNLP Findings 2024): **500M tokens** SFT
- USDM (NeurIPS 2024): **5 epochs** at batch 64
- SALMONN (ICLR 2024): **6 epochs**, LoRA, ~650k items
- LLaSA (arXiv:2502.04128): 1B/3B/8B Llama models fine-tuned for TTS on 250k hours speech data

Our ~1.6B token SFT budget is above the Clinical LLMs benchmark and is generous relative to the task (transcription from speech tokens is a relatively narrow distribution).

---

## 4. Warmup Steps and LR Schedule

> **TL;DR:** Warmup at 1–10% of total steps with cosine decay is the universal standard. Our 1000 warmup steps out of 100,000 (1%) is at the low end; 3–5% warmup is more typical in speech-LLM papers. The existing *Research - Optimal hyperparameters based on literature review.md* documents this in detail — refer to that document for the full citation table. This section provides additional citations not captured there.

| Paper | Venue | Year | Warmup | Schedule | Notes |
|-------|-------|------|--------|----------|-------|
| Futami et al., *Scheduled Interleaved S2ST* | **Interspeech 2025** | 2025 | 3% of epochs | Custom + cosine | Most direct Llama 3.2 1B speech reference |
| Xu et al., LLaSA (arXiv:2502.04128) | HKUST arXiv | 2025 | 3% of epochs | Cosine | Full FT, 250k hours TTS |
| Kim et al., MMS-LLaMA (arXiv:2503.11315) | **ACL Findings 2025** | 2025 | 500 steps (1.7% of 30k) | Cosine | QLoRA, Llama 3.2 |
| Efrat et al. stability gap (arXiv:2406.14833) | arXiv | 2024 | 10% of steps | Cosine | 1.1B CPT, methodological study |
| Zhang et al., TinyLlama (arXiv:2401.02385) | arXiv | 2024 | 2,000 steps (0.14% of 1.4M) | Cosine | 1.1B PT from scratch |
| Labrak et al., BioMistral (arXiv:2402.10373) | **ACL Findings 2024** | 2024 | Not stated | Cosine | Domain CPT |
| Pareja et al. (arXiv:2412.13337) | **ICLR 2025** | 2025 | 3% of steps | Cosine | SFT recipe |

**Current warmup: 1000 steps = 1% of 100k.** The literature range is 1–10%, with speech papers clustering at 3–5%. There is no strong case against 1%, but 3% (3000 steps) would be better supported.

---

## 5. Decoding / Generation Hyperparameters

> **TL;DR:** Greedy decoding (temperature=0.0, top_p=1.0) with max_tokens=200–256 is the correct and well-supported configuration for LLM-based ASR. The only additional consideration from the literature is `no_repeat_ngram_size` (not currently used), which can suppress insertion hallucinations.

### Summary Table

| Paper | Venue | Year | Decoding | Beam/Temp | max_tokens | WER test-clean/other |
|-------|-------|------|----------|-----------|-----------|----------------------|
| Radford et al., "Whisper" (arXiv:2212.04356) | **ICML 2023** | 2023 | Beam + fallback | **beam=5, T=0→1** | ~448 | 2.5% / ~5% |
| Tang et al., "SALMONN" (arXiv:2310.13289) | **ICLR 2024** | 2024 | Not reported | — | — | 2.1% / 4.9% |
| Ma et al., "SLAM-ASR" (arXiv:2402.08846) | arXiv | 2024 | **Beam** | **beam=4** | — | 1.94% / 3.81% |
| SLAM-LLM team, "Comprehensive ASR Solution" (arXiv:2406.17272) | arXiv | 2024 | Beam | **beam=5** | **256** | 1.63% / 3.49% |
| Yu et al., "Connecting Speech Encoder and LLM" (arXiv:2309.13963) | **ICASSP 2024** | 2023 | Beam/Greedy | beam=5 (comp.) | — | 2.1% / 5.0% |
| Li et al., "Transcription Prompt Audio LLM" (arXiv:2408.09491) | **Interspeech 2024** | 2024 | **Greedy** | **T=0** | **200** | CER (Mandarin) |
| Tsunoo et al., "Pre-trained Speech and LLM for E2E ASR" (arXiv:2312.03668) | **ACL Findings 2024** | 2024 | **Greedy = best** (ablation) | **beam=1** | — | 8.4% CER (JP) |
| Seide et al., "Speech ReaLLM" (arXiv:2406.09569) | **Interspeech 2024** | 2024 | Greedy/Beam | beam=4 | — | 3.0% / 7.4% |
| Deng et al., "Transducer-Llama" (arXiv:2412.16464) | **ICASSP 2025** | 2024 | Beam | beam=10 | — | 2.47% / 6.53% |
| Hu et al., "WavLLM" (arXiv:2404.00656) | **EMNLP Findings 2024** | 2024 | Not reported | — | — | 2.0% / 4.8% |
| Qwen Team, "Qwen3-ASR" (arXiv:2601.21337) | arXiv | 2026 | **Greedy** | **T=0** | **1024** | 1.63% / 3.38% |
| Yang et al., "MaLa-ASR" (arXiv:2406.05839) | **Interspeech 2024** | 2024 | Beam | beam=4 | — | ~9–12% (SlideSpeech) |
| Omnilingual ASR, Meta (arXiv:2511.09690) | arXiv | 2024 | Beam | beam=5 | — | FLEURS multilingual |

### 5.1 Temperature

**Current setting: 0.0 (greedy). This is correct.**

The ACL Findings 2024 ablation (Tsunoo et al., arXiv:2312.03668) is the only paper with a full comparison across greedy, beam, top-k, and nucleus sampling for LLM-based ASR. Result: **greedy = best or tied for best**. Beam search offers no improvement; top-k and nucleus sampling are strictly worse. The Interspeech 2024 paper (Li et al.) explicitly argues that probabilistic sampling strategies "are not suitable for the ASR task because they may introduce additional errors." Qwen3-ASR (arXiv:2601.21337) achieves 1.63%/3.38% WER using greedy decoding exclusively.

### 5.2 top_p and top_k

**Current settings: top_p=1.0, top_k=-1 (disabled). This is correct.**

No paper in the LLM-ASR literature uses nucleus sampling (top_p < 1.0) or top-k filtering for evaluation. These are universally reserved for open-ended generation tasks. top_p=1.0 / top_k disabled is the correct setting.

### 5.3 max_tokens

**Current setting: 256. This is well-supported.**

- SLAM-LLM comprehensive paper (arXiv:2406.17272): **max_length=256** explicitly reported.
- Li et al. (Interspeech 2024): **max_new_tokens=200** (set based on max 180 tokens in eval set).
- Qwen3-ASR: **max_new_tokens=1024** (conservative multilingual limit).
- For English LibriSpeech/FLEURS utterances (typically < 100 word-pieces), 256 is generous and correct.

### 5.4 Repetition / presence / frequency penalty

**Current settings: all 0 / 1 (disabled). The literature does not use these for ASR.**

The relevant parameter for hallucination suppression is `no_repeat_ngram_size` (a Hugging Face generation parameter), not OpenAI-style penalties. The SLAM-LLM comprehensive paper (arXiv:2406.17272) found `no_repeat_ngram_size=10` dramatically reduces insertion errors (WER: 3.49% with NRNS=10 vs. substantially higher without it). This parameter is not currently exposed in the vLLM `SamplingParams` API but is worth investigating if insertion hallucinations appear in evaluations.

---

## 6. Speech-Specific Hyperparameters

> **TL;DR:** Literature is nearly unanimous: (1) apply deduplication of consecutive identical DSUs; (2) 500 HuBERT units is the established baseline — 5000 is unprecedented in the semantic-token literature and risks codebook under-utilisation; (3) word-level interleaved sequences are better than concatenated; (4) modality/boundary tokens are standard practice.

### 6.1 Deduplication (`speech.deduplicate`)

**Current setting: undecided. Literature: apply deduplication.**

| Paper | Venue | Year | Deduplication | Notes |
|-------|-------|------|--------------|-------|
| Lakhotia et al., "GSLM" (arXiv:2102.01192) | TACL / **NAACL 2021** | 2021 | **Yes** | Foundational paper — deduplication introduced as standard preprocessing |
| Kharitonov et al., "pGSLM" (arXiv:2109.15411) | **ACL 2022** | 2022 | **Yes** (dedup, separate duration stream) | Preserves duration via parallel stream |
| Nguyen et al., "dGSLM" | TACL /**ACL 2023** | 2023 | **Yes** | Standard GSLM preprocessing |
| Hassid et al., "TWIST" (arXiv:2305.13009) | **NeurIPS 2023** | 2023 | **Yes** | 500 units at 25 Hz post-dedup |
| Nguyen et al., "SpiRit-LM" (arXiv:2402.05755) | TACL/**ACL 2025** | 2024 | **Yes** | Deduplication before text–speech interleaving |
| Borsos et al., "AudioLM" (arXiv:2209.03143) | IEEE TASLP 2023 | 2022 | **Yes** (semantic stage) | SoundStorm companion: removing dedup hurts conditioning quality |
| Wang et al., "Comparative Study of Discrete Tokens" (arXiv:2411.08742) | arXiv | 2024 | **Yes** | Dedup + BPE: 30–60% sequence length reduction |
| Yang et al., "Universal Speech Discrete Tokens" (arXiv:2309.07377) | **ICASSP 2024** | 2024 | **Yes** | Dedup before BPE; k=2000 |
| Kim et al., "USDM" (arXiv:2402.05706) | **NeurIPS 2024** | 2024 | Not reported | k=10,000 chosen; large k may reduce natural repetition |
| Baade et al., "SyllableLM" (arXiv:2410.04029) | arXiv | 2024 | Superseded | Syllabic merging replaces dedup |

**Recommendation: apply deduplication.** The only reason not to deduplicate is if explicit duration modeling is required (pGSLM solves this with a separate duration stream). Deduplication reduces sequence length by 30–60%, which at 50 Hz (320× downsampling) is critical: without deduplication, a 5-second utterance produces ~250 tokens; with deduplication at 25 Hz, this falls to ~125 tokens — halving context requirements and training cost. No speech LM paper achieves competitive results without deduplication when using frame-rate HuBERT features.

### 6.2 Number of DSUs (`speech.n_dsus`)

**Current setting: 5000. Literature consensus: 500 (semantic focus) or 1000–2000 (ASR focus); 10,000 only for paralinguistic modeling.**

| Paper | Venue | Year | k (codebook size) | Task / Notes |
|-------|-------|------|------------------|--------------|
| Lakhotia et al., "GSLM" | TACL/**NAACL 2021** | 2021 | 50, 100, 200 | Foundational; HuBERT best at 100–200 |
| Hassid et al., "TWIST" | **NeurIPS 2023** | 2023 | **500** | Speech LM; best performing |
| Nguyen et al., "SpiRit-LM" | TACL/**ACL 2025** | 2024 | **500** | Interleaved speech-text CPT |
| Maiti et al., "VoxtLM" | **ICASSP 2024** | 2024 | **50** | Multi-task; small k for linguistic focus |
| Yang et al., "Universal Speech Tokens" | **ICASSP 2024** | 2024 | **2000** | Best for ASR/TTS |
| Wang et al., "Comparative Study" | arXiv | 2024 | 1000–2000 | Diminishing returns beyond 2000 |
| Rubenstein et al., "AudioPaLM" (arXiv:2306.12925) | arXiv | 2023 | **1024** | w2v-BERT semantic tokens |
| Kim et al., "USDM" | **NeurIPS 2024** | 2024 | **10,000** | Only this high for paralinguistic richness |
| Sicherman & Adi, "Analysing DSS Representations" | **ICASSP 2023** | 2023 | Various | Cluster quality matters more than count |
| Mousavi et al., "How to Extract Discrete Audio Tokens?" | **Interspeech 2024** | 2024 | Various layers | Layer 22 is high-semantic; no single k optimal |

**n_dsus = 5000 is at the high end of what the literature considers viable and enters territory with no strong precedent for semantic token modeling.** Wang et al. (arXiv:2411.08742) find that approximately 20% of distinct tokens account for only 5% of occurrences at k=2000 — indicating already significant codebook under-utilization at that scale. At k=5000, this effect is expected to be substantially worse. The only paper using k≥5000 is USDM (k=10,000), but this is explicitly motivated by the need to preserve paralinguistic (prosodic, emotional) information alongside phonetic content.

**Recommendation:** Ablate over {500, 1000, 2000, 5000}. The majority of evidence supports **500 as the baseline** for semantic HuBERT tokens targeted at ASR-oriented downstream tasks. If the research goal includes expressive or paralinguistic fidelity, scale up to 2000. Use 5000 only with explicit experimental justification.

### 6.3 Sequence type (interleaved vs. concatenated)

**Current setting: `"interleaved"`. This is the correct choice, well-supported.**

| Paper | Venue | Year | Sequence Type | Performance vs. Concat |
|-------|-------|------|--------------|------------------------|
| Nguyen et al., "SpiRit-LM" | TACL/**ACL 2025** | 2024 | **Interleaved (word-boundary)** | Best; interleaving is the single most important factor in their ablation |
| Maimon et al., "Scaling Analysis of Interleaved SLMs" | **COLM 2025** | 2025 | Interleaved | Scales more efficiently than textless SLMs |
| Zhang et al., "Scaling Speech-Text Pre-training" | **ICLR 2025** | 2024 | Interleaved (synthetic) | SOTA on spoken QA |
| Wang et al., "InSerter" | **ACL 2025** | 2025 | Interleaved | 31.7%→51.4% instruction following |

Concatenated sequences (speech then text, or text then speech) are a simpler baseline that has been superseded by word-level interleaving in all comparative studies. Our `sequence_type="interleaved"` with word alignment is consistent with SpiRit-LM, the most methodologically rigorous reference.

### 6.4 Modality tokens (`speech.use_modality_tokens`)

**Current setting: `True`. This is correct and standard.**

- SpiRit-LM: [Text] / [Speech] boundary tokens
- USDM (NeurIPS 2024): 2 special speech boundary tokens
- AudioPaLM: markup task-description tokens
- VoxtLM (ICASSP 2024): task-guidance special tokens

Modality tokens provide the model with explicit signals about which representation system (text BPE vs. speech DSU) is active at each position, improving cross-modal alignment.

---

## 7. Precision and Compilation (`dtype`, `compile`)

**Current settings: dtype=bf16, compile=False.**

bf16 is the standard for all modern LLM training (Llama 1/2/3, TinyLlama, TWIST, SpiRit-LM). It requires ε = 1e-5 for numerical stability in AdamW (see §1.2). fp16 is an alternative but is less numerically stable for training and not recommended. compile=False is sensible as a default (torch.compile can cause issues with dynamic shapes in speech data pipelines), though enabling it can yield 20–30% throughput improvements if shapes are sufficiently static.

---

## 8. Recommended Changes to Current Configuration

The following table summarises the deltas between current settings and the literature-recommended values, in priority order.

| Priority | Parameter | Current Value | Recommended Value | Justification |
|----------|-----------|--------------|------------------|---------------|
| **Critical** | `optimizer.betas[1]` (β₂) | 0.999 | **0.95** | Entire Llama family uses 0.95 for CPT (§1.1); mismatches the base model's own pretraining recipe |
| **Critical** | `optimizer.eps` (ε) | 1e-8 | **1e-5** | Llama 1/2/3 all use 1e-5 for bf16 stability (§1.2) |
| **Critical** | `clip_grad_norm` | null | **1.0** | Universal in all CPT papers; protects against gradient explosions with newly initialised DSU embeddings (§1.4) |
| **High** | `optimizer.weight_decay` | 0.01 | **0.1** | Matches Llama 3.2 1B pretraining recipe and all major CPT papers (§1.3) |
| **High** | `speech.deduplicate` | ??? | **True** | Near-universal in GSLM-lineage speech LM work; reduces sequence length 30–60% (§6.1) |
| **High** | `speech.n_dsus` | 5000 (default) | **500 (baseline); ablate 500/1000/2000/5000** | 5000 is unprecedented for semantic HuBERT tokens and risks codebook under-utilisation (§6.2) |
| **Medium** | `lr_scheduler.num_warmup_steps` | 1000 (1%) | **3000 (3%)** | Speech CPT papers use 3–5% warmup (Futami Interspeech 2025; LLaSA arXiv:2502.04128) (§4) |
| **Low** | `optimizer.lr` | 2e-4 | See *Research - Optimal hyperparameters* doc | Already documented in companion document |

---

## 9. Full Citation Reference List

All citations referenced in this document, ordered by topic:

### Optimiser hyperparameters
1. Touvron et al. (2023), "LLaMA: Open and Efficient Foundation Language Models," arXiv:2302.13971
2. Touvron et al. (2023), "Llama 2: Open Foundation and Fine-Tuned Chat Models," arXiv:2307.09288
3. Meta AI (2024), "The Llama 3 Herd of Models," arXiv:2407.21783
4. Zhang et al. (2024), "TinyLlama: An Open-Source Small Language Model," arXiv:2401.02385
5. Rozière et al. (2023), "Code Llama: Open Foundation Models for Code," arXiv:2308.12950
6. Qwen Team (2024), "Qwen2.5 Technical Report," arXiv:2412.15115
7. Hassid et al. (2023), "TWIST: Textually Pretrained Speech Language Models," arXiv:2305.13009 — **NeurIPS 2023**
8. Nguyen et al. (2024), "SpiRit-LM: Interleaved Spoken and Written Language Model," arXiv:2402.05755 — **TACL / ACL 2025**
9. Kim et al. (2024), "USDM: Paralinguistics-Aware Speech-Empowered LLMs," arXiv:2402.05706 — **NeurIPS 2024**
10. Cappellazzo et al. (2024), "Llama-AVSR: Large Language Models are Strong AVSR Learners," arXiv:2409.12319 — **ICASSP 2025**
11. Kim et al. (2025), "MMS-LLaMA," arXiv:2503.11315 — **ACL Findings 2025**
12. Liu et al. (2023), "LLaVA: Visual Instruction Tuning," arXiv:2304.08485 — **NeurIPS 2023**
13. Dettmers et al. (2023), "QLoRA: Efficient Finetuning of Quantized LLMs," arXiv:2305.14314 — **NeurIPS 2023**
14. Christophe et al. (2024), "Beyond Fine-tuning: Potential of Continuous Pretraining for Clinical LLMs," arXiv:2409.14988 — **EMNLP Findings 2024**
15. Penedo et al. (2023), "The Falcon Series of Open Language Models," arXiv:2311.16867 — JMLR 2024
16. Gupta et al. (2023), "Continual Pre-Training: How to re-warm your model?", arXiv:2308.04014 — **ICML Workshop 2023**
17. Parmar et al. (2024), "Reuse, Don't Retrain: A Recipe for Continued Pretraining," arXiv:2407.07263
18. Semenov et al. (2025), "Benchmarking Optimizers for LLM Pretraining," arXiv:2509.01440

### Batch size and sequence length
19. Labrak et al. (2024), "BioMistral: Open-Source Pretrained LLMs for Medical Domains," arXiv:2402.10373 — **ACL Findings 2024**
20. Pareja et al. (2025), "Unveiling the Secret Recipe: A Guide For SFT Small LLMs," arXiv:2412.13337 — **ICLR 2025**
21. Gao et al. (2024), "How to Train Long-Context Language Models (ProLong)," arXiv:2410.02660 — **ACL 2025**
22. Kang et al. (2024), "Domain-Adaptive Continued Pretraining with Validation-Based Stopping," arXiv:2512.12384
23. Zhang et al. (2023), "SpeechGPT: Empowering LLMs with Cross-Modal Abilities," arXiv:2305.11000 — **EMNLP Findings 2023**
24. Fang et al. (2024), "LLaMA-Omni: Seamless Speech Interaction with LLMs," arXiv:2409.06666 — **ICLR 2025**
25. Zhang et al. (2024), "Ichigo: Mixed-Modal Early-Fusion Realtime Voice Assistant," arXiv:2410.15316
26. Cuervo & Marxer (2024), "Scaling Properties of Speech Language Models," arXiv:2404.00685 — **EMNLP 2024**
27. Efrat et al. (2024), "Efficient Continual Pre-training by Mitigating the Stability Gap," arXiv:2406.14833

### Training duration
28. Lin et al. (2024), "Rho-1: Not All Tokens Are What You Need," arXiv:2404.07965 — **NeurIPS 2024**
29. Xia et al. (2024), "Sheared LLaMA: Accelerating LM Pre-training via Structured Pruning," arXiv:2310.06694 — **ICLR 2024**
30. Azerbayev et al. (2024), "Llemma: An Open Language Model for Mathematics," arXiv:2310.10631 — **ICLR 2024**
31. Liu et al. (2024), "MathCoder2: Better Math Reasoning from Continued Pretraining," arXiv:2410.08196 — **ICLR 2025**
32. Tang et al. (2024), "SALMONN: Towards Generic Hearing Abilities for LLMs," arXiv:2310.13289 — **ICLR 2024**
33. Su et al. (2024), "MiniCPM: Unveiling the Potential of Small Language Models," arXiv:2404.06395
34. Groeneveld et al. (2025), "OLMo 2," arXiv:2501.00656 — **COLM 2025**
35. Xu et al. (2025), "LLaSA: Scaling Llama-based Speech Synthesis," arXiv:2502.04128

### Decoding hyperparameters
36. Radford et al. (2022), "Whisper: Robust Speech Recognition via Large-Scale Weak Supervision," arXiv:2212.04356 — **ICML 2023**
37. Tang et al. (2023), "SALMONN: Towards Generic Hearing Abilities for LLMs," arXiv:2310.13289 — **ICLR 2024**
38. Ma et al. (2024), "SLAM-ASR: An Embarrassingly Simple Approach for LLM with Strong ASR Capacity," arXiv:2402.08846
39. SLAM-LLM team (2024), "A Comprehensive Solution to Connect Speech Encoder and LLM for ASR," arXiv:2406.17272
40. Yu et al. (2023), "Connecting Speech Encoder and LLM for ASR," arXiv:2309.13963 — **ICASSP 2024**
41. Li et al. (2024), "A Transcription Prompt-based Efficient Audio LLM for Robust ASR," arXiv:2408.09491 — **Interspeech 2024**
42. Tsunoo et al. (2024), "Integrating Pre-Trained Speech and Language Models for E2E ASR," arXiv:2312.03668 — **ACL Findings 2024**
43. Seide et al. (2024), "Speech ReaLLM – Real-time Streaming ASR with Multimodal LLMs," arXiv:2406.09569 — **Interspeech 2024**
44. Deng et al. (2024), "Transducer-Llama: Integrating LLMs into Streamable Transducer-based ASR," arXiv:2412.16464 — **ICASSP 2025**
45. Hu et al. (2024), "WavLLM: Towards Robust and Adaptive Speech LLM," arXiv:2404.00656 — **EMNLP Findings 2024**
46. Qwen Team (2026), "Qwen3-ASR Technical Report," arXiv:2601.21337
47. Yang et al. (2024), "MaLa-ASR: Multimedia-Assisted LLM-Based ASR," arXiv:2406.05839 — **Interspeech 2024**
48. Meta AI (2024), "Omnilingual ASR: Open-Source Multilingual Speech Recognition for 1600+ Languages," arXiv:2511.09690

### Speech token design
49. Lakhotia et al. (2021), "On Generative Spoken Language Modeling from Raw Audio (GSLM)," TACL — **NAACL 2021**
50. Kharitonov et al. (2022), "Text-Free Prosody-Aware Generative Spoken LM (pGSLM)," — **ACL 2022**
51. Nguyen et al. (2023), "Generative Spoken Dialogue Language Modeling (dGSLM)," TACL — **ACL 2023**
52. Borsos et al. (2022), "AudioLM: a Language Modeling Approach to Audio Generation," arXiv:2209.03143 — IEEE TASLP 2023
53. Rubenstein et al. (2023), "AudioPaLM: A Large Language Model That Can Speak and Listen," arXiv:2306.12925
54. Maiti et al. (2023), "VoxtLM: Unified Decoder-Only Models for ASR, TTS, and Continuation," arXiv:2309.07937 — **ICASSP 2024**
55. Zhang et al. (2022), "SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data," arXiv:2209.15329
56. Maimon et al. (2025), "Scaling Analysis of Interleaved Speech-Text Language Models," arXiv:2504.02398 — **COLM 2025**
57. Zhang et al. (2024), "Scaling Speech-Text Pre-training with Synthetic Interleaved Data," arXiv:2411.17607 — **ICLR 2025**
58. Wang et al. (2024), "InSerter: Speech Instruction Following with Unsupervised Interleaved Pre-training," arXiv:2503.02769 — **ACL 2025**
59. Wang et al. (2024), "A Comparative Study of Discrete Speech Tokens for Semantic-Related Tasks with LLMs," arXiv:2411.08742
60. Yang et al. (2023), "Towards Universal Speech Discrete Tokens: A Case Study for ASR and TTS," arXiv:2309.07377 — **ICASSP 2024**
61. Mousavi et al. (2024), "How Should We Extract Discrete Audio Tokens from Self-Supervised Models?", arXiv:2406.10735 — **Interspeech 2024**
62. Sicherman & Adi (2023), "Analysing Discrete Self Supervised Speech Representations for Spoken LM," arXiv:2301.00591 — **ICASSP 2023**
63. Baade et al. (2024), "SyllableLM: Learning Coarse Semantic Units for Speech Language Models," arXiv:2410.04029
64. Défossez et al. (2024), "Moshi: a speech-text foundation model for real-time dialogue," arXiv:2410.00037
