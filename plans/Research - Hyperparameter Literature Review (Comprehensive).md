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
| Meta AI, "The Llama 3 Herd of Models" (arXiv:2407.21783) | arXiv/ICML 2025 <!-- FACT-CHECK ERROR: venue is ICLR 2025, not ICML 2025. Confirmed via proceedings.iclr.cc/paper_files/paper/2025/file/90d1fc07f46e31387978b88e7e057a31-Paper-Conference.pdf --> | 2024 | PT+SFT | 8B–405B | 0.9 | **0.95** | 1e-5 | 0.1 | 1.0 |
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

The entire Llama family (1, 2, 3, 3.2) uses β₂ = 0.95, not the Adam default of 0.999. This choice was deliberate: 0.95 down-weights older squared gradients more aggressively, stabilising large-scale training and allowing a higher effective learning rate. Every major empirical optimizer benchmarking study (Semenov et al. 2025, Wen et al. 2025 "Fantastic Pretraining Optimizers") confirms β₂ = 0.95 as optimal for causal LM pretraining. The same convention is followed in all major speech-LLM CPT papers that warm-start from a Llama base (TWIST, NeurIPS 2023; SpiRit-LM, TACL 2025 <!-- FACT-CHECK NOTE: SpiRit-LM published in TACL vol. 13 (2025); presentation at ACL 2025 is not yet confirmed as of March 2026 but is plausible -->; USDM, NeurIPS 2024).

For SFT, β₂ = 0.999 (the Adam default) is used by many papers (QLoRA, LLaVA, Llama-AVSR). One recent speech paper (MMS-LLaMA, ACL Findings 2025) uses the intermediate value 0.98.

**Recommended setting:**
- CPT full-parameter: **β₂ = 0.95** (matches Llama 3.2 1B's own pretraining recipe and every comparable CPT paper)
- SFT full-parameter: **β₂ = 0.95** (safe default; 0.999 also defensible)

### 1.2 ε (Adam epsilon, numerical stability floor)

**Current setting: 1e-8. Llama-family standard: 1e-5.**

Llama 1, 2, and 3 all explicitly use ε = 1e-5, which is 1000× larger than the default. <!-- FACT-CHECK NOTE: Llama 1 and 2 are confirmed to use ε = 1e-5. For Llama 3 (arXiv:2407.21783), multiple secondary sources report ε = 1e-8; the claim that Llama 3 uses ε = 1e-5 could not be independently verified from the technical report via web search and should be checked directly against Table 3 of arXiv:2407.21783. --> The larger value prevents the adaptive step size from becoming pathologically large in bf16/fp16 mixed-precision training when the denominator (√v̂ + ε) is very small. Other model families (Qwen2.5, Mistral) use 1e-8 and are also stable, but for a Llama-based model running in bf16, the Llama-family ε = 1e-5 is the conservatively correct choice.

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
| Groeneveld et al., "OLMo 2" (arXiv:2501.00656) <!-- FACT-CHECK NOTE: the arXiv PDF lists Pete Walsh as first author (not Groeneveld); listed as "OLMo Team, Pete Walsh, Luca Soldaini, Dirk Groeneveld et al." --> | **COLM 2025** | 2025 | Mid-training | 7B–32B | 50B–300B per run | — | SFT → DPO → RLVR | General |

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
3. Meta AI (2024), "The Llama 3 Herd of Models," arXiv:2407.21783 <!-- FACT-CHECK ERROR: published at ICLR 2025, not ICML 2025 -->
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
34. Groeneveld et al. (2025), "OLMo 2," arXiv:2501.00656 — **COLM 2025** <!-- FACT-CHECK NOTE: first-listed author on arXiv is Pete Walsh, not Dirk Groeneveld; paper is credited to "OLMo Team" with Walsh and Soldaini preceding Groeneveld in the author list -->
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

---

## PI Review: Evaluation of Recommendations

*Reviewed by: Senior PI, Speech and Language Processing — 2026-03-17*

*Context: Llama 3.2 1B, continued pretraining and fine-tuning to integrate HuBERT k-means discrete speech units for ASR, single A6000 GPU, ~48h compute budget.*

---

### Overall Assessment

This is a competently assembled literature review with genuine breadth and a reasonable set of conclusions. The citation table approach is effective and shows real effort. However, several extrapolations warrant closer scrutiny, one recommendation carries non-trivial risk if applied mid-project, and the document is weaker in certain areas (n_dsus, deduplication for ASR, decoding completeness) than its confident tone suggests. The following sections address each evaluation criterion in order.

---

### 1. Cross-Domain Extrapolations: Validity Assessment

**What is well-supported:** The Llama-family optimizer citations (Llama 1/2/3, TinyLlama, Code Llama) are directly applicable and constitute a strong foundation. These are the same architecture family, same training paradigm (causal LM), and in the case of Code Llama, the same CPT methodology. Using them to support β₂=0.95 and ε=1e-5 is defensible.

**Where extrapolation is strained:** The inclusion of Christophe et al. (Clinical LLMs, EMNLP Findings 2024) in the optimizer table is the weakest entry. A clinical NLP CPT paper on 7B models contributes nothing that the Llama family, TinyLlama, or Code Llama citations do not already cover more directly. Its presence suggests the document is padding the evidence base rather than relying on the strongest citations. This is not harmful per se, but the document should not treat domain-CPT papers from medicine or finance as independent corroboration of optimizer choices when the same choices are already well-established in the base architecture's own technical reports.

**The AMD CPT Playbook and NVIDIA Parmar et al.** are blog/arXiv entries, not peer-reviewed papers. Their presence in the optimizer section is fine for background context but should not be cited alongside NeurIPS/ACL papers as equivalent evidence. The document does not distinguish these in its priority reasoning.

**Verdict:** The cross-domain extrapolations in §1 are mostly harmless and the conclusions would stand without them. Flag the clinical CPT citation as unnecessary padding; do not rely on it as independent evidence.

---

### 2. Model-Scale Extrapolations: Validity Assessment

**What is well-supported:** TinyLlama (1.1B) is the single most important citation in this document for optimizer and training duration questions, and the document correctly identifies it as such. The TinyLlama stability gap paper (arXiv:2406.14833) is the most methodologically rigorous 1B-scale CPT study in the reference list. Similarly, Kang et al. (Llama-3.2-1B, single GPU) at §2.1 and §3.1 is a genuine like-for-like comparison.

**Where scale extrapolation is questionable:** The Llama 3, Code Llama, BioMistral, and Falcon citations all operate at 7B–405B. The document correctly uses them to establish optimizer conventions but does not explicitly acknowledge that some hyperparameters *do* scale with model size. Specifically:

- **β₂ = 0.95 at 7B–405B does not automatically imply β₂ = 0.95 is optimal at 1B.** Semenov et al. (arXiv:2509.01440) is cited and does cover 100M–1B, which provides the only direct empirical confirmation at 1B scale. This should be foregrounded more prominently rather than listed equivalently with 70B-scale papers.
- **Warmup steps:** The document recommends 3000 steps based on Futami (Interspeech 2025, full FT, speech) and LLaSA. Both are relevant, but LLaSA trains on 250k hours of TTS data — a substantially different corpus size and domain from ASR CPT on MLS English. The "3% of epochs" framing from LLaSA may not translate cleanly to a step-count recommendation without knowing the total step count in those experiments.
- **Effective batch size comparisons in §2 are not normalized by model size.** TinyLlama uses 2M tokens/step on 16× A100-40G; our 12,288 tokens/step is 160× smaller. The claim that "our single-GPU token rate per step is lower, compensated by more steps" is true but glosses over whether the gradient signal quality is comparable. At very small effective batch sizes, gradient noise can prevent convergence to the same minima regardless of total token count.

**Verdict:** The scale extrapolations are acceptable as working assumptions but the 1B-scale-specific evidence (Semenov et al., TinyLlama stability gap, Kang et al.) should be weighted more heavily than the 7B+ citations when they diverge, and the document does not make this hierarchy explicit enough.

---

### 3. Potentially Harmful Recommendations

**Changing β₂ from 0.999 to 0.95 mid-project carries real risk and the document underplays this.**

This is the most important safety concern in the review. The document recommends β₂=0.95 as "Critical" priority, but does not discuss the operational implications of changing this parameter after training has already begun.

AdamW's second moment estimate v_t is an exponential moving average with decay rate β₂. If training has run for N steps with β₂=0.999, the v_t buffers encode a weighted history over the last ~1/(1-0.999)=1000 steps. Switching to β₂=0.95 mid-run discards this history abruptly: the new decay rate forgets 95% of accumulated gradient information per step, giving effective memory of only ~20 steps. This causes a transient spike in effective learning rate (because v_t is now stale and will be small relative to current gradients) followed by rapid adaptation. This can manifest as a loss spike that looks like instability, even if the long-run trajectory is correct.

**The document should explicitly state:** (a) if training has already started with β₂=0.999, measure the loss spike upon switching; (b) the safest practice is to reinitialise the optimizer state when changing β₂; (c) the recommended optimizer changes should be applied at the start of a new run, not retroactively patched into an existing checkpoint. This is not a reason to avoid the recommendation — β₂=0.95 is the correct target — but the researcher needs to know the mechanics.

Similarly, changing ε from 1e-8 to 1e-5 mid-run also resets the effective adaptive step sizes; this is less dangerous (it makes steps smaller, not larger) but is still a discontinuity.

**Verdict:** Mark the β₂ and ε changes as "apply at run start only, with optimizer state reset" to prevent inadvertent mid-run instability.

---

### 4. The n_dsus=5000 Recommendation: Evidence Adequacy

**The evidence does not strongly support the claim that 5000 is "problematic" — it supports the claim that 5000 is unprecedented, which is different.**

The document correctly notes that Wang et al. (arXiv:2411.08742) found ~20% of distinct tokens accounting for only 5% of occurrences at k=2000. The extrapolation to "this effect is expected to be substantially worse at k=5000" is plausible but is an extrapolation, not a measured result at k=5000. The document conflates "no strong precedent" with "likely harmful."

The actual picture from the literature is:

- k=50–200: semantic focus, best for speech LM (GSLM)
- k=500: best for TWIST/SpiRit-LM style interleaved CPT
- k=1000–2000: best for ASR and TTS (Yang et al. ICASSP 2024; Wang et al.)
- k=10,000: motivated by paralinguistic richness (USDM)

The gap from k=2000 to k=5000 is genuinely unexplored in the literature and the codebook under-utilization argument is the *strongest* available reason to be cautious at k=5000. However:

1. The codebook under-utilization argument assumes HuBERT features. If the k-means clustering was performed on a large and diverse corpus, k=5000 might simply have finer acoustic granularity without creating dead entries — this has not been measured for the specific setup in use.
2. For ASR specifically, finer acoustic granularity *could* be beneficial (capturing consonant distinctions that k=500 merges), though the burden of proof is on this claim.
3. The document's recommendation to ablate over {500, 1000, 2000, 5000} is the right call, but the framing should be "we have no evidence 5000 works, and moderate evidence 500–2000 is better" rather than "5000 is problematic."

**Verdict:** The ablation recommendation is correct. The framing overstates the case against 5000. The researcher should run the ablation with codebook utilization statistics (proportion of active codes, entropy of code usage distribution) as explicit dependent variables alongside WER.

---

### 5. The Deduplication Recommendation: Caveats for ASR

**The recommendation to apply deduplication is well-supported for speech language modeling but the ASR-specific caveats are absent.**

The citations are accurate: GSLM, pGSLM, TWIST, SpiRit-LM all apply deduplication for speech LM. The document is correct that deduplication reduces sequence length 30–60%, which is a genuine practical benefit.

However, **for ASR specifically, deduplication introduces a non-trivial information loss that the document does not acknowledge:**

1. **Phoneme duration information is destroyed.** Vowel length, geminate consonants, and speaking rate variation are encoded partly in the repetition count of consecutive identical tokens. After deduplication, a short /æ/ and a long /æ:/ map to the same sequence. For English LibriSpeech this may be acceptable; for other languages or atypical speech (dysarthria, foreign-accented speech), duration loss can affect transcription quality.

2. **The sequence compression argument cuts both ways.** Deduplication at k=500 with 50 Hz encoding reduces a 5-second utterance from ~250 to ~125 tokens. But at k=5000 with finer quantization, consecutive frames are already less likely to be identical (finer distinctions mean more state changes), so the compression benefit may be smaller than at k=500. The interaction between n_dsus and deduplication benefit is not addressed.

3. **The pGSLM solution (separate duration stream) is noted but dismissed.** For a project that later wants to scale to more natural speech tasks or non-English data, preserving the option to re-introduce duration modeling is worth considering in the architecture design.

4. No paper in the cited list tests ASR accuracy with and without deduplication at the same k value. The SLAM-ASR / SLAM-LLM papers cited in §5 (which achieve the best WER numbers in the decoding table) are continuous-embedding systems and do not use discrete HuBERT tokens at all, so their implicit "no deduplication" approach is not comparable.

**Verdict:** Apply deduplication for the initial experiments — the sequence length reduction justification is pragmatically strong and the evidence base from speech LM is clear. But register this as an assumption to revisit, and ensure the ablation over n_dsus also tracks the interaction with deduplication. If duration-sensitive downstream tasks are in scope, plan a dedup-off baseline.

---

### 6. Training Token Budget Analysis (§3): Rigor Assessment

**The analysis is useful but the comparisons are not fully fair and the uncertainty range is wider than presented.**

Specific concerns:

1. **The 4.9B token calculation assumes max_steps=100,000 runs to completion.** The document does not acknowledge that on a single A6000 with a 48-hour budget, the achievable step count depends on throughput (tokens/second). At batch=16, seq=768, gradient accumulation=4, a rough estimate for Llama 3.2 1B in bf16 on A6000 is ~5,000–8,000 tokens/second (depending on flash attention, packing, and overhead). At 8,000 tokens/second, 4.9B tokens takes ~170 hours — well beyond the 48-hour budget. Even at the optimistic end, 100k steps may not be achievable in 48 hours for CPT. The document should include a throughput estimate and a realistic achievable-steps projection for the stated budget.

2. **Comparing against Rho-1 (15B tokens, math domain, TinyLlama-1B) is questionable.** The Rho-1 result is for a highly specialized math CPT task. Domain concentration (math) means each token carries more signal per byte than in ASR speech token sequences, which are more repetitive. The per-token information content comparison is not addressed.

3. **The warm-start efficiency claim ("warm-starting reduces required steps by ~4×") is attributed to TWIST and SpiRit-LM but is not a quantified finding in either paper.** Both papers note qualitatively that warm-starting converges faster than cold-start speech LM training. Neither provides a 4× speedup number. This is a reasonable heuristic but should not be stated as a measured ratio.

4. **The SFT duration analysis (§3.2) is the weakest part.** The comparison to Clinical LLMs (500M tokens SFT on 7B) and USDM (5 epochs at batch 64) involves different task complexities, model sizes, and dataset characteristics. For ASR as a CPT→SFT pipeline, the SFT stage is relatively narrow-distribution (transcription from tokens), and the most directly relevant comparison would be the number of hours of training data, not raw token counts. An estimate of the MLS English subset being used (in hours) and the resulting token count would anchor this better.

**Verdict:** The training duration analysis establishes that the budget is "not obviously insufficient" but does not rigorously confirm it is sufficient. Add a throughput estimate for the A6000 setup and make the achievable-steps claim explicit. The 4× warm-start efficiency claim should be labelled as a heuristic, not a cited result.

---

### 7. Decoding Analysis (§5): Completeness Assessment

**The decoding section is the most complete section in the document. The greedy decoding recommendation is well-supported and the Tsunoo et al. ablation is the key citation. However, several items are missing:**

1. **Beam search should not be dismissed entirely without acknowledging the tradeoff in evaluation contexts.** For a research paper, reporting results with both greedy and beam=5 is standard practice, not because beam is expected to improve performance, but because most published baselines (including Whisper) use beam search. Reporting greedy-only makes fair comparison to published WER numbers harder. The document should note that greedy is used for development/speed, but beam=5 results should be reported in any publication.

2. **The no_repeat_ngram_size discussion is useful but incomplete.** The document correctly notes this is not available via vLLM SamplingParams. The practical implication — that if vLLM is the inference engine, this mitigation is unavailable — should be made explicit. If insertion hallucinations are a problem in practice, the researcher may need to either implement post-processing or switch to a Hugging Face generate() path for evaluation.

3. **Normalization is not discussed.** For ASR evaluation, whether to report WER on raw output versus normalized output (lowercased, punctuation stripped) is a major methodological decision. The WER numbers in the decoding table range from ~1.6% to ~12%, partly because some use normalized, some do not. The evaluation pipeline should standardize on a normalization scheme (e.g., Whisper-style normalization) and the document should recommend this explicitly.

4. **The max_tokens=256 recommendation** is well-supported for English LibriSpeech utterances, but the document should note that for long-form inference or multi-turn evaluation, this limit may need to increase. The current 256 is appropriate for the stated task but should be flagged as a setting to revisit if evaluation moves beyond clean LibriSpeech segments.

5. **No discussion of length penalty** for beam search evaluation contexts, nor of forced EOS handling, both of which affect WER in practice.

**Verdict:** The decoding analysis is sound for the core claim (greedy is correct for development). Strengthen it by (a) recommending beam=5 for final reported results to ensure comparability, (b) clarifying the normalization approach, (c) addressing the vLLM limitation for no_repeat_ngram_size.

---

### 8. Top 3–5 Priority Actions

Based on the totality of this review, the researcher should prioritize the following:

**Action 1 (Highest priority): Fix the optimizer configuration before starting any new run.**
Apply β₂=0.95, ε=1e-5, weight_decay=0.1, and clip_grad_norm=1.0 together, at the start of a fresh training run with optimizer state re-initialized. Do not patch these into an ongoing run. The β₂ change in particular is important: the current 0.999 is mismatched to the Llama family's own pretraining recipe and the existing CPT literature. If training has already completed under 0.999, the trained model may still be fine (the mismatch affects optimization dynamics, not the final architecture), but future runs must use the correct values.

**Action 2 (Highest priority): Determine n_dsus before investing significant compute.**
The choice of k (codebook size) affects every subsequent experiment. Allocate a small probe: train for 5,000–10,000 steps with k ∈ {500, 1000, 2000, 5000}, measure validation loss, WER on a small dev set, and codebook utilization (percentage of active codes, code entropy). This ablation should run before committing to any full 100k-step run. The investment is small relative to the cost of discovering mid-project that k=5000 has 40% dead codes.

**Action 3 (High priority): Confirm the effective throughput and achievable token budget.**
Run a 100-step benchmark with the actual configuration (batch=16, seq=768, grad_accum=4, bf16, fused AdamW, flash attention if enabled) and measure tokens/second. Multiply by 48 hours to get the realistic maximum token budget. If the result is substantially less than 4.9B tokens, scale down max_steps accordingly and revisit whether the training duration analysis in §3 still holds.

**Action 4 (High priority): Standardize evaluation methodology before comparing any results.**
Decide on (a) normalization scheme (recommend Whisper-style), (b) whether to report greedy or beam=5 or both, (c) which test sets (LibriSpeech test-clean, test-other) will be the primary metrics. These decisions should be fixed before running the first evaluation, not reverse-engineered from what gives the best numbers.

**Action 5 (Medium priority): Increase warmup to 3000 steps (3%) at the start of new runs.**
The evidence base for this is weaker than for the optimizer changes (the speech papers at 1B scale cluster around 3%), but the cost of increasing warmup from 1% to 3% is very low and the downside risk (insufficient warmup causing early instability) is real with newly initialized DSU embeddings. This should be bundled with the optimizer reset in Action 1.

---

### 9. Hypotheses Requiring Empirical Validation

The document implicitly raises several hypotheses that are not settled by the literature and should be treated as empirical questions to be answered by the project's own experiments:

**H1: β₂=0.95 converges faster or to a better loss minimum than β₂=0.999 at 1B scale for speech CPT.** The literature provides strong reason to believe this, but no paper directly compares the two values in a controlled experiment at this exact scale and task. The Semenov et al. benchmark is the closest but covers general LM pretraining from scratch, not CPT of a pre-trained model on speech tokens. *Test:* short parallel runs (10k steps) with both β₂ values, same LR and everything else equal.

**H2: Codebook utilization at k=5000 is substantially worse than at k=500 or k=1000 for the specific MLS English HuBERT features used in this project.** Wang et al. measured this at k=2000 for a different corpus and tokenizer. The result may not transfer. *Test:* compute and log the per-training-step histogram of active code indices. A Gini coefficient or entropy of the empirical code distribution will quantify this cheaply.

**H3: Deduplication improves WER on LibriSpeech for this model, over and above the sequence length reduction.** The speech LM literature assumes deduplication is beneficial, but these results are primarily for speech generation quality (ABX, sBLEU, continuation naturalness), not transcription accuracy. It is conceivable that for ASR with discrete tokens, no-dedup at a high k (5000) preserves acoustic detail that improves WER. *Test:* a paired comparison — same n_dsus, same everything else, dedup on vs. off — evaluated on WER. This is a cheap experiment with high informational value.

**H4: The 1% warmup (1000 steps) is sufficient to avoid early instability, given that DSU embeddings are newly initialized.** The document recommends 3% but does not present direct evidence that 1% causes instability in this setting. The newly initialized embeddings are the key risk factor. *Test:* Monitor gradient norm of the embedding layer specifically during the first 2000 steps of a run with 1000-step warmup. If gradients are well-behaved (no spikes), the recommendation to increase warmup is precautionary rather than essential.

**H5: The 4× warm-start efficiency advantage claimed for speech CPT from a pre-trained text LLM holds at this compute scale.** If true, 4.9B tokens is substantially more than is needed for the model to acquire the speech modality. If false (if the efficiency advantage is smaller), the training may need to run longer. *Test:* plot validation loss and WER on dev set every 5k steps and look for the convergence elbow. The location of this elbow relative to total compute will directly calibrate the warm-start efficiency for this specific setup.

---

### Summary

The document is a genuine, high-effort literature review that will serve the project well as a reference. The core optimizer recommendations (β₂=0.95, ε=1e-5, clip=1.0) are correct and well-supported. The most important qualifications are: (1) apply the optimizer changes only at a clean run start with state reset; (2) the n_dsus ablation is the single highest-uncertainty decision and should be resolved with a small probe before committing compute; (3) the deduplication recommendation is sound for speech LM but ASR-specific validation is needed; (4) the training duration analysis needs a throughput reality check against the 48-hour budget; (5) the decoding analysis needs a normalization decision and a note about beam search for publication-quality comparisons. None of these qualifications undermine the overall direction — they refine it.

---

## Fact-Check Annotations

*Fact-checked: 2026-03-17. Each claim in the priority list was verified via web search against primary sources (arXiv, ACL Anthology, NeurIPS/ICLR/ACL proceedings pages). Inline HTML comments (`<!-- FACT-CHECK ... -->`) have been added adjacent to erroneous claims in the document body above.*

| # | Claim | Paper / Location | Verified? | Correction / Note |
|---|-------|-----------------|-----------|-------------------|
| 1 | Published venue "arXiv/ICML 2025" | "The Llama 3 Herd of Models" arXiv:2407.21783 — Summary Table §1, §9 citation | **ERROR** | Published at **ICLR 2025**, not ICML 2025. Confirmed via proceedings.iclr.cc PDF at file/90d1fc07f46e31387978b88e7e057a31. Inline comment added. |
| 2 | NeurIPS 2023 | TWIST arXiv:2305.13009 | CORRECT | Confirmed via proceedings.neurips.cc and neurips.cc/virtual/2023/poster/71490. |
| 3 | TACL / ACL 2025 | SpiRit-LM arXiv:2402.05755 | CORRECT (TACL confirmed; ACL 2025 presentation plausible) | Confirmed in TACL vol. 13 (2025), pp. 30–52 (aclanthology.org/2025.tacl-1.2). ACL 2025 TACL-track presentation is scheduled but presentation assignment not independently confirmed in search results. |
| 4 | NeurIPS 2024 | USDM arXiv:2402.05706 | CORRECT | Confirmed via neurips.cc/virtual/2024/poster/95416 and proceedings.neurips.cc. |
| 5 | NeurIPS 2023 Oral | LLaVA arXiv:2304.08485 | CORRECT | Confirmed via GitHub repo tag "[NeurIPS'23 Oral]" and papers.nips.cc entry. |
| 6 | NeurIPS 2023 | QLoRA arXiv:2305.14314 | CORRECT | Confirmed via proceedings.neurips.cc and neurips.cc/virtual/2023/poster/71815. |
| 7 | ACL Findings 2024 | BioMistral arXiv:2402.10373 | CORRECT | Confirmed via aclanthology.org/2024.findings-acl.348. |
| 8 | ACL 2025 (long paper) | ProLong arXiv:2410.02660 | CORRECT | Confirmed via aclanthology.org/2025.acl-long.366. |
| 9 | NeurIPS 2024 Oral | Rho-1 arXiv:2404.07965 | CORRECT | Confirmed as NeurIPS 2024 Oral via neurips.cc/virtual/2024/oral/98004. Also noted as best paper runner-up. |
| 10 | ICLR 2024 | Sheared LLaMA arXiv:2310.06694 | CORRECT | Confirmed via proceedings.iclr.cc and GitHub "[ICLR 2024]" tag. |
| 11 | Authors "Ma et al." and WER 1.94%/3.81% | SLAM-ASR arXiv:2402.08846 | PARTIALLY VERIFIED | First author Ziyang Ma confirmed. WER values (1.94%/3.81%) stated in doc for test-clean/test-other not independently confirmed from web search snippets (no table content returned); the paper does report LibriSpeech SOTA-level results. |
| 12 | ICLR 2025 | LLaMA-Omni arXiv:2409.06666 | CORRECT | Confirmed via proceedings.iclr.cc PDF reference and openreview.net entry. |
| 13 | arXiv 2601.21337 (2026) | Qwen3-ASR | CORRECT | Confirmed: submitted January 29, 2026; WER 1.63%/3.38% on LibriSpeech test-clean/other with greedy decoding confirmed. |
| 14 | arXiv:2306.12925, Rubenstein et al. | AudioPaLM | CORRECT | Confirmed: Rubenstein et al. (30 authors), arXiv June 2023. No conference venue claimed in document (arXiv only). |
| 15 | "TACL / NAACL 2021" | GSLM Lakhotia et al. arXiv:2102.01192 | UNVERIFIED (TACL confirmed; NAACL 2021 presentation not independently confirmed) | TACL vol. 9 (2021) publication confirmed (aclanthology.org/2021.tacl-1.79). TACL papers from vol. 9 were presented at NAACL, ACL, or EMNLP 2021. The specific assignment of this paper to NAACL 2021 vs. EMNLP 2021 could not be confirmed from search results. The "NAACL 2021" claim should be verified directly against the NAACL 2021 programme. |
| 16 | TinyLlama uses β₂ = 0.95 | arXiv:2401.02385 — Summary Table §1 | CORRECT | Confirmed: TinyLlama uses AdamW with β₁=0.9, β₂=0.95, weight decay=0.1. |
| 17 | Llama 3 uses ε = 1e-5, weight_decay = 0.1, β₂ = 0.95 | arXiv:2407.21783 — Summary Table §1, §1.2 narrative | PARTIALLY VERIFIED — ε = 1e-5 is UNVERIFIED for Llama 3 | β₂ = 0.95 and weight_decay = 0.1 confirmed for Llama 3. ε = 1e-5 confirmed for Llama 1 and Llama 2. For Llama 3, secondary sources report ε = 1e-8 (not 1e-5); this claim requires direct verification against Table 3 of arXiv:2407.21783. Inline comment added in §1.2. |
| 18 | "Tang et al.", ICLR 2024 | SALMONN arXiv:2310.13289 | CORRECT | First author Changli Tang (ByteDance/Tsinghua) confirmed. ICLR 2024 acceptance confirmed via openreview.net and proceedings.iclr.cc. |
| 19 | EMNLP Findings 2023 | SpeechGPT arXiv:2305.11000 | CORRECT | Confirmed via aclanthology.org/2023.findings-emnlp.1055. |
| 20 | COLM 2025 | OLMo 2 arXiv:2501.00656 | CORRECT (venue); NOTE on authorship | COLM 2025 confirmed via openreview.net PDF header. However, document cites as "Groeneveld et al." while arXiv PDF lists Pete Walsh as first author (Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, ...). Inline comment added. |

### Summary of Errors Found

**Confirmed errors requiring correction:**

1. **Llama 3 venue** (two locations): The document states "arXiv/ICML 2025" for arXiv:2407.21783. The correct venue is **ICLR 2025**. This affects the Summary Table in §1, the citation list in §9 (item 3), and the associated text in §1.1 of the PI Review section. Inline comments added at each occurrence.

**Items requiring manual verification against primary sources:**

2. **Llama 3 ε = 1e-5**: Web search was unable to confirm this. Llama 1 and 2 use 1e-5; secondary sources for Llama 3 report 1e-8. Verify directly against Table 3 of arXiv:2407.21783 (ICLR 2025 version). If Llama 3 uses 1e-8, then the recommendation in §1.2 and the PI Review §1 ("Using them to support β₂=0.95 and ε=1e-5 is defensible") are not undermined (since Llama 1/2 still support 1e-5), but the claim "Llama 1, 2, and 3 all explicitly use ε = 1e-5" would be inaccurate.

3. **GSLM NAACL 2021 presentation**: TACL vol. 9 (2021) confirmed; specific conference presentation at NAACL 2021 not confirmed. Could equally have been EMNLP 2021.

4. **OLMo 2 first author**: Document cites as "Groeneveld et al." — Pete Walsh is the first-listed author in the arXiv PDF. This is a minor citation convention issue; Dirk Groeneveld is a co-author.

---

## Additional References (Research Supplement)

*Compiled: 2026-03-17. This section documents papers found through a targeted gap-filling search of ICLR 2024/2025, ICML 2024/2025, NeurIPS 2024/2025, ACL/EMNLP/NAACL 2024/2025, Interspeech 2024/2025, and ICASSP 2024/2025 that are NOT already cited in the body of this document. Citations are grouped by the gap they address.*

---

### Gap 1: Optimizer comparisons for LLM fine-tuning (memory-efficient Adam variants)

**[A] Zhang et al. (2024), "Adam-mini: Use Fewer Learning Rates To Gain More," arXiv:2406.16793 — ICLR 2025**

- Authors: Yushun Zhang, Congliang Chen, Ziniu Li, Tian Ding, Chenwei Wu, Diederik P. Kingma, Yinyu Ye, Zhi-Quan Luo, Ruoyu Sun
- Proposes Adam-mini, which partitions parameters into blocks following Hessian structure and assigns one learning rate per block, reducing the v-term memory by at least 99.9% vs. standard Adam.
- Key values: performs on par with or better than AdamW across models 39M–13B for pretraining, SFT, and RLHF; achieves 49.5% higher throughput than AdamW on Llama-2 7B pretraining (2× A800-80G).
- Relevance (Gap 1): provides an empirically validated AdamW alternative that preserves β₂/ε conventions at 1B scale. Does not change the recommended β₂=0.95 target, but establishes that optimizer choice can meaningfully affect throughput on memory-limited hardware (A6000 48GB).

**[B] Huang et al. (2025), "SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training," arXiv:2501.06842 — ICLR 2025**

- Authors: Tianjin Huang, Ziquan Zhu, Gaojie Jin, Lu Liu, Zhangyang Wang, Shiwei Liu
- Proposes SPAM, which adds periodic momentum reset and spike-aware clipping to Adam to suppress gradient spikes (which can be 1000× larger than typical gradients in LLM training).
- Key values: experiments span LLM pretraining at 60M–1B, 4-bit pretraining, RL; SPAM consistently surpasses AdamW variants. A memory-efficient sparse-momentum variant outperforms GaLore and Adam-mini under memory constraints.
- Relevance (Gap 1 + Gap 3): directly relevant to the gradient clipping question — SPAM provides a complementary mechanism to fixed-threshold clipping. For a project initialising 5000 new DSU embeddings, gradient spikes during early CPT are a concrete risk. This paper provides the most recent ICLR 2025 evidence that spike-aware mechanisms help at exactly 1B scale.

**[C] Wen et al. (2025), "Fantastic Pretraining Optimizers and Where to Find Them," arXiv:2509.02046**

- Authors: Kaiyue Wen et al.
- Systematic evaluation of 10 deep learning optimizers across 4 model scales (0.1B–1.2B) and data-to-model ratios (1–8× Chinchilla optimum). Addresses two methodological issues in prior work: unequal hyperparameter tuning and limited evaluation setups.
- Key values: actual speedup of proposed optimizers over well-tuned AdamW baselines is lower than claimed and decreases with model size to only 1.1× for 1.2B models; matrix-based preconditioners (Muon, SOAP) show speedup decreasing from 1.4× (0.1B) to 1.1× (1.2B).
- Relevance (Gap 1): independently confirms the existing document's implicit assumption that AdamW with tuned β₂/ε remains competitive at 1B scale. Note: published after the main literature review cutoff but provides direct 1B-scale empirical confirmation — it is a companion to Semenov et al. (arXiv:2509.01440, already cited §1 summary table) from the same period.

---

### Gap 2: Weight decay ablations for domain adaptation CPT

**[D] Zosa et al. (2025), "Continued Pretraining: A Practical Playbook for Language-Specific LLM Adaptation," AMD ROCm Blog (2025-06-18)**

- Authors: Elaine Zosa, Jouni Luoma, Kai Hakala, Antti Virtanen, Mika Koistinen, Jonathan Burdge
- Multilingual CPT playbook using Llama 3-based 8B and 70B models on AMD Instinct MI250X (LUMI supercomputer). Runs ablation over weight decay directly: tests 0.01 vs. the Megatron-LM default of 0.1.
- Key finding: "a lower value of 0.01 did not measurably affect downstream evaluation performance; we stayed with default 0.1." Other findings: cosine schedule preferred over trapezoidal; single epoch of domain data better than repetition; LR=3e-4 effective at 8B scale.
- Relevance (Gap 2): this is the only documented controlled ablation of weight_decay=0.01 vs. 0.1 in a CPT context. The result directly supports the main document's recommendation (§1.3) that weight_decay=0.1 is the correct default, and confirms that 0.01 is not measurably better or worse — it simply has no advantage over the established Llama recipe.
- Note: this is a practitioner blog post, not a peer-reviewed paper. It should not be cited as equivalent to conference papers but provides the only empirical data point for this specific ablation.

---

### Gap 3: Gradient clipping for LLM training stability

**[E] (Authors TBC), "AdaGC: Improving Training Stability for Large Language Model Pretraining," arXiv:2502.11034 (2025)**

- Proposes Adaptive Gradient Clipping based on Local Gradient Norm (AdaGC): automatically adjusts local clipping thresholds per parameter using exponential moving average of gradient norms. An OpenReview submission (ZQcDUhEOg9) confirms peer review status.
- Key values: on Llama-2 7B, AdaGC completely eliminates loss spikes and reduces WikiText perplexity by 3.5% (+0.14pp LAMBADA accuracy) compared to global gradient clipping (clip=1.0). On Llama-2 13B: 0.65% lower training loss, 1.47% lower validation perplexity. On CLIP ViT-Base: 25% faster convergence than StableAdamW with full spike elimination. AdaGC is optimizer-agnostic and introduces negligible memory overhead.
- Relevance (Gap 3): provides the most detailed 2025 study of what global gradient clipping with clip_grad_norm=1.0 achieves vs. adaptive alternatives. The key point for this project: standard clip=1.0 does prevent spikes but is sub-optimal relative to adaptive methods; the main document's recommendation to enable clip_grad_norm=1.0 is the correct minimum threshold, and adaptive methods are an available upgrade path.

**[F] Kumar & Owen et al. (2025), "ZClip: Adaptive Spike Mitigation for LLM Pre-Training," arXiv:2504.02507**

- Published April 2025 (code at github.com/bluorion-com/ZClip).
- Proposes ZClip: uses z-score-based anomaly detection on running EMA statistics of gradient norms to detect and mitigate spikes adaptively without storing historical data.
- Key values: tested specifically on 1B-parameter LLaMA models; ZClip consistently outperforms both fixed-threshold clipping (clip=1.0) and percentile-based approaches; expands the feasible learning rate space, enabling faster convergence.
- Relevance (Gap 3): ZClip experiments are specifically on 1B LLaMA models, making this the most directly scale-matched study on gradient clipping in the supplement. Confirms that fixed clip=1.0 is a sound baseline while adaptive clipping provides additional stability benefits at exactly our target model scale.

---

### Gap 4: BF16 vs FP16 precision for LLM training

**[G] Kumar et al. (2024), "Scaling Laws for Precision," arXiv:2411.04330 — ICLR 2025 (Oral)**

- Authors: Tanishq Kumar, Zachary Ankner, Benjamin Spector, Blake Bordelon, Niklas Muennighoff, Mansheej Paul, Cengiz Pehlevan, Christopher Ré, Aditi Raghunathan
- Develops "precision-aware" scaling laws for both training and inference over 465+ pretraining runs, validated on models up to 1.7B trained on up to 26B tokens. Received ICLR 2025 Oral distinction.
- Key values: training in lower precision reduces effective parameter count; training larger models in lower precision can be compute-optimal; BF16 has become de facto standard; FP8 introduces instabilities not fully resolved by current methods.
- Relevance (Gap 4): the strongest current theoretical and empirical treatment of precision effects on LLM training. Directly supports the main document's recommendation of dtype=bf16 (§7) by establishing that BF16 occupies the efficiency-stability Pareto frontier for 1B-scale training. Also establishes the scaling-law framework that predicts when moving to FP8 would become worthwhile (not at our 1B/4.9B-token scale).

**[H] Lee et al. (2024), "To FP8 and Back Again: Quantifying Reduced Precision Effects on LLM Training Stability," arXiv:2405.18710**

- Authors: Joonhyung Lee, Jeongin Bae, Byeongwook Kim, Se Jung Kwon, Dongsoo Lee
- Systematically studies training stability across BF16, FP16, and FP8 formats for LLM pretraining. Tests robustness across random seeds, learning rates, and datasets. An OpenReview submission (pNgyXuGcx4) confirms peer review.
- Key finding: "BF16 has become the de facto standard for LLM training"; current FP8 methods are "not robust enough to allow their use as economical replacements" for BF16. FP16 is less stable than BF16 due to limited dynamic range causing gradient underflow.
- Relevance (Gap 4): most targeted study of the BF16 vs. FP16 question for LLM training. Provides explicit evidence that BF16 is preferred over FP16, and that the ε=1e-5 recommendation in §1.2 is specifically motivated by BF16's reduced mantissa precision (7 bits vs. 10 bits in FP16).

---

### Gap 5: Discrete speech token vocabulary size ablations

**[I] Chang et al. (2024), "The Interspeech 2024 Challenge on Speech Processing Using Discrete Units," arXiv:2406.07725 — Interspeech 2024**

- Authors: Xuankai Chang, Jiatong Shi, Jinchuan Tian, Yuning Wu, Yuxun Tang, Yihan Wu, Shinji Watanabe, Yossi Adi, Xie Chen, Qin Jin. Proceedings of Interspeech 2024, pp. 2559–2563, doi: 10.21437/Interspeech.2024-1878.
- Presents the Interspeech 2024 shared task on discrete-unit ASR, TTS, and singing voice synthesis. 40 submissions across all tracks. ASR baseline uses k=2,000 k-means clusters on features from layer 21 of WavLM-Large (1024-dim), with deduplication then BPE (vocab size 6,500). Trained for 100 epochs on a single Nvidia V100-32GB GPU, approximately 18 hours total training time.
- Key values: **k=2,000** for ASR baseline (not 500); **WavLM-Large layer 21**; deduplication applied before BPE; BPE vocab = 6,500; best ASR submissions used semantic tokens from SSL models.
- Relevance (Gap 5 + Gap 6 + Gap 7): this is the most directly relevant 2024 conference benchmark for discrete-unit ASR. Its use of k=2,000 as the competition baseline (rather than the traditional k=500 used by TWIST/SpiRit-LM) provides new 2024 evidence that k=2,000 is the contemporary default for ASR-focused discrete unit work. Note: uses WavLM-Large rather than HuBERT, but the k-means clustering methodology is identical. Should be added to the §6.2 n_dsus table.

**[J] Dekel & Fernandez (2024), "Exploring the Benefits of Tokenization of Discrete Acoustic Units," Interspeech 2024, pp. 2780–2784, doi: 10.21437/Interspeech.2024-533**

- Authors: Avihu Dekel, Raul Fernandez (Microsoft Research)
- Demonstrates that BPE tokenization of discrete acoustic units (DAUs) yields significant improvements in performance, training speed, and inference speed across grapheme-to-phoneme conversion, grapheme-to-DAU conversion, and unsupervised speech generation (DAU language modeling).
- Relevance (Gap 5): provides 2024 Interspeech evidence that the effective vocabulary size (after BPE merges) matters independently from the k-means codebook size, supporting the ablation recommendation in §6.2. The combination of k=2,000 + BPE vocab 6,500 from the Interspeech challenge [I] vs. k=500 + no BPE from TWIST/SpiRit-LM represents the key design fork to explore.

---

### Gap 7: Speech token deduplication effects

**[K] Futami et al. (2025), "Scheduled Interleaved Speech-Text Training for Speech-to-Speech Translation with LLMs," arXiv:2506.10299 — Interspeech 2025**

- Authors: Futami, Tsunoo, Kashiwagi, Ito, Shahmohammadi, Arora, Watanabe (Waseda University / Johns Hopkins University)
- Fine-tunes LLaMA3.2-1B for speech-to-speech translation on the CVSS corpus (21 languages + English). Uses k=2,048 k-means clusters from w2v-BERT layer 20; explicitly reports training **without applying deduplication**. Vocabulary: approximately 12K original LLaMA text BPEs + 2,048 speech units. Optimizer: Adam, LR=5e-5. Uses scheduled interleaving: initial text ratio p=0.9, decayed by 0.1 every 300 steps to 0 (pure speech units).
- Key finding on deduplication: training without deduplication is a deliberate choice, distinct from the GSLM-lineage default. Results show competitive S2ST performance across languages.
- Relevance (Gap 7 + Gap 9): this is the only 2025 Interspeech paper fine-tuning Llama 3.2 1B for speech tasks with discrete units that explicitly states no deduplication was applied. Combined with k=2,048, it expands the parameter space explored in the literature: {dedup=False, k=2,048} is a viable operating point for at least generation tasks. Also relevant to Gap 9 (vocabulary extension): the approach of simply appending k new speech unit tokens to the LLaMA vocabulary without elaborate initialization is the default practice here and in most speech LLM papers.

---

### Gap 8: Packing vs. non-packing for SFT

**[L] Wang et al. (2024), "Packing Analysis: Packing Is More Appropriate for Large Models or Datasets in Supervised Fine-tuning," arXiv:2410.08081**

- Authors: Shuhe Wang, Guoyin Wang, Yizhong Wang, Jiwei Li, Eduard Hovy, Chen Guo
- First comprehensive study comparing packing vs. padding for SFT across datasets (69K–1.2M samples) and models (8B–70B). Covers benchmarks in knowledge, reasoning, and coding; includes time efficiency analysis.
- Key finding: packing is more beneficial for larger models (8B+) and larger datasets (>500K samples). For smaller setups, gains from packing may not materialize and cross-sequence contamination can occur.
- Relevance (Gap 8): directly addresses the `packed=False` setting in the codebase. Our setup uses a small model (1B) and a relatively small SFT dataset (MLS English ASR), placing us squarely in the regime where this paper finds packing to be less beneficial. The current `packed=False` default is therefore well-calibrated for our scale.

**[M] Dong et al. (2024), "Threshold Filtering Packing for Supervised Fine-Tuning: Training Related Samples within Packs," arXiv:2408.09327 — NAACL 2025**

- Authors: Jiancheng Dong, Lei Jiang, Wei Jin, Lu Cheng. Published in Proceedings of NAACL 2025 (Long Papers), pp. 4422–4435, Albuquerque, New Mexico.
- Proposes TFP (Threshold Filtering Packing): selects samples with semantically related context for packing to avoid cross-contamination, while maintaining sufficient within-pack diversity.
- Key values: improvements of up to 7% on GSM8K, 4% on HumanEval, 15% on bias benchmarks vs. standard naive packing.
- Relevance (Gap 8): establishes that naive packing (random concatenation) introduces cross-contamination artifacts. The `packed=False` default in the codebase correctly avoids this problem. If packing is added later for efficiency, TFP-style related-sample grouping would be the correct implementation to consider.

---

### Gap 9: Vocabulary extension and new token embedding initialization

**[N] Mundra et al. (2024), "An Empirical Comparison of Vocabulary Expansion and Initialization Approaches for Language Models," arXiv:2407.05841 — CoNLL 2024 (co-located with EMNLP 2024)**

- Authors: Nandini Mundra, Aditya Nanda Kishore Khandavally, Raj Dabre, Ratish Puduppully, Anoop Kunchukuttan, Mitesh M. Khapra. Published in Proceedings of the 28th Conference on Computational Natural Language Learning (CoNLL 2024), aclanthology.org/2024.conll-1.8.
- Empirical study of initialization strategies for new vocabulary tokens added to pretrained LMs. Tests random, mean-of-existing, compositional, and cross-lingual embedding approaches. Establishes theoretically that initializing within the convex hull of existing embeddings is a well-motivated baseline.
- Key finding: proposed Constrained Word2Vec (CW2V) approach (no cross-lingual embeddings required) is competitive with stronger baselines; mean initialization of existing embeddings is a strong practical default.
- Relevance (Gap 9): directly relevant to how the N_DSU new embedding rows should be initialized in the Llama 3.2 1B embedding matrix (and lm_head). The peer-reviewed 2024 result supporting mean initialization as the theoretically principled default complements practical guidance from the TWIST/SpiRit-LM papers (which initialize DSU embeddings randomly and rely on CPT to learn representations). For our setting, both approaches are defensible; this paper is the most rigorous recent citation for the mean-init option.

---

### Gap 4 (additional) / Warmup mechanisms: Two NeurIPS 2024 papers on LR warmup theory

**[O] Kalra & Barkeshli (2024), "Why Warmup the Learning Rate? Underlying Mechanisms and Improvements," arXiv:2406.09405 — NeurIPS 2024**

- Authors: Dayal Singh Kalra, Maissam Barkeshli. Proceedings of NeurIPS 2024, neurips.cc/virtual/2024/poster/95431.
- Studies mechanisms of LR warmup across FCNs, ResNets, and Transformers with both SGD and Adam. Identifies the primary mechanism: warmup forces the network to flatter regions of the loss landscape (lower Hessian sharpness), enabling larger target learning rates.
- Key finding: warmup reduces the top Hessian eigenvalue, enabling the model to tolerate higher LR; in some settings, warmup can be eliminated entirely by choosing the initial LR using the "loss catapult mechanism."
- Relevance (warmup, §4): provides NeurIPS 2024 theoretical grounding for the warmup recommendation. Particularly relevant to the PI review's H4 hypothesis: the mechanism (sharpness reduction) is most critical when new parameters are added, strengthening the case for at least 3% warmup when DSU embeddings are being learned from random initialization.

**[P] Kosson, Messmer & Jaggi (2024), "Analyzing & Reducing the Need for Learning Rate Warmup in GPT Training," arXiv:2410.23922 — NeurIPS 2024**

- Authors: Atli Kosson, Bettina Messmer, Martin Jaggi (EPFL). Proceedings of NeurIPS 2024, neurips.cc/virtual/2024/poster/94618.
- Focused study on GPT training with AdamW/Lion. Identifies three root causes for why warmup is needed with Adam: (1) Adam's momentum handling causes artificially large initial updates; (2) early updates are large relative to initial weight magnitudes; (3) early gradients are highly correlated, limiting effective batch size. Shows that modifications addressing these issues can reduce or eliminate the need for warmup.
- Relevance (warmup, §4): provides the most mechanistic 2024 NeurIPS analysis of why AdamW in particular requires warmup. Point (1) — large initial Adam updates — is especially relevant to our setting where DSU embeddings start with large gradients (untrained parameters) while the rest of the model is already well-trained. This asymmetry strengthens the case for at least 3% warmup when adding new modality tokens.

---

### Supplementary citation table for existing sections

The following papers provide supporting evidence for claims already made in the main document and should be considered for inclusion in the respective summary tables:

| New Ref | Venue | Year | Gap Addressed | Section to Augment |
|---------|-------|------|---------------|-------------------|
| [A] Zhang et al., Adam-mini (arXiv:2406.16793) | **ICLR 2025** | 2024 | Optimizer alternatives at 1B scale | §1 summary table |
| [B] Huang et al., SPAM (arXiv:2501.06842) | **ICLR 2025** | 2025 | Gradient stability, spike-aware optimizer | §1.4 |
| [C] Wen et al., Fantastic Optimizers (arXiv:2509.02046) | arXiv 2025 | 2025 | Optimizer benchmark at 0.1B–1.2B | §1 summary table |
| [D] Zosa et al., CPT Playbook (AMD ROCm Blog) | Blog 2025 | 2025 | Weight decay ablation 0.01 vs 0.1 | §1.3 |
| [E] AdaGC (arXiv:2502.11034) | arXiv 2025 | 2025 | Adaptive gradient clipping | §1.4 |
| [F] ZClip (arXiv:2504.02507) | arXiv 2025 | 2025 | Adaptive gradient clipping, 1B LLaMA | §1.4 |
| [G] Kumar et al., Scaling Laws Precision (arXiv:2411.04330) | **ICLR 2025 Oral** | 2024 | BF16 vs FP16 vs FP8 | §7 |
| [H] Lee et al., To FP8 (arXiv:2405.18710) | arXiv 2024 | 2024 | BF16 vs FP16 training stability | §7 |
| [I] Chang et al., IS2024 Challenge (arXiv:2406.07725) | **Interspeech 2024** | 2024 | k=2000 DSU for ASR, deduplication | §6.2 summary table |
| [J] Dekel & Fernandez, DAU Tokenization | **Interspeech 2024** | 2024 | BPE on top of discrete speech tokens | §6.2 |
| [K] Futami et al., Scheduled Interleaved (arXiv:2506.10299) | **Interspeech 2025** | 2025 | k=2048, no deduplication, Llama 3.2 1B | §6.1, §6.2 |
| [L] Wang et al., Packing Analysis (arXiv:2410.08081) | arXiv 2024 | 2024 | Packing vs. padding for small-model SFT | §2.2 |
| [M] Dong et al., TFP (arXiv:2408.09327) | **NAACL 2025** | 2024 | Packing contamination in SFT | §2.2 |
| [N] Mundra et al., Vocab Expansion (arXiv:2407.05841) | **CoNLL/EMNLP 2024** | 2024 | New token embedding initialization | §6 |
| [O] Kalra & Barkeshli, Why Warmup (arXiv:2406.09405) | **NeurIPS 2024** | 2024 | LR warmup mechanisms (Adam) | §4 |
| [P] Kosson et al., Reducing Warmup (arXiv:2410.23922) | **NeurIPS 2024** | 2024 | LR warmup root causes with AdamW | §4 |

---

### Gaps with insufficient new evidence (no new citations added)

The following gaps were searched but did not yield papers that are not already cited and that provide materially new evidence beyond what is in the main document:

- **Multi-epoch SFT for speech tasks (Gap 10):** The Interspeech 2024 challenge [I] trains for 100 epochs, but this is a small ASR encoder-decoder system, not an LLM fine-tuning setup. No new ICASSP/Interspeech 2024–2025 paper was found that directly studies epoch count for LLM-based ASR SFT in a way not already covered by USDM (5 epochs, NeurIPS 2024) and SALMONN (6 epochs, ICLR 2024) in the main document.

- **HuBERT discrete units for ASR with language models (Gap 6):** The main document already cites the most important papers (Yang et al. ICASSP 2024, Wang et al. arXiv 2024, Mousavi et al. Interspeech 2024). The Interspeech 2024 challenge [I] (added above) is the main new entry; it uses WavLM-Large rather than HuBERT but the discrete unit extraction methodology is equivalent. No additional HuBERT-specific ICASSP/Interspeech 2024–2025 ablation paper was found that is not already covered.
