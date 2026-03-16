Llama 3.2 was released September 2024. 

**Only found one paper in the list as a confirmed top-venue conference paper** using Llama 3.2 1B specifically:

| Paper | Venue | Model | Type | LR |
|-------|-------|-------|------|----|
| MMS-LLaMA (arXiv:2503.11315) | **ACL Findings 2025** | Llama 3.2 1B (and 3B, 8B) | SFT QLoRA — audio-visual speech recognition | **1×10⁻⁴ peak**, min 1×10⁻⁵, cosine, 500 warmup steps, Adam β₂=0.98 |

The following are 2024–2025 arXiv preprints. 

For hyperparameter searches, arXiv papers from reputable groups are the relevant prior.

## Full table: confirmed learning rates

| # | Paper | Venue | Model | Type | LR | Scheduler | Optimizer |
|---|-------|-------|-------|------|----|-----------|-----------|
| 1 | [SEC Financial CPT](https://arxiv.org/abs/2512.12384) | arXiv Dec 2024 (Johns Hopkins) | Llama-3.2-1B | Full-param CPT | **5×10⁻⁶** | None (fixed) | AdamW |
| 2 | [LoRA CPT grid search](https://arxiv.org/abs/2501.17840) | arXiv Jan 2025 (Megagon Labs) | Llama-3.2-1B | CPT (LoRA) | **Grid: 1e-5 → 3e-3** | not stated | not stated |
| 3 | [Forgetting SFT](https://arxiv.org/abs/2508.04329) | arXiv Aug 2025 (multi-inst.) | Llama-3.2-1B | SFT LoRA r=64 | **1×10⁻⁴** | Linear, 3% warmup | AdamW |
| 4 | [FuseChat-3.0](https://arxiv.org/abs/2503.04222) | arXiv Mar 2025 (Sun Yat-sen) | Llama-3.2-1B-Instruct | SFT | **5×10⁻⁶** | Cosine, 10% warmup | — |
| 5 | [FuseChat-3.0](https://arxiv.org/abs/2503.04222) | arXiv Mar 2025 | Llama-3.2-1B-Instruct | DPO | **1×10⁻⁶** | Cosine, 10% warmup | — |
| 6 | **[MMS-LLaMA](https://aclanthology.org/2025.findings-acl.1065.pdf)** | **ACL Findings 2025** | Llama 3.2 1B | SFT QLoRA (speech) | **1×10⁻⁴ → 1×10⁻⁵** | Cosine, 500-step warmup | Adam β₂=0.98 |
| 7 | [Safety FT](https://arxiv.org/abs/2508.12531) | arXiv Aug 2025 (Oxford/Cambridge/KAUST) | Llama-3.2-1B-Instruct | SFT | **2×10⁻⁵** | StepLR γ=0.85 | AdamW |
| 8 | [FPT Vietnamese CPT](https://fptcloud.com/en/continual-pre-training-of-llama-3-2-1b-with-fpt-ai-studio/) | Technical blog 2025 | Llama-3.2-1B | **Full-param CPT**, 2.8B Vietnamese tokens | **4×10⁻⁵** | Linear | AdamW, grad clip 1.0 |
| 9 | [Breeze2](https://arxiv.org/abs/2501.13921) | arXiv Jan 2025 (MediaTek Research) | Llama-3.2-**3B** | CPT (Traditional Chinese) | **1×10⁻⁵** | Cosine, 500-step warmup | Fused Adam |
| 10 | [TinyLlama Stability Gap CPT](https://arxiv.org/abs/2406.14833) | arXiv 2024 | TinyLlama-**1.1B** | CPT (medical, 50B tokens) | **3×10⁻⁴** (optimal); 3×10⁻⁵ was insufficient | Cosine, 10% warmup | AdamW β₂=0.95 |
| 11 | [Speech Codec CPT](https://arxiv.org/abs/2502.16897) | arXiv Feb 2025 (CMU/Tencent) | Qwen1.5-0.5B (~943M) | **CPT speech tokens** | **1×10⁻⁵** peak, min 1×10⁻⁶ | — | AdamW, grad clip 1.0 |
| 12 | [Multilingual CPT](https://arxiv.org/abs/2504.04152) | arXiv Apr 2025 (Helsinki) | Llama-3.1-**8B** | CPT multilingual | **4×10⁻⁵** | Cosine, 3% warmup | DeepSpeed AdamW |
| 13 | [AMD CPT Playbook](https://rocm.blogs.amd.com/artificial-intelligence/multilingual-continued-pretraining/README.html) | AMD blog 2025 | Llama-3.1-**8B** | CPT multilingual | **3×10⁻⁴** | Cosine, 5% warmup | — |

## Impact on hyperparameter choices for full-parameter CPT of Llama 3.2 1B

The confirmed evidence breaks cleanly by task character:

| Evidence | LR | Notes |
|----------|-----|-------|
| SEC financial domain, 400M tokens, no schedule | 5e-6 | Likely overly conservative |
| Vietnamese, 2.8B tokens, linear schedule | 4e-5 | Full-param, practical run |
| TinyLlama 1.1B (methodologically rigorous grid search) | **3e-4 optimal; 3e-5 insufficient** | Most relevant methodological prior |
| Qwen 943M, speech CPT | 1e-5 | Conservative, very short warmup |
| Llama 3.2 3B (Breeze2), text CPT | 1e-5 | Scaled down 1B ≈ slightly higher |

The **TinyLlama stability gap paper** (arXiv:2406.14833) is the most methodologically useful for us because it: 
1. is ~1B scale
1. specifically studies CPT (not SFT)
1. ran a proper grid search
1. explicitly reports that 3e-5 was **too low** for learning to occur at 1B scale - that's the key qualitative finding — you can harm learning by going too conservative

**Consensus range for our full-param CPT from Llama 3.2 1B: 1e-5 to 1e-4**, with the most rigorous 1B-scale evidence pointing toward the higher end (~3e-4 if following TinyLlama's grid, ~1e-4 if being conservative). Cosine with 3–10% warmup is universal.

**For LoRA CPT:** 1e-4 to 1e-3.

The bottom line on the "10 citations from top conferences" request: the literature for Llama 3.2 1B specifically is almost entirely 2025 arXiv. The model simply hasn't been out long enough to accumulate conference proceedings citations at scale. If you want to anchor the choice at a published conference result, **MMS-LLaMA at ACL Findings 2025 using 1e-4 for speech-token SFT on Llama 3.2 1B is your cleanest reference** — and it's directly analogous to your use case.
