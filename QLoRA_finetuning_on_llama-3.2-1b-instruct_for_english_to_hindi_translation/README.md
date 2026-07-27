# Fine-Tuning LLaMA 3.2-1B Instruct for English → Hindi Translation

Fine-tune Meta's **LLaMA 3.2-1B Instruct** model for English-to-Hindi translation using **QLoRA** (4-bit quantization + LoRA adapters) on the `ai4bharat/samanantar` dataset. The fine-tuned model is evaluated using the **BLEU score**.

---

## Overview

| Item | Detail |
|---|---|
| Base Model | `meta-llama/Llama-3.2-1B-Instruct` |
| Dataset | `ai4bharat/samanantar` (Hindi subset, ~10.1M rows) |
| Training Samples | 1,000 (subset due to Colab limits) |
| Technique | QLoRA — 4-bit NF4 quantization + LoRA |
| Total Parameters | 749,275,136 |
| Trainable Parameters (LoRA) | 11,272,192 (~0.90%) |
| Evaluation Metric | BLEU Score (sacrebleu) |
| **BLEU Score** | **8.17** (evaluated on 10 held-out samples) |
| Platform | Google Colab (GPU) |

---

## Getting Started

### 1. Clone the Repository

```bash
git clone --no-checkout --depth 1 https://github.com/aksaini2003/LLM-Fine-tuning.git
cd LLM-Fine-tuning
git sparse-checkout init --cone
git sparse-checkout set QLoRA_finetuning_on_llama-3.2-1b-instruct_for_english_to_hindi_translation
git checkout main
```

### 2. Install Dependencies

```bash
pip install transformers accelerate peft trl datasets sentencepiece bitsandbytes sacrebleu
```

### 3. Hugging Face Authentication

You need access to the gated `meta-llama/Llama-3.2-1B-Instruct` model. Log in via:

```bash
huggingface-cli login
```

Or set your token directly in the notebook:

```python
from huggingface_hub import login
login(token="your_hf_token_here")
```

### 4. Open the Notebook

Upload and run the notebook in **Google Colab** (GPU runtime recommended):

```
finetuning_llama_3_2_1b_instruct_for_translation_from_english_to_hindi.ipynb
```

---

## Pipeline

### 1. Dataset Preparation
- Loads the `ai4bharat/samanantar` Hindi dataset (~10.1M parallel sentence pairs).
- Subsets to **1,000 training samples** due to Colab resource constraints.
- Formats each example using an instruction-response template:

```
### Instruction:
<English sentence>

### Response:
<Hindi sentence>
```

### 2. Model Loading with 4-bit Quantization
- Loads `LLaMA 3.2-1B Instruct` using **4-bit NF4** precision via `BitsAndBytesConfig`.
- Uses `bfloat16` compute dtype and double quantization for memory efficiency.
- Model architecture: 16 decoder layers, 2048 hidden size, vocabulary size 128,256.

### 3. LoRA Configuration
- Applies LoRA adapters to all major projection layers:
  `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- LoRA rank `r=16`, alpha `32`, dropout `0.05`.
- Only **~0.90%** of parameters are trainable (11.27M out of 1.25B).

### 4. Training
| Hyperparameter | Value |
|---|---|
| Epochs | 3 |
| Batch size | 4 |
| Gradient accumulation steps | 2 (effective batch = 8) |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Warmup ratio | 0.03 |
| Optimizer | `paged_adamw_8bit` |
| Precision | FP16 |

Checkpoints are saved per epoch to Google Drive.

### 5. Inference
Loads the fine-tuned LoRA checkpoint (`checkpoint-375`) and runs greedy decoding (`do_sample=False`, `repetition_penalty=1.1`, `max_new_tokens=128`).

**Sample outputs (fine-tuned model):**

| English | Hindi (Generated) |
|---|---|
| I love machine learning | मैं मशीन लर्निंग को बहुत पसंद करता हूँ। |
| The weather is nice today. | सunny weather today. *(partial failure)* |
| I love learning artificial intelligence. | मैं AI को सीखना पसंद करता हूं। |

### 6. Evaluation
- Evaluated on **10 held-out samples** (rows 1000–1009) using `corpus_bleu` from `sacrebleu`.
- **BLEU Score: 8.17**

> The low BLEU score is expected — only 1,000 training samples were used due to Colab GPU/runtime limits. A larger training set would significantly improve performance.

---

## Project Structure

```
.
├── finetuning_llama_3_2_1b_instruct_for_translation_from_english_to_hindi.ipynb
└── README.md
```

---

## Notes

- A Hugging Face token with access to the **gated LLaMA 3.2 repository** is required.
- The prompt format (`### Instruction` / `### Response`) must be consistent between training and inference — do not change it after training.
- Model checkpoints are saved to and loaded from **Google Drive** (`/content/drive/MyDrive/llama_lora/`).