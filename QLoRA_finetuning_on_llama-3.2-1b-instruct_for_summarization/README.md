# Fine-Tuning LLaMA 3.2-1B Instruct for Text Summarization

Fine-tune Meta's **LLaMA 3.2-1B Instruct** model for news article summarization using **QLoRA** (4-bit quantization + LoRA adapters) on the `CNN/DailyMail` dataset. The fine-tuned model is evaluated using **ROUGE scores**.

---

## Overview

| Item | Detail |
|---|---|
| Base Model | `meta-llama/Llama-3.2-1B-Instruct` |
| Dataset | `abisee/cnn_dailymail` (v3.0.0) |
| Training Samples Used | 1,000 (subset due to Colab limits) |
| Technique | QLoRA — 4-bit NF4 quantization + LoRA |
| Total Parameters | 749,275,136 |
| Trainable Parameters (before freeze) | 262,735,872 |
| Trainable Parameters (after freeze + LoRA) | 0 base + LoRA adapters only |
| Global Steps | 375 |
| Training Loss | 1.1535 |
| Mean Token Accuracy | 52.09% |
| Train Runtime | ~6,622 seconds (~1.84 hours) |
| Platform | Google Colab (GPU) |

---

## Evaluation Results (ROUGE)

Evaluated on **10 held-out samples** from the CNN/DailyMail test set using stemming.

| Metric | Score |
|---|---|
| **ROUGE-1** | **0.3858** |
| **ROUGE-2** | **0.1637** |
| **ROUGE-L** | **0.2604** |
| **ROUGE-Lsum** | **0.3567** |

---

## Limitations

- Training performed on only 1,000 samples because of Google Colab GPU limitations.
- Evaluation performed on a small held-out subset (10 samples).
- The project is intended to demonstrate the complete QLoRA fine-tuning workflow rather than achieve state-of-the-art summarization performance.

---


## Future Work

- Train on a larger subset of CNN/DailyMail
- Evaluate on the complete test split
- Compare against the original Llama-3.2-1B-Instruct model
- Experiment with different LoRA ranks and learning rates

---

## Getting Started

### 1. Clone the Entire Repo

```bash
git clone https://github.com/aksaini2003/LLM-Fine-tuning.git
cd LLM-Fine-tuning
```

### 2. Clone Only This Subfolder

```bash
git clone --no-checkout --depth 1 https://github.com/aksaini2003/LLM-Fine-tuning.git
cd LLM-Fine-tuning
git sparse-checkout init --cone
git sparse-checkout set QLoRA_finetuning_on_llama-3.2-1b-instruct_for_summarization
git checkout main
```

### 3. Install Dependencies

```bash
pip install transformers accelerate peft trl datasets sentencepiece bitsandbytes evaluate rouge-score
```

### 4. Hugging Face Authentication

LLaMA 3.2 is a gated model — request access on Hugging Face first, then log in:

```bash
huggingface-cli login
```

Or inside the notebook:

```python
from huggingface_hub import login
login(token="your_hf_token_here")
```

### 5. Open the Notebook in Google Colab

Upload and run with a **GPU runtime**:

```
finetuning_llama_3_2_1b_instruct_for_summarization.ipynb
```

---

## Pipeline

### 1. Dataset Preparation
- Loads `abisee/cnn_dailymail` (v3.0.0) — fields: `article`, `highlights`, `id`.
- Full training set mapped with a formatting function, then subsetted to **1,000 samples**.
- Prompt template used for training:

```
###Instruction
<news article>

### Response
<highlight summary>
```

### 2. Model Loading with 4-bit Quantization
- Loads `LLaMA 3.2-1B Instruct` in **4-bit NF4** via `BitsAndBytesConfig`.
- `bfloat16` compute dtype + double quantization for memory efficiency.
- Total parameters: **749,275,136** | Initial trainable: **262,735,872** → frozen to **0** via `prepare_model_for_kbit_training`.

### 3. LoRA Configuration
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Rank `r=16`, alpha `32`, dropout `0.05`, bias `none`.

### 4. Training

| Hyperparameter | Value |
|---|---|
| Epochs | 3 |
| Per-device batch size | 4 |
| Gradient accumulation steps | 2 (effective batch = 8) |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Warmup ratio | 0.03 |
| Optimizer | `paged_adamw_8bit` |
| Precision | BF16 |
| Save strategy | Every 25 steps |
| Max checkpoints kept | 2 |
| Output dir (Drive) | `/content/drive/MyDrive/llama_using_LoRA_summarization` |

Training supports automatic resume via `resume_from_checkpoint=True` in case of Colab session drops.

### 5. Inference
- Loads base model + LoRA adapter from `checkpoint-375` on Google Drive.
- Tokenizer also loaded from the checkpoint path.
- Greedy decoding: `do_sample=False`, `repetition_penalty=1.1`, `max_new_tokens=128`.

**Sample outputs (fine-tuned model vs reference):**

| Topic | Reference | Generated |
|---|---|---|
| Palestine joins ICC | ICC gets jurisdiction; Israel & US opposed | Palestinians sign Rome Statute; ICC opens war crimes probe |
| Miracle dog Theia | Hit by car, buried, survived; needs home | Dog survived car strike and burial; vet hospital received donations |
| Bob Barker returns | Barker, 91, returned to host The Price Is Right | Bob Barker returns as host; stepped down in 2007 |

### 6. Evaluation
- 10 samples from `dataset['test']` (rows 0–9).
- ROUGE computed via `evaluate.load("rouge")` with `use_stemmer=True`.

---

## Project Structure

```
finetuning_llama_3_2_1b_instruct_for_summarization/
├── finetuning_llama_3_2_1b_instruct_for_summarization.ipynb
└── README.md
```

---

## Notes

- Only 1,000 training samples were used due to Colab GPU/runtime limits — more data would significantly boost ROUGE scores.
- The prompt format must stay consistent between training and inference.
- Checkpoints are saved to and loaded from **Google Drive**.
- A Hugging Face token with **gated access** to `meta-llama/Llama-3.2-1B-Instruct` is required.
