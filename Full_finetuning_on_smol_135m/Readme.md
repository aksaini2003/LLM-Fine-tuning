# 🚀 Full Fine-Tuning of SmolLM2-135M on the Alpaca Dataset

This project demonstrates **full parameter fine-tuning** of **HuggingFaceTB/SmolLM2-135M** using the first **20,000 samples** from the **`yahma/alpaca-cleaned`** dataset.

The goal of this project was to build an end-to-end supervised fine-tuning pipeline, compare the base and fine-tuned models, and understand the capabilities and limitations of small language models.

---

## 📌 Model

* **Base Model:** `HuggingFaceTB/SmolLM2-135M`
* **Fine-Tuning:** Full Parameter Fine-Tuning (No LoRA/PEFT)

---

## 📂 Dataset

* **Dataset:** `yahma/alpaca-cleaned`
* **Training Samples:** First 20,000 instruction-response pairs

---

## 🛠️ What This Notebook Covers

* Loading and preprocessing the Alpaca dataset
* Formatting instruction-response prompts
* Tokenization
* Loading the SmolLM2-135M model
* Performing inference with the base model
* Full parameter fine-tuning using TRL's `SFTTrainer`
* Saving model checkpoints
* Loading the fine-tuned checkpoint
* Comparing the base and fine-tuned models on unseen prompts

---

## 📈 Results

Compared to the base model, the fine-tuned model:

* Improved instruction-following
* Generated more coherent responses
* Reduced unrelated generations such as random code completions

Since SmolLM2-135M is a relatively small language model, it still struggles with reasoning-intensive tasks, which highlights the trade-offs between model size and capability.

---

## 🧰 Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* TRL (`SFTTrainer`)
* Accelerate

---

## 🚀 Getting Started

Clone the repository:

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Open and run:

```text
Full_finetuning_on_smol_llm.ipynb
```

---

## 📚 Learning Outcomes

This project helped me gain practical experience with:

* Supervised Fine-Tuning (SFT)
* Full parameter fine-tuning of LLMs
* Instruction prompt formatting
* Model evaluation
* Comparing base vs. fine-tuned models
* Working with Hugging Face and TRL

---

## 🔮 Future Improvements

* Fine-tune larger SmolLM variants
* Experiment with LoRA/QLoRA
* Evaluate using standard benchmarks
* Train on larger instruction datasets
* Compare different fine-tuning strategies

---

## 🤝 Contributing

Suggestions and improvements are always welcome. Feel free to open an issue or submit a pull request.

---

## ⭐ If you found this project useful

Consider giving the repository a ⭐ to support the project.
