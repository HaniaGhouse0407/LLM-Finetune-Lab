<div align="center">

# 🧬 LLM Fine-Tuning Lab

**LoRA · QLoRA · PEFT Fine-Tuning Pipeline with Live Training Dashboard**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Author](https://img.shields.io/badge/Author-Hania_Ghouse-F59E0B?style=flat-square)](https://github.com/HaniaGhouse0407)

</div>

---

## 🧠 Overview

A complete LLM fine-tuning workbench supporting LoRA and QLoRA (4-bit) on Mistral-7B, Llama-3, Gemma, and Phi-2. Includes a live training dashboard, inference playground, and benchmark evaluation.

---

## ✨ Features

- ✅ QLoRA 4-bit quantization (bitsandbytes) — fine-tune 7B models on 8GB GPU
- ✅ Configurable LoRA rank, alpha, dropout, and target modules
- ✅ Live training loss chart updating in real time
- ✅ Inference playground with temperature and max-token controls
- ✅ MMLU, HellaSwag, TruthfulQA benchmark comparison (base vs fine-tuned)
- ✅ HuggingFace Hub push — one click to share your fine-tuned model

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/HaniaGhouse0407/LLM-Finetune-Lab.git
cd LLM-Finetune-Lab

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables (if needed)
cp .env.example .env
# Edit .env with your API keys

# 4. Run
streamlit run app.py
```

---

## 🛠️ Tech Stack

![transformers](https://img.shields.io/badge/transformers-FFD21E?style=flat-square)  ![peft](https://img.shields.io/badge/peft-555555?style=flat-square)  ![bitsandbytes](https://img.shields.io/badge/bitsandbytes-555555?style=flat-square)  ![accelerate](https://img.shields.io/badge/accelerate-555555?style=flat-square)  ![trl](https://img.shields.io/badge/trl-555555?style=flat-square)  ![datasets](https://img.shields.io/badge/datasets-555555?style=flat-square)  ![streamlit](https://img.shields.io/badge/streamlit-FF4B4B?style=flat-square)  ![torch](https://img.shields.io/badge/torch-EE4C2C?style=flat-square)

---

## 📁 Project Structure

```
LLM-Finetune-Lab/
├── app.py              # Main Streamlit/Gradio application
├── requirements.txt    # Dependencies
├── .env.example        # Environment variable template
└── README.md
```

---

## 🎯 Target Roles

> LLM Engineer · Research Engineer · ML Engineer

---

<div align="center">

Made by [Hania Ghouse](https://github.com/HaniaGhouse0407) · 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/hania-ghouse/)
[![Google Scholar](https://img.shields.io/badge/Scholar-Research-4285F4?style=flat-square&logo=google-scholar)](https://scholar.google.com/citations?user=iVWuM4wAAAAJ)

</div>
