# 🏥 Medical Text Summarizer

<p align="center">
  <img src="https://img.shields.io/badge/Model-BART--large--CNN-blueviolet?style=for-the-badge&logo=huggingface" />
  <img src="https://img.shields.io/badge/Fine--tuned%20on-PubMed-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/UI-Gradio-FF7C00?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Deployed-HuggingFace%20Spaces-FFD21E?style=for-the-badge&logo=huggingface" />
</p>

<p align="center">
  A production-grade NLP system that summarizes medical articles, clinical notes, and research abstracts using a fine-tuned BART model trained on 10,000 PubMed biomedical papers.
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/AbhijithSK/medical-summarizer"><strong>🚀 Live Demo →</strong></a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="https://huggingface.co/AbhijithSK/medical-summarizer"><strong>🤗 Model on HF Hub →</strong></a>
</p>

---

## 📌 What It Does

Paste any medical text — a clinical note, a PubMed abstract, a discharge summary — and the model generates a concise, accurate summary in seconds. Built for clinicians, researchers, and medical students who need to process large volumes of biomedical literature quickly.

**Input:** Long-form medical text (clinical notes, research abstracts, patient records)  
**Output:** Concise 2–4 sentence summary capturing the key findings

---

## 🧠 Model Architecture

```
Input Text (512 tokens max)
        ↓
BART Encoder (Bidirectional Transformer)
        ↓
Cross-Attention
        ↓
BART Decoder (Autoregressive)
        ↓
Generated Summary (128 tokens max)
```

| Component | Detail |
|-----------|--------|
| Base Model | `facebook/bart-large-cnn` |
| Fine-tuning Data | PubMed Summarization (`ccdv/pubmed-summarization`) |
| Training Samples | 10,000 biomedical articles |
| Parameters | 406,290,432 |
| Max Input Length | 512 tokens |
| Max Summary Length | 128 tokens |
| Decoding Strategy | Beam Search (num_beams=4) |

---

## 📊 Evaluation Results

Evaluated on 200 held-out PubMed validation samples:

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.3614 |
| ROUGE-2 | 0.1379 |
| ROUGE-L | 0.2318 |
| ROUGE-Lsum | 0.3060 |
| Eval Loss | 2.231 |

> ROUGE-1 of 0.36 on biomedical text is a strong baseline for a single-epoch fine-tune on consumer hardware.

---

## 🗂️ Project Structure

```
Medical-Summarizer/
├── data/
│   └── preprocess.py        # PubMed dataset loading & preprocessing
├── model/
│   ├── train.py             # Fine-tuning pipeline with HuggingFace Trainer
│   └── eval_metrics.py      # ROUGE evaluation utilities
├── app/
│   ├── app.py               # Gradio web interface
│   └── requirements.txt     # Space dependencies
├── requirements.txt         # Full project dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/Abhijithsk/Medical-Summarizer.git
cd Medical-Summarizer
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the App Locally

```bash
python app/app.py
```

Open `http://127.0.0.1:7860` in your browser.

### 3. Fine-tune the Model Yourself

```bash
python model/train.py
```

Trains on 10k PubMed samples for 1 epoch (~10 hours on CPU, ~45 min on GPU).

---

## 💡 Example

**Input (Clinical Note):**
```
A 78-year-old female returns for a recheck regarding ongoing spinal stenosis, 
syncope, and poorly controlled essential hypertension. BP is elevated at 155/92 mmHg. 
Will decrease Atenolol to 25mg daily and add Amlodipine 5mg daily...
```

**Output (Generated Summary):**
```
A 78-year-old female with spinal stenosis, syncope, and poorly controlled 
hypertension presents with elevated BP at 155/92 mmHg. Treatment plan includes 
Atenolol dose reduction and addition of Amlodipine.
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | BART-large-CNN (Facebook AI) |
| Fine-tuning | HuggingFace Transformers + Seq2SeqTrainer |
| Dataset | PubMed Summarization (119k papers) |
| Evaluation | ROUGE via HuggingFace Evaluate |
| UI | Gradio 6.x |
| Deployment | HuggingFace Spaces |
| Model Registry | HuggingFace Hub |

---

## 📈 Training Pipeline

```
PubMed Dataset (119,924 papers)
        ↓
Subset selection (10,000 train / 200 val)
        ↓
Tokenization (max_length=512, padding, truncation)
        ↓
Seq2SeqTrainer (lr=2e-5, batch=2, epochs=1)
        ↓
ROUGE evaluation on validation set
        ↓
Model saved → pushed to HuggingFace Hub
```

---

## 🔗 Links

- 🤗 **Live Demo:** https://huggingface.co/spaces/AbhijithSK/medical-summarizer
- 📦 **Model Hub:** https://huggingface.co/AbhijithSK/medical-summarizer
- 📊 **Dataset:** https://huggingface.co/datasets/ccdv/pubmed-summarization

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<p align="center">Built with ❤️ using HuggingFace Transformers, PyTorch, and Gradio</p>