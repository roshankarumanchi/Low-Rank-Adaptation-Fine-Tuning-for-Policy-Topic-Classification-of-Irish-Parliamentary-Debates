# 🏛️ Low Rank Adaptation Fine-Tuning for Policy Topic Classification of Irish Parliamentary Debates

An NLP project that classifies Irish parliamentary speeches into 14 policy categories using **LoRA fine-tuning** on a large language model. Real debate transcripts were collected directly from the **Oireachtas Open Data API**, labelled, and used to fine-tune **Qwen2.5-1.5B-Instruct** with 4-bit quantisation — improving classification accuracy from **45% (zero-shot)** to **65% (LoRA fine-tuned)**.

> 📌 Built as part of the MSc Data Analytics programme — Module: Domain Applications, National College of Ireland  
> 👨‍🏫 Lecturer: Dr. Kislay Raj

---

## 📊 Results

| Method | Model | Accuracy |
|--------|-------|----------|
| Zero-Shot Baseline | facebook/bart-large-mnli | 45% |
| LoRA Fine-Tuned | Qwen2.5-1.5B-Instruct (4-bit) | **65%** |

> **+20% accuracy improvement** through parameter-efficient fine-tuning with LoRA — without training the full model.

---

## 🗺️ System Architecture

![System Architecture](system_architecture_diagram.png)

The pipeline flows through 6 modules:
1. **Oireachtas Open Data API** — HTTP GET requests to collect real Dáil Éireann debate transcripts
2. **Data Collection Module** — Paginated API scraping with XML parsing (BeautifulSoup)
3. **Raw Speech Data** — Stored as JSON/XML
4. **Preprocessing & Labelling Module** — Text cleaning, keyword-based category labelling
5. **Zero-Shot Baseline vs LoRA Fine-Tuning** — Two parallel classification approaches
6. **Evaluation & Explainability** — Accuracy, F1 scores, and keyword-level explainability analysis

---

## 🗂️ Project Structure

```
irish-parliament-nlp/
│
├── notebooks/
│   └── Irish_Parliamentary_Debate_NLP_LoRA_FineTuning.ipynb
│
├── data/
│   ├── debates.json                  # Raw API response (50 sitting days)
│   ├── oireachtas_labeled_clean.csv  # Full labelled dataset (5,550 speeches)
│   ├── train.csv                     # Training split (3,885 records)
│   ├── val.csv                       # Validation split (832 records)
│   └── test.csv                      # Test split (833 records)
│
├── results/                          # Evaluation outputs (generated on run)
│
├── irish_parliamentary_debate_nlp_lora_finetuning.py   # Full pipeline as Python script
├── system_architecture_diagram.png
├── requirements.txt
└── README.md
```

---

## 🗃️ Dataset

**Source:** [Oireachtas Open Data API](https://api.oireachtas.ie/v1/debates)  
**Coverage:** 34th Dáil Éireann — July 2024 to June 2025  
**Total speeches:** 5,550  
**Split:** 70% train / 15% validation / 15% test

### Category Distribution

| Category | Speeches |
|----------|----------|
| Government & Administration | 1,047 |
| Defence & Security | 573 |
| Health | 542 |
| Housing | 517 |
| Education & Childcare | 503 |
| Economy & Business | 431 |
| Transport & Infrastructure | 412 |
| Foreign Affairs & Trade | 386 |
| Energy & Environment | 311 |
| Social Welfare | 273 |
| Agriculture & Rural | 200 |
| Arts, Culture & Sport | 169 |
| Employment & Labour | 131 |
| Justice & Legacy Issues | 55 |

---

## 🧠 Model & LoRA Configuration

**Base model:** `Qwen/Qwen2.5-1.5B-Instruct`  
**Quantisation:** 4-bit (NF4) via BitsAndBytes  
**Fine-tuning method:** LoRA (Low Rank Adaptation) via PEFT

```python
LoraConfig(
    r=16,                    # Rank — controls adapter size
    lora_alpha=32,           # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

**Why LoRA?** Instead of retraining all 1.5B parameters, LoRA inserts small trainable matrices into the attention layers — drastically reducing compute and memory requirements while maintaining strong performance.

---

## 🔍 Explainability

The project includes a keyword-level explainability analysis to verify the model isn't making coincidental classifications. Key tokens driving predictions are extracted and analysed per category — showing the model genuinely learns policy-relevant vocabulary (e.g. "housing", "rent", "Garda", "carbon").

---

## 🛠️ Tech Stack

- **Python 3** — Core language
- **Transformers / PEFT** — LoRA fine-tuning
- **BitsAndBytes** — 4-bit quantisation
- **PyTorch** — Model training
- **Qwen2.5-1.5B-Instruct** — Base LLM
- **facebook/bart-large-mnli** — Zero-shot baseline
- **BeautifulSoup / lxml** — XML speech parsing
- **Scikit-learn** — Evaluation metrics
- **Oireachtas Open Data API** — Data source

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/roshankarumanchi/irish-parliament-nlp.git
   cd irish-parliament-nlp
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the full pipeline via the notebook:
   ```bash
   jupyter notebook notebooks/Irish_Parliamentary_Debate_NLP_LoRA_FineTuning.ipynb
   ```

   Or run the Python script directly (recommended on Colab with GPU):
   ```bash
   python irish_parliamentary_debate_nlp_lora_finetuning.py
   ```

> ⚠️ **GPU recommended** — LoRA fine-tuning requires a CUDA-enabled GPU. This project was developed on Google Colab (T4 GPU).

---

## 👤 Author

**Roshan Karumanchi**  
MSc Data Analytics — National College of Ireland, Dublin  
B.Tech Computer Science & Engineering — SRM University  
[LinkedIn](https://www.linkedin.com/in/) | [GitHub](https://github.com/roshankarumanchi)
