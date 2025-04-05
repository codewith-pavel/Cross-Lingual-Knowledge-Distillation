# Bridging High-Resource to Low-Resource Language Gaps in Clinical NLP

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Repo](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/codewith-pavel/Cross-Lingual-Knowledge-Distillation)

This repository hosts the complete implementation and experimental artifacts for the paper:

**"Refining Clinical Outcome Prediction via Cross-Lingual Methods and Adaptive Translation Strategies with SHAP"**  
*Accepted at [Conference/Journal Name]*

---

## 📌 Abstract

Low-resource languages like Bengali are often underserved in clinical natural language processing (NLP) tasks due to the lack of annotated datasets and pretrained language models. In this work, we address these challenges through a pipeline incorporating **cross-lingual knowledge distillation (CLKD)** and **adaptive translation strategies**. Our key innovations include:

- **Adaptive translation** using Google Translate, AI Sheets™, and Gemini 2.0 to create high-quality Bengali clinical corpora.
- **Model interpretability** using SHAP to provide granular insight into feature importance.
- **Ensemble teacher models** leveraging ClinicalBERT and dynamic attention-weighted ensembling.
- **Cost-sensitive learning frameworks** to mitigate class imbalance in clinical predictions (mortality and length-of-stay).

We achieve **state-of-the-art AUC scores**: 
- **0.8716** for *mortality prediction*
- **0.7877** for *length-of-stay prediction* using MIMIC-III.

---

```

**Pretrained Models**: Download from HuggingFace:
- [BanglaBERT](https://huggingface.co/csebuetnlp/banglabert)
- [ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
- [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base)

---

## 🧠 Key Components

### 🔁 Translation Workflow

We use a multi-source translation mechanism:
- **Google Translate API**
- **Gemini 2.0 (Google AI)** for contextual clinical translation.
- Back-translation evaluation with **BLEU**, **ROUGE**, and **METEOR**.

```python
# Gemini 2.0 Translation Prompt
generation_config = {
    "temperature": 0.1,
    "top_p": 0.4,
    "top_k": 40,
    "max_output_tokens": 8192
}
```

---

### 📘 Cross-Lingual Knowledge Distillation (CLKD)

- **Teacher Model**: Ensemble of ClinicalBERTs with task-specific fine-tuning.
- **Student Model**: Bengali-supported multilingual transformers like XLM-RoBERTa or custom DistilBERT.
- **Attention fusion**: Weighted softmax over logits from each teacher.

---

### 🔍 Model Interpretability with SHAP

We use SHAP to understand the influence of translated text segments:

```python
import shap
explainer = shap.Explainer(model, tokenizer)
shap_values = explainer(bengali_text)
shap.plots.text(shap_values[0])
```

---

### ⚖️ Cost-Sensitive Learning

To combat clinical class imbalance:
\[ w_c = \frac{N}{K \cdot n_c} \]
Where:
- \(N\): total number of samples
- \(K\): total number of classes
- \(n_c\): number of samples in class \(c\)

Implemented with `sklearn.utils.class_weight` and used directly in `loss_fn`.

---

## 📊 Results

| Model Type              | Mortality AUC | Length-of-Stay AUC | Inference Speed (iter/s) |
|------------------------|---------------|---------------------|---------------------------|
| **Ensemble (Teacher)** | **0.8716**    | **0.7877**          | 2.3                       |
| **XLM-RoBERTa (CLKD)** | 0.8493        | 0.7134              | 4.7                       |
| **Mixed-Distil-BERT**  | 0.7516        | 0.6878              | 7.3                       |

### 🔤 Translation Quality

| Model            | BLEU-1 | ROUGE-2 |
|------------------|--------|----------|
| **Gemini 2.0**   | 0.7820 | 0.7749   |
| **Google Translate** | 0.7053 | 0.7134   |

---

## 📂 Dataset

- **MIMIC-III**: Contains 92,293 English clinical notes. 
- Bengali-translated corpus created through adaptive translation strategies.
- Protected under PhysioNet's data use agreement.

> 🔒 Note: Bengali translations are not publicly released due to institutional data sharing restrictions.

---

## 🤝 Contributing

We welcome contributions in:
- Expanding support for new low-resource languages (e.g., Amharic, Sinhala).
- Improving translation quality metrics and interpretability.
- Enhancing knowledge distillation frameworks with multimodal inputs.

Fork, create a feature branch, and submit a pull request (PR).

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 📄 Citation

```bibtex
@article{pavel2024bridging,
  title={Bridging High-Resource to Low-Resource Language Gaps: Refining Clinical Outcome Prediction via Cross-Lingual Methods and Adaptive Translation Strategies with SHAP},
  author={Pavel, Mahir Afser and Islam, Rafiul and Hasan, Mohammad Junayed and Mahdy, M.R.C.},
  journal={Journal of Biomedical Informatics},
  year={2024}
}
```

---
