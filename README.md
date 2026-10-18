<p align="center">
  <img src="assets/readme-hero.svg" alt="Phishing Email Classifier — custom animated project banner" width="100%" />
</p>

<h1 align="center">Phishing Email Classifier</h1>

<p align="center">
  <strong>Make suspicious email signals easier to explain.</strong><br/>
  A modular scikit-learn pipeline that combines email text with lightweight security heuristics to classify messages as phishing or legitimate.
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-Pipeline-3776AB?style=for-the-badge&amp;logo=python&amp;logoColor=white" alt="Python — Pipeline" /></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit-learn-Machine_learning-F7931E?style=for-the-badge&amp;logo=scikitlearn&amp;logoColor=white" alt="scikit-learn — Machine learning" /></a>
  <a href="https://pandas.pydata.org/"><img src="https://img.shields.io/badge/Pandas-Data_prep-150458?style=for-the-badge&amp;logo=pandas&amp;logoColor=white" alt="Pandas — Data prep" /></a>
  <a href="https://github.com/tracolerd/Phishing-Email-Classifier"><img src="https://img.shields.io/badge/Model_pipeline-Explainable_signals-BE123C?style=for-the-badge&amp;logo=github&amp;logoColor=white" alt="Model pipeline — Explainable signals" /></a>
</p>

<p align="center">
  <a href="#project-details">Contents</a> · <a href="#project-setup">Install / run</a> · <a href="#project-architecture">Pipeline</a> · <a href="#project-limitations">Limitations</a> · <a href="#creator-developer-maintainer">Creator</a>
</p>

---

<a id="project-details"></a>
## 📌 Table of Contents

- [Overview](#-overview)
- [Why This Project](#-why-this-project)
- [Key Features](#-key-features)
- [How It Works](#-how-it-works)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Feature Engineering](#-feature-engineering)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Training the Model](#-training-the-model)
- [Using the Classifier](#-using-the-classifier)
- [Example JSON Input](#-example-json-input)
- [Example Output](#-example-output)
- [Confidence Levels](#-confidence-levels)
- [Explainability](#-explainability)
- [Design Decisions](#-design-decisions)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Security Notes](#-security-notes)
- [Technology Stack](#-technology-stack)
- [Contributing](#-contributing)
- [Author](#-author)

---

## 🎯 Overview

Phishing remains one of the most common ways attackers attempt to trick users into revealing credentials, making payments, opening malicious content, or taking other unsafe actions.

This project explores a practical defensive-security approach: instead of relying on a single indicator, it combines multiple signals from an email and feeds them into a machine-learning model.

The classifier considers two major categories of information:

1. **Email text** — the subject and body.
2. **Security-oriented heuristics** — URL count, attachment presence, and SPF authentication failure.

The result is a lightweight detection system that can be trained offline and then used through a simple CLI interface for inference.

---

## 💡 Why This Project

A phishing email can look convincing when evaluated from only one perspective. For example:

- A suspicious email may contain urgent language but no attachment.
- A legitimate newsletter may contain many URLs.
- An email with an attachment is not automatically malicious.
- An SPF failure is a useful signal, but it is not proof of phishing by itself.

Because of this, the project deliberately combines **textual features** with **structured heuristics** rather than treating any single indicator as a definitive answer.

This makes the project a useful demonstration of how **cybersecurity signals and machine learning can work together in a defensive workflow**.

---

## ✨ Key Features

- 🔎 **Phishing vs. legitimate email classification**
- 🧹 Custom text-cleaning transformer for HTML and normalization
- 📝 TF-IDF representation with unigram + bigram features
- 🌐 URL-count heuristic feature
- 📎 Attachment-presence heuristic feature
- 🔐 SPF-failure heuristic feature
- 🌲 Random Forest classifier
- 🧩 End-to-end Scikit-learn `Pipeline`
- 🔗 `ColumnTransformer` for parallel text and numeric branches
- 📦 Serialized model using Joblib
- 💻 Command-line inference interface
- 🧾 JSON input/output mode
- 🧑‍💻 Interactive inference mode
- 💬 Human-readable phishing explanations
- 🛡️ Defensive input validation
- ♻️ Stateless custom transformers designed to remain compatible with Scikit-learn tooling

---

<a id="project-architecture"></a>
## ⚙️ How It Works

At a high level, the system follows this flow:

```text
                ┌──────────────────────┐
                │     Email Input      │
                │ Subject + Body +     │
                │ Security Heuristics  │
                └──────────┬───────────┘
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
      ┌───────────────┐          ┌────────────────┐
      │ Text Cleaning │          │ Heuristic      │
      │ HTML removal  │          │ Extraction     │
      │ Normalization │          │ URL / Attach / │
      │ Subject+Body  │          │ SPF            │
      └───────┬───────┘          └───────┬────────┘
              │                          │
              ▼                          ▼
      ┌───────────────┐          ┌────────────────┐
      │   TF-IDF      │          │ StandardScaler │
      │ 1-2 grams     │          │ Numeric values │
      └───────┬───────┘          └───────┬────────┘
              │                          │
              └────────────┬─────────────┘
                           ▼
                  ┌─────────────────┐
                  │ Random Forest   │
                  │ Classification  │
                  └────────┬────────┘
                           ▼
             ┌───────────────────────────┐
             │ PHISHING / LEGITIMATE     │
             │ Probability + Confidence  │
             │ Explanation Signals       │
             └───────────────────────────┘
```

The full preprocessing and classification workflow is saved as a single Joblib artifact, allowing inference to reuse the exact preprocessing steps that were used during training.

---

## 🧠 Machine Learning Pipeline

The project uses an end-to-end Scikit-learn `Pipeline`.

### 1. Text branch

The subject and body are processed by a custom `TextCleanerTransformer`.

The transformer:

- Converts values to strings
- Removes HTML using BeautifulSoup
- Lowercases text
- Removes characters outside `[a-z0-9]`
- Collapses repeated whitespace
- Combines the cleaned subject and body

The cleaned text is then converted into numerical features with `TfidfVectorizer`.

Current TF-IDF configuration:

```python
{
    "max_features": 5000,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
}
```

### 2. Heuristic branch

The project extracts three numeric signals:

```text
url_count
has_attachments
spf_fail
```

The `HeuristicsExtractorTransformer` validates these columns, converts non-numeric values safely, replaces missing values with `0`, and produces a numeric matrix.

The values are then standardized using `StandardScaler`.

### 3. Feature combination

A `ColumnTransformer` keeps the two feature branches separate while allowing Scikit-learn to combine their outputs.

### 4. Classifier

The final estimator is a `RandomForestClassifier` configured with:

```python
{
    "n_estimators": 300,
    "max_depth": 20,
    "min_samples_leaf": 4,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
}
```

The training code is explicitly designed around a high-precision-oriented configuration intended to reduce unnecessary false-positive classifications.

---

## 🧩 Feature Engineering

The system uses a hybrid feature strategy.

### Textual signals

The model can learn patterns from words and short phrases appearing in the subject and body, such as urgency, account warnings, requests to verify information, payment-related language, and other contextual patterns present in the training data.

### Numeric security signals

| Feature | Description |
|---|---|
| `url_count` | Number of URLs associated with the email |
| `has_attachments` | `1` when an attachment is present, otherwise `0` |
| `spf_fail` | `1` when SPF authentication is marked as failed, otherwise `0` |

These signals are not treated as absolute proof of malicious behavior. They are additional evidence supplied to the model.

---

## 📊 Dataset

The included dataset is **synthetically generated** by `data/generate_mock_data.py`.

The generator creates:

- **1,200 total records**
- **600 phishing emails**
- **600 legitimate emails**
- A **50/50 balanced class distribution**
- Reproducible generation using seed `42`

The generator intentionally introduces overlap and noise between classes. For example, legitimate newsletters may contain several URLs, and legitimate messages may contain attachments. This is useful for demonstrating that classification should not be based on a single simplistic rule.

Dataset columns:

```text
subject
body
has_attachments
url_count
spf_fail
label
```

Where:

```text
label = 1  → Phishing
label = 0  → Legitimate
```

> **Important:** Because the current dataset is synthetic, model performance on this repository should not be interpreted as real-world phishing-detection performance. A production detector should be trained and validated on representative, properly labeled email data collected under appropriate privacy and security controls.

---

## 🗂️ Project Structure

```text
Phishing-Email-Classifier/
│
├── app.py                         # CLI inference application
├── requirements.txt               # Python dependencies
│
├── data/
│   ├── generate_mock_data.py      # Synthetic dataset generator
│   └── mock_emails.csv            # Generated training dataset
│
├── models/
│   └── phishing_pipeline.joblib   # Serialized trained pipeline
│
└── src/
    ├── __init__.py
    ├── preprocess.py              # Custom Scikit-learn transformers
    └── train.py                   # Model training + evaluation
```

---

<a id="project-setup"></a>
## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/tracolerd/Phishing-Email-Classifier.git
cd Phishing-Email-Classifier
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The project currently depends on Scikit-learn, Joblib, Pandas, NumPy, BeautifulSoup, and lxml. fileciteturn6file0

---

## 🏋️ Training the Model

The repository includes both a dataset and a generated model artifact. To retrain the pipeline from source, run:

```bash
python -m src.train
```

The training workflow performs the following steps:

```text
Load CSV
   ↓
Validate required columns
   ↓
Stratified 80/20 train-test split
   ↓
Build preprocessing pipeline
   ↓
Train Random Forest
   ↓
Generate classification report
   ↓
Export complete pipeline with Joblib
```

The split uses:

```python
TEST_SIZE = 0.20
RANDOM_STATE = 42
```

The trained artifact is written to:

```text
models/phishing_pipeline.joblib
```

The entire preprocessing + model pipeline is serialized together, which helps keep training-time and inference-time transformations consistent.

---

## 🧪 Using the Classifier

The project provides two inference modes.

### JSON mode

Pass the complete email payload directly from the command line:

```bash
python app.py --json '{"subject":"Verify now","body":"Click http://evil.com","url_count":5,"has_attachments":1,"spf_fail":1}'
```

### Interactive mode

Run:

```bash
python app.py --interactive
```

You will then be prompted for:

```text
Subject
Body
URL count
Attachments (0/1)
SPF fail (0/1)
```

---

## 🧾 Example JSON Input

```json
{
  "subject": "Urgent: Verify your account",
  "body": "We detected unusual activity. Click here to confirm your account.",
  "url_count": 5,
  "has_attachments": 1,
  "spf_fail": 1
}
```

The CLI validates the required fields before inference and returns an error when required input is missing or the JSON payload is malformed.

---

## 📤 Example Output

The application returns structured JSON containing:

```json
{
  "phishing_probability": 0.9876,
  "classification_label": "PHISHING",
  "confidence": "HIGH",
  "explanations": [
    "SPF authentication failure detected",
    "High URL count detected",
    "Unexpected attachment present",
    "Suspicious keywords detected: urgent, verify, account",
    "Model confidence: 99% phishing probability"
  ],
  "input_summary": {
    "url_count": 5,
    "has_attachments": 1,
    "spf_fail": 1
  }
}
```

> The probabilities shown above are an **illustrative output format**, not a recorded benchmark result. Actual values depend on the serialized model and the input provided.

---

## 📈 Confidence Levels

The CLI maps the predicted phishing probability to a simple confidence label:

| Phishing Probability | Confidence |
|---:|---|
| `>= 0.80` or `<= 0.20` | HIGH |
| `0.65–0.79` or `0.21–0.35` | MEDIUM |
| `0.36–0.64` | LOW |

These thresholds are application-level interpretation rules. They should not be confused with a calibrated probability guarantee.

---

## 💬 Explainability

For emails classified as phishing, the application generates lightweight human-readable explanations.

The explanation engine checks for signals such as:

- SPF authentication failure
- High URL count
- Elevated URL count
- Unexpected attachments
- Suspicious keywords in the subject/body
- High model phishing probability

The keyword list includes terms such as:

```text
urgent
verify
suspended
click here
confirm
password
account
limited
unusual activity
prize
winner
claim
bank
paypal
update billing
```

These explanations are **heuristic summaries** intended to make the result easier to interpret. They are not an exact feature-attribution method such as SHAP or LIME.

---

## 🏗️ Design Decisions

### End-to-end pipeline

Instead of preprocessing data manually and then saving only the classifier, the project stores the preprocessing stages and classifier together in one Scikit-learn pipeline.

This reduces the chance of training/inference preprocessing mismatches.

### Custom Scikit-learn transformers

`TextCleanerTransformer` and `HeuristicsExtractorTransformer` implement Scikit-learn's estimator interfaces through `BaseEstimator` and `TransformerMixin`.

That makes them compatible with common pipeline operations such as cloning and serialization.

### Defensive validation

The transformers validate DataFrame types and required columns, while numeric features are coerced safely where possible.

This makes the data-processing layer more robust when inputs eventually come from APIs, databases, or other external systems.

### Reproducibility

Random seeds are fixed at `42` for synthetic data generation and model training/splitting.

---

<a id="project-limitations"></a>
## ⚠️ Limitations

This project is best understood as a **research/portfolio-style defensive ML prototype**, not a complete enterprise email-security gateway.

Important limitations include:

1. **Synthetic training data** — the current dataset is programmatically generated rather than collected from real-world mail streams.
2. **Limited feature set** — URL count, attachment presence, and SPF failure are useful signals but do not cover the full phishing attack surface.
3. **No URL reputation analysis** — the system does not query domain reputation, WHOIS, DNS intelligence, URL sandboxing, or threat-intelligence feeds.
4. **No sender/header analysis** — the current inference schema does not model the full email-header ecosystem, reply-to mismatches, display-name spoofing, DKIM/DMARC alignment, Received headers, or sender reputation.
5. **No attachment inspection** — attachments are represented only as a binary presence flag; files themselves are not scanned.
6. **Synthetic-language bias** — text patterns learned from generated examples may not generalize to real phishing campaigns, especially multilingual or highly targeted attacks.
7. **Probability is not certainty** — a high probability score does not guarantee that an email is malicious.
8. **Explainability is heuristic** — the explanation list is generated from explicit rules and keywords rather than full model attribution.

These limitations are important when interpreting any test result from the repository.

---

## 🔮 Future Improvements

A stronger next-generation version could add:

### Data and evaluation

- Real-world, privacy-preserving phishing datasets
- Time-based train/validation/test splits
- Cross-validation
- Precision, recall, F1, ROC-AUC, PR-AUC, and confusion-matrix analysis
- Threshold tuning based on operational cost
- Probability calibration

### Email security intelligence

- SPF, DKIM, and DMARC alignment analysis
- Sender-domain reputation
- Reply-To / From mismatch detection
- Display-name spoofing detection
- Domain age and DNS signals
- URL lexical analysis
- Redirect-chain analysis
- Threat-intelligence integration
- Attachment metadata and malware-scan integrations

### Machine learning

- Linear baselines such as Logistic Regression
- Gradient boosting models
- Character-level and word-level representations
- Transformer-based NLP models
- Ensemble approaches
- Explainable ML with SHAP

### Deployment

- REST API with FastAPI
- Web dashboard
- Docker support
- CI/CD testing
- Model versioning
- Centralized logging and monitoring
- Drift detection
- Secure model update workflow

### Defensive integration

The classifier could eventually become one component inside a larger mail-security pipeline:

```text
Email Gateway
     ↓
Header / Sender Analysis
     ↓
URL & Domain Intelligence
     ↓
Content ML Classifier
     ↓
Attachment Security
     ↓
Risk Aggregation
     ↓
SOC / SIEM / Mailbox Action
```

This architecture would be substantially closer to a production defensive-security system than relying on a single ML classifier.

---

## 🔐 Security Notes

This project is intended for **defensive cybersecurity and email-security research**.

When integrating real email data:

- Remove or minimize sensitive personal information where possible.
- Avoid committing real credentials, tokens, API keys, or private email datasets to source control.
- Treat email bodies and attachments as untrusted input.
- Apply appropriate access controls and retention policies.
- Validate and sanitize any future API or database integrations.
- Keep threat-intelligence integrations and model artifacts versioned and auditable.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Core implementation |
| **Scikit-learn** | Preprocessing, feature transformation, pipeline, and model training |
| **Random Forest** | Final classification model |
| **Pandas** | Dataset and tabular-data handling |
| **NumPy** | Numerical processing |
| **BeautifulSoup4** | HTML-to-text cleaning |
| **lxml** | HTML parsing backend |
| **Joblib** | Model/pipeline serialization |
| **CLI + JSON** | Lightweight inference interface |

Dependency requirements are maintained in `requirements.txt`. fileciteturn6file0

---

## 🤝 Contributing

Contributions are welcome.

A useful contribution should ideally include:

1. A clear problem statement.
2. Reproducible changes.
3. Tests or evaluation evidence where appropriate.
4. Documentation updates for user-facing behavior.
5. No sensitive data, credentials, or proprietary email content.

Suggested contribution areas include improved datasets, stronger evaluation, richer email-security features, model comparison, API deployment, and explainability.

---

## ⭐ Project Summary

This project demonstrates a complete mini machine-learning workflow for a cybersecurity use case:

```text
Synthetic Data Generation
        ↓
Data Validation
        ↓
Custom Feature Engineering
        ↓
TF-IDF + Security Heuristics
        ↓
Random Forest Training
        ↓
Classification Report
        ↓
Serialized Model
        ↓
CLI Inference
        ↓
Probability + Decision + Explanation
```

The core idea is simple but important: **phishing detection is stronger when multiple independent signals are evaluated together rather than relying on a single rule.**

---

> ⚠️ **Disclaimer:** This project is provided for educational, research, and defensive-security purposes. It should not be treated as a standalone replacement for enterprise email-security controls, threat intelligence, sandboxing, or human security review.

---
---

---

## Creator, Developer & Maintainer

**Created, designed, developed, documented, and maintained independently by [Nurul Faiyaz](https://github.com/tracolerd).**

<p align="center">
  <a href="https://www.facebook.com/faiyaz.888"><img src="https://img.shields.io/badge/Facebook-1877F2?style=for-the-badge&amp;logo=facebook&amp;logoColor=white" alt="Facebook" /></a>
  <a href="https://www.instagram.com/tracolerd"><img src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&amp;logo=instagram&amp;logoColor=white" alt="Instagram" /></a>
  <a href="https://www.linkedin.com/in/faiyaz-xyz"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&amp;logo=linkedin&amp;logoColor=white" alt="LinkedIn" /></a>
  <a href="https://x.com/faiyaz_explores"><img src="https://img.shields.io/badge/X-000000?style=for-the-badge&amp;logo=x&amp;logoColor=white" alt="X" /></a>
  <a href="https://github.com/tracolerd"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&amp;logo=github&amp;logoColor=white" alt="GitHub" /></a>
  <a href="mailto:scrollfaiyaz@gmail.com"><img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&amp;logo=gmail&amp;logoColor=white" alt="Email" /></a>
  <a href="https://nurul-faiyaz.netlify.app/"><img src="https://img.shields.io/badge/Portfolio-Visit%20Website-0F766E?style=for-the-badge&amp;logo=googlechrome&amp;logoColor=white" alt="Website" /></a>
</p>

<p align="center">
  <a href="https://www.facebook.com/faiyaz.888">Facebook</a> ·
  <a href="https://www.instagram.com/tracolerd">Instagram</a> ·
  <a href="https://www.linkedin.com/in/faiyaz-xyz">LinkedIn</a> ·
  <a href="https://x.com/faiyaz_explores">X / Twitter</a> ·
  <a href="https://github.com/tracolerd">GitHub</a> ·
  <a href="mailto:scrollfaiyaz@gmail.com">Email</a> ·
  <a href="https://nurul-faiyaz.netlify.app/">Website</a>
</p>

## License

See [LICENSE](LICENSE) for the license terms.
