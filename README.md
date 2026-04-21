# 📊 Social Media Sentiment Analyzer

> **Big-data sentiment analysis powered by PySpark, Spark MLlib & Streamlit**

A production-grade analytics platform that processes social media data (Sentiment140 — 1.6M tweets) through a full PySpark pipeline, trains a Logistic Regression classifier with Spark MLlib, and visualises results in a premium dark-themed Streamlit dashboard.

---

## ✨ Features

| Module | Description |
|--------|-------------|
| **🏠 Dashboard** | KPI cards, pie chart, trend line, keyword bars, word clouds, topic treemap, hourly heatmap |
| **🔍 Analysis** | Deep filtering, sentiment cards, topic breakdowns, statistical summaries |
| **🤖 ML Model** | Train Spark MLlib Logistic Regression with configurable hyperparameters, confusion matrix, per-class metrics |
| **🔮 Predict** | Custom tweet sentiment prediction using VADER + TextBlob + trained ML model |
| **⚙️ Settings** | Pipeline config, Spark connection test, dataset schema explorer |

### UI Highlights
- 🌑 **Premium dark glassmorphism** theme with animated gradients
- ✨ **Staggered fade-in** micro-animations
- 🎨 **Curated colour palette** (purple / teal / pink accents)
- 📱 **Responsive** breakpoints for tablet and mobile
- ⚡ **Interactive Plotly** charts with dark-mode styling

---

## 🚀 Quick Start

### Prerequisites
- **Python** 3.10+
- **Java** 8, 11, or 17 (for PySpark)  — set `JAVA_HOME`

### Install

```bash
# Clone
git clone <repo-url>
cd DSBDAL_Miniprj

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
streamlit run app/main.py
```

### (Optional) Load Real Data

1. Download [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) from Kaggle
2. Place CSV in `data/sample/`
3. Select **📁 Sentiment140** in the sidebar

---

## 📁 Project Structure

```
DSBDAL_Miniprj/
├── app/
│   ├── main.py                  # Entry point
│   ├── config.py                # Colours, paths, Spark config
│   ├── assets/
│   │   └── style.css            # Production dark theme (500+ lines)
│   ├── components/
│   │   ├── sidebar.py           # Navigation + filters
│   │   ├── dashboard.py         # 4-tab KPI dashboard
│   │   ├── analysis.py          # Deep analysis page
│   │   ├── ml_page.py           # ML training & eval UI
│   │   ├── predict_page.py      # Custom tweet predictor
│   │   └── visualizations.py    # 12 chart functions
│   ├── spark/
│   │   ├── session.py           # Cached SparkSession
│   │   ├── preprocessing.py     # Text cleaning UDFs
│   │   ├── sentiment.py         # VADER / TextBlob UDFs
│   │   ├── loader.py            # Sentiment140 pipeline
│   │   └── ml_pipeline.py       # MLlib LR pipeline
│   └── utils/
│       ├── helpers.py           # Data gen, formatting
│       └── constants.py         # Icons, mappings
├── data/sample/                 # Place Sentiment140 CSV here
├── models/                      # Saved MLlib models
├── requirements.txt
└── README.md
```

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit, Plotly, WordCloud |
| **Processing** | Apache Spark (PySpark) |
| **ML** | Spark MLlib (Logistic Regression) |
| **NLP** | VADER, TextBlob |
| **Data** | Pandas, NumPy |

---

## 📊 ML Pipeline

```
Text → Tokenizer → StopWordsRemover → HashingTF → IDF → LogisticRegression
```

**Metrics computed:** Accuracy, Weighted Precision/Recall/F1, Per-class metrics, Confusion Matrix, Feature Importance

---

## 📜 License

Academic project — DSBDA Lab Mini Project.
