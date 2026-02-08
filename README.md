# Retrieval Engine Competition — Data Science Project (M1 VMI)

This repository contains our **Information Retrieval Engine** project for the data science course in **Master 1 – Vision & Machine Intelligence (VMI)** second semester Paris Cité. The project is based on the Kaggle competition **“Retrieval Engine Competition”**.

**🔗 Kaggle Competition Link:** [https://www.kaggle.com/competitions/retrieval-engine-competition](https://www.kaggle.com/competitions/retrieval-engine-competition)

---

## Authors

| Name | Role / Focus |
| :--- | :--- |
| **Maksym DOLHOV** | Team Member |
| **Mehdi AGHAEI** | Team Member |
| **Nguyễn Hồ Bảo KHÁNH** | Team Member |
| **Nima DAVARI** | Team Member |

---

## Project Goal

The objective is to build an Information Retrieval (IR) system.
**Given:**
1.  A corpus of **Documents**.
2.  A set of **Queries**.
3.  **Relevance Judgments** (Ground Truth or Vérité Térrain).

**Task:**
Build a system that, for every query, **retrieves and ranks** the most relevant documents from the corpus.

---

## Repository Structure


```text
.
├── data/                     # GIT IGNORED (Local storage only)
│   ├── docs.json
│   ├── queries_train.json
│   ├── queries_test.json
│   └── qgts_train.json
├── notebooks/                #Jupyter notebooks for wxplains & experiments
│   ├── basic_analysis.ipynb
├── src/    
│   ├── __init__.py           # python package
│   ├── preprocess.py
│   ├── utils.py
│   ├── models.py             # logic goes here
│   └── evaluation.py         # Calculates Precision/Recall         
├── outputs/                  # GIT IGNORED (Logs, metrics, submission files)
│   ├── runs/
│   ├── submission.csv
├── report/                   # Slides, notes, and final report assets
├── .gitignore                
├── PIPELINE.md               #explanation of our workflow
├── requirements.txt          #dependencies
├── main.py                   #Runs the whole machine  
└── README.md  
```

## Setup & Usage
1. Environment Setup
Clone the repository and create a virtual environment:
```Bash
# Create virtual env
python3 -m venv .venv
```

# Activate (Linux/Mac)
```bash
source .venv/bin/activate
```
# Activate (Windows)
```bash
.venv\Scripts\activate
```

2. Install Dependencies
```Bash
pip install -r requirements.txt
```

# Or manually:
```bash
pip install numpy pandas tqdm scikit-learn rank_bm25 nltk sentence-transformers torch
```

3. Data Ingestion
Download the dataset from Kaggle and place the JSON files into data.

4. Running the Pipeline

Evaluation
We evaluate our models using the provided training relevance judgments (qrels). Key metrics include:
 - Precision@K: Proportion of relevant docs in the top $K$ results.
 - MAP (Mean Average Precision): Measures the quality of ranking across all queries. 
 
