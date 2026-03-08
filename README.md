# KKU Corpus Analytics Streamlit App

A lightweight corpus analysis app inspired by AntConc.
Based on this research article:
Sameephet, B., Poonpon, K., Pradutshon, N., & Chansanam, W. (2025). Automated Vocabulary Profiling of TOEIC Listening Materials: A CEFR-Aligned Approach for EFL Learners. HighTech and Innovation Journal, 6(3), 991–1012. https://doi.org/10.28991/HIJ-2025-06-03-015

## Features
- Word List
- Concordance (KWIC)
- Concordance Plot
- File View
- Clusters / N-Grams
- Collocates
- Keyword List
- Exact / wildcard / regex search
- Thai-aware tokenization with PyThaiNLP
- Export current results to one Excel workbook

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
- Push `app.py`, `antconc_optimized.py`, `requirements.txt`, and `README.md` to GitHub.
- In Streamlit Community Cloud, choose the repository and set the main file to `app.py`.
- Deploy.

## Search tips
- Exact: `learn`
- Wildcard: `learn*`, `*ing`, `l?arn`
- Regex: `^learn`, `.*\d.*`
