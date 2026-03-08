# AntConc-like Streamlit App

A lightweight corpus analysis app inspired by AntConc.

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
