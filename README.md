# MarAcademy Learning Analytics & Personalization Chatbot

## Overview
A domain-specific chatbot for MarAcademy that answers questions about learner analytics, performance, and personalized recommendations, built using Hugging Face Transformers (T5), TensorFlow, and Streamlit.

## Features
- Conversational QA about learner engagement, progress, and recommendations
- Predicts learner performance category (At-Risk, Moderate, Excellent)
- Interactive UI with Streamlit
- Evaluation metrics: BLEU, F1-score, qualitative analysis

## Structure
- `data/`: Datasets (synthetic learner data, conversational pairs)
- `src/`: Scripts for preprocessing, model training, evaluation, chatbot logic
- `ui/`: Streamlit web app
- `notebook/`: Jupyter Notebook (EDA, model experiments)
- `report/`: Final project report

## Quickstart
```bash
pip install -r requirements.txt
python src/data_processing.py
python src/model_utils.py
streamlit run ui/app.py
```

## Assignment Rubric Alignment
See `report/LAPS_Project_Report.pdf` for details.

## Demo Video
[Link to demo video](#) (Add after recording)

## Authors
Omar Keita, MarAcademy