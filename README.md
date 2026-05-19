# Movie Feedback Sentiment Analysis

A Flask web app to predict sentiment (Positive/Negative) from movie feedback text.

## Deployed Link

https://sentiment-analysis-13k3.onrender.com

## Model Details

- Model: Bernoulli Naive Bayes (BernoulliNB)
- Vectorizer: CountVectorizer (`cv.pkl`)
- Accuracy: 0.8487

## Project Structure

```text
Sentiment_Analysis/
|-- app.py
|-- model.pkl
|-- cv.pkl
|-- IMDB Dataset.csv
|-- Sentiment-Analysis.ipynb
|-- requirements.txt
|-- README.md
|-- templates/
|   `-- index.html
`-- static/
    `-- style.css
```

## Installation

1. Clone or download this project.
2. Open terminal in project folder.
3. (Recommended) Create a virtual environment:

```bash
python -m venv .venv
```

4. Activate virtual environment:

Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

5. Install required packages:

```bash
pip install -r requirements.txt
```

## NLTK Setup (One Time)

This app uses NLTK stopwords. Run once:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

## Run the App

```bash
python app.py
```

Open browser:

- http://127.0.0.1:5000/

## Input / Output

### Input

- User enters movie feedback text in textarea.
- Example: `This movie was boring and too long.`

### Output

- Predicted sentiment:
  - Positive Sentiment
  - Negative Sentiment
- Shows:
  - Your input
  - Prediction input used (preprocessed text)

## Preprocessing Pipeline

The backend applies these steps before prediction:

1. Remove HTML tags
2. Convert text to lowercase
3. Remove special characters
4. Remove stopwords (keeps important negation words)
5. Apply stemming
6. Transform using CountVectorizer
7. Predict using BernoulliNB model

## Requirements

Main dependencies:

- flask
- scikit-learn
- nltk
- numpy
- pandas

## Notes

- Keep `model.pkl` and `cv.pkl` in root folder.
- If scikit-learn version warning appears, use the same sklearn version used while training.

## Future Improvements

- Show confidence score/probability
- Add prediction history
- Add model comparison page
