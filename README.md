# Sentiment Analysis API

A customer feedback sentiment analysis app built with FastAPI and PyTorch, featuring both a custom LSTM model and a pretrained RoBERTa model.

---

## Stack

- Backend: FastAPI + PyTorch (LSTM)
- Pretrained Model: Cardiff RoBERTa (3-class sentiment)
- Frontend: React (Vite)

---

## Project Structure

sentiment-analysis/
├── main.py
├── train.py
├── requirements.txt
├── README.md
├── model/
│   ├── sentiment_model.py
│   └── pretrained_model.py
├── routes/
│   ├── sentiment.py
│   └── pretrained.py
└── frontend/
    ├── src/
    └── package.json

---

## Requirements

- Python 3.9+
- Node.js 18+
- ~1GB free disk space (for pretrained model)

---

## Setup

### 1. Clone the repository

git clone https://github.com/YOUR_USERNAME/sentiment-analysis.git
cd sentiment-analysis

---

## Backend Setup (FastAPI)

### 2. Create virtual environment

python3 -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

### 3. Install dependencies

pip install -r requirements.txt

---

### 4. Train the LSTM model (optional)

PYTHONPATH=. python train.py

This will:
- download Yelp dataset
- train the model
- save weights to model/trained_model.pt

---

### 5. Start the backend server

PYTHONPATH=. uvicorn main:app --reload --port 8000

API:
http://localhost:8000

Docs:
http://localhost:8000/docs

---

## Frontend Setup (React)

cd frontend
npm install
npm run dev

Frontend runs at:
http://localhost:5173

---

## Notes

- First request to pretrained endpoint downloads the model (~500MB)
- Ensure backend is running before using frontend

---

## Endpoints

Method | Endpoint                        | Description
------ | -------------------------------- | -----------------------------
GET    | /health                         | Health check
POST   | /api/v1/analyze                 | Single text — LSTM
POST   | /api/v1/analyze/bulk            | Multiple texts — LSTM
POST   | /api/v1/analyze/pretrained      | Single text — RoBERTa
POST   | /api/v1/analyze/pretrained/bulk | Multiple texts — RoBERTa

---

## Example Request

curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "this product is absolutely amazing"}'

---

## Example Response

{
  "label": "Positive",
  "scores": {
    "Negative": 0.0,
    "Neutral": 0.0,
    "Positive": 100.0
  },
  "summary": "100.0% positive, 0.0% neutral, 0.0% negative — clearly positive feedback"
}