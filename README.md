# Sentiment Analysis API

A customer feedback sentiment analysis API built with FastAPI and PyTorch.

## Stack
- **Backend**: FastAPI + PyTorch (LSTM)
- **Pretrained Model**: Cardiff RoBERTa (3-class sentiment)

## Project Structure
```
sentiment-analysis/
├── main.py                 # FastAPI app entry point
├── train.py                # Model training script
├── requirements.txt
├── README.md
├── model/
│   ├── sentiment_model.py  # LSTM model + inference
│   └── pretrained_model.py # HuggingFace Cardiff model
└── routes/
    ├── sentiment.py        # LSTM endpoints
    └── pretrained.py       # Pretrained endpoints
```

## Setup

### 1. Install Python
```bash
sudo apt update
sudo apt install python3 python3-pip -y
```

### 2. Install dependencies
```bash
pip3 install -r requirements.txt --break-system-packages
```

### 3. Train the LSTM model
```bash
PYTHONPATH=. python3 train.py
```
Downloads the Yelp dataset from HuggingFace and trains the LSTM.
Saves weights to `model/trained_model.pt`.

### 4. Start the server
```bash
PYTHONPATH=. python3 -m uvicorn main:app --reload --port 8000
```

### 5. Open API docs
http://localhost:8000/docs

The first request to a pretrained endpoint downloads the Cardiff model (~500MB) to `model/pretrained/`.

## Endpoints

| Method | Endpoint                        | Description                     |
|--------|---------------------------------|---------------------------------|
| GET    | /health                         | Health check                    |
| POST   | /api/v1/analyze                 | Single text — LSTM              |
| POST   | /api/v1/analyze/bulk            | Multiple texts — LSTM           |
| POST   | /api/v1/analyze/pretrained      | Single text — Cardiff RoBERTa   |
| POST   | /api/v1/analyze/pretrained/bulk | Multiple texts — Cardiff RoBERTa|

## Example Request
```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "this product is absolutely amazing"}'
```

## Example Response
```json
{
  "label": "Positive",
  "scores": {
    "Negative": 0.0,
    "Neutral": 0.0,
    "Positive": 100.0
  },
  "summary": "100.0% positive, 0.0% neutral, 0.0% negative — clearly positive feedback"
}
```