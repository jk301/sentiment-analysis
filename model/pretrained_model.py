# pretrained_model.py

from transformers import pipeline

# stores the pipeline in memory so we don't reload it on every request
_pipeline = None

# loads the pretrained model once and reuses it
def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        print("Loading pretrained model...")
        _pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            cache_dir="model/pretrained"
        )
        print("Pretrained model loaded")
    return _pipeline

# takes a string and returns label and scores using pretrained model
def analyze_sentiment_pretrained(text):
    pipe = _get_pipeline()
    result = pipe(text, truncation=True, max_length=512)[0]
    
    label = result["label"].capitalize()
    confidence = round(result["score"] * 100, 1)
    
    return {
        "label": label,
        "confidence": confidence,
        "summary": f"{confidence}% confident — {label.lower()} feedback"
    }