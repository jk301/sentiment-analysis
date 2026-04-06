# pretrained.py

from fastapi import APIRouter
from pydantic import BaseModel
from model.pretrained_model import analyze_sentiment_pretrained

# defines the request body shape
class FeedbackRequest(BaseModel):
    text: str

# creates a router for pretrained endpoints
router = APIRouter()

# endpoint that uses the pretrained model
@router.post("/analyze/pretrained")
def analyze_pretrained(req: FeedbackRequest):
    result = analyze_sentiment_pretrained(req.text)
    return result


# defines the request body shape for bulk analysis
class BulkFeedbackRequest(BaseModel):
    texts: list[str]

# endpoint that receives multiple texts and returns pretrained sentiment for each
@router.post("/analyze/pretrained/bulk")
def analyze_pretrained_bulk(req: BulkFeedbackRequest):
    results = []
    for text in req.texts:
        result = analyze_sentiment_pretrained(text.strip())
        results.append(result)

    # count each label
    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for r in results:
        counts[r["label"]] += 1

    return {
        "results": results,
        "summary": {
            "total": len(results),
            "positive": counts["Positive"],
            "neutral": counts["Neutral"],
            "negative": counts["Negative"]
        }
    }