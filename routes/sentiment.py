# route sentiment #

from fastapi import APIRouter
from pydantic import BaseModel
from model.sentiment_model import analyze_sentiment

# defines the request body shape
class FeedbackRequest(BaseModel):
    text: str

# creates a router 
router = APIRouter()

# endpoint that receives text and returns sentiment
@router.post("/analyze")
def analyze(req: FeedbackRequest):
    result = analyze_sentiment(req.text)
    return result


# defines the request body shape for bulk analysis
class BulkFeedbackRequest(BaseModel):
    texts: list[str]

# endpoint that receives multiple texts and returns sentiment for each
@router.post("/analyze/bulk")
def analyze_bulk(req: BulkFeedbackRequest):
    results = []
    for text in req.texts:
        result = analyze_sentiment(text.strip())
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