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