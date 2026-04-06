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