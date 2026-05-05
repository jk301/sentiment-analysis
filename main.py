from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.sentiment import router as sentiment_router
from routes.pretrained import router as pretrained_router

app = FastAPI()

# allows the frontend to talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# registers the sentiment routes under /api/v1
app.include_router(sentiment_router, prefix="/api/v1")

# registers the sentiment routes under /api/v1
app.include_router(pretrained_router, prefix="/api/v1")

@app.get("/")
def home():
    return {"message": "Hello World"}

@app.get("/health")
def health():
    return {"status": "ok"}