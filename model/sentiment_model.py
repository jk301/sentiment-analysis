# sentiment_model.py #

import torch
import torch.nn as nn
import numpy as np

class SimpleVocab:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self._next = 2

    def add(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self._next
            self._next += 1

    def encode(self, text, max_len=64):
        tokens = text.lower().split()[:max_len]
        ids = [self.word2idx.get(t, 1) for t in tokens]
        ids += [0] * (max_len - len(ids))
        return ids

    @property
    def size(self):
        return self._next
    
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        pooled = out.mean(dim=1)
        return self.fc(pooled)
    

# stores the model and vocab in memory so we don't reload them on every request
_vocab = None
_model = None
LABELS = ["Negative", "Neutral", "Positive"]

# builds a basic vocabulary from common sentiment words
def _build_vocab():
    vocab = SimpleVocab()
    words = (
        "good great excellent amazing wonderful fantastic love like enjoy perfect"
        " bad terrible awful horrible hate dislike poor worst broken damaged"
        " okay average neutral fine mediocre acceptable decent"
        " product service quality fast slow delivery price value"
        " happy satisfied disappointed frustrated impressed recommend"
    ).split()
    for w in words:
        vocab.add(w)
    return vocab


# loads trained weights if available, otherwise falls back to random weights
def _get_model():
    global _vocab, _model
    if _model is None:
        _vocab = _build_vocab()
        _model = SentimentLSTM(vocab_size=_vocab.size)
        
        try:
            checkpoint = torch.load("model/trained_model.pt", weights_only=True)
            # rebuild vocab from checkpoint
            _vocab.word2idx = checkpoint["word2idx"]
            _vocab._next = max(checkpoint["word2idx"].values()) + 1
            # reload model with correct vocab size
            _model = SentimentLSTM(vocab_size=_vocab.size)
            _model.load_state_dict(checkpoint["model_state"])
            print("Loaded trained model weights")
        except FileNotFoundError:
            print("No trained weights found, using random weights")
        
        _model.eval()
    return _vocab, _model


# generates a plain summary based on the scores
def generate_summary(label, scores):
    top_score = scores[label]
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # determine strength based on confidence
    if top_score >= 70:
        strength = "clearly"
    elif top_score >= 50:
        strength = "moderately"
    else:
        strength = "mixed feedback, leaning"
    
    # build the score string
    score_str = ", ".join(f"{v}% {k.lower()}" for k, v in sorted_scores)
    
    return f"{score_str} — {strength} {label.lower()} feedback"


# takes a string, runs it through the model, returns label and scores
def analyze_sentiment(text):
    vocab, model = _get_model()
    ids = vocab.encode(text)
    tensor = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1).squeeze().numpy()

    label = LABELS[int(np.argmax(probs))]
    scores = {LABELS[i]: round(float(probs[i]) * 100, 1) for i in range(3)}

    return {
        "label": label,
        "scores": scores,
        "summary": generate_summary(label, scores)
    }
    
