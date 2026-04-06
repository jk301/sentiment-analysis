# train.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model.sentiment_model import SentimentLSTM, SimpleVocab

# wraps our text data into a format PyTorch can work with
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=64):
        self.encodings = [vocab.encode(t, max_len) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encodings[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )
    

# prepares a small hardcoded dataset for training
def get_training_data():
    texts = [
        # positive
        "this product is amazing", "absolutely love it", "great quality highly recommend",
        "fantastic service very happy", "excellent product works perfectly",
        "best purchase ever", "very satisfied with this", "wonderful experience",
        # negative
        "terrible product broke immediately", "worst purchase ever avoid",
        "completely disappointed very bad", "horrible quality do not buy",
        "awful experience never again", "very poor quality waste of money",
        "broken on arrival terrible", "hate this product useless",
        # neutral
        "it is okay nothing special", "average product does the job",
        "decent quality for the price", "neither good nor bad just fine",
        "acceptable but not great", "mediocre product works sometimes",
        "okay experience could be better", "fine for basic use",
    ]
    labels = [2]*8 + [0]*8 + [1]*8  # 2=positive, 0=negative, 1=neutral
    return texts, labels




# trains the model and saves weights to a file
def train():
    # prepare data
    texts, labels = get_training_data()
    vocab = SimpleVocab()
    for text in texts:
        for word in text.lower().split():
            vocab.add(word)

    dataset = SentimentDataset(texts, labels, vocab)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # initialize model
    model = SentimentLSTM(vocab_size=vocab.size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # training loop
    model.train()
    for epoch in range(50):
        total_loss = 0
        for texts_batch, labels_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(texts_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50 — Loss: {total_loss:.4f}")

    # save model and vocab
    torch.save({
        "model_state": model.state_dict(),
        "word2idx": vocab.word2idx
    }, "model/trained_model.pt")
    print("Model saved to model/trained_model.pt")

if __name__ == "__main__":
    train()