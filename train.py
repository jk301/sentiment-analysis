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
    

# downloads yelp dataset and converts star ratings to 3 classes
def get_training_data(num_samples_per_class=3000):
    from datasets import load_dataset

    print("Downloading Yelp dataset...")
    dataset = load_dataset("yelp_review_full", split="train", streaming=True)

    texts = {0: [], 1: [], 2: []}

    for item in dataset:
        rating = item.get("label")  # 0-4 in yelp (0=1star, 4=5star)
        text = item.get("text", "").strip()

        if not text or rating is None:
            continue

        if rating <= 1:
            label = 0  # negative
        elif rating == 2:
            label = 1  # neutral
        else:
            label = 2  # positive

        if len(texts[label]) < num_samples_per_class:
            texts[label].append(text)

        if all(len(texts[l]) >= num_samples_per_class for l in [0, 1, 2]):
            break

    all_texts = texts[0] + texts[1] + texts[2]
    all_labels = [0]*len(texts[0]) + [1]*len(texts[1]) + [2]*len(texts[2])

    print(f"Negative: {len(texts[0])}, Neutral: {len(texts[1])}, Positive: {len(texts[2])}")
    return all_texts, all_labels



# trains the model and saves weights to a file
def train():
    # prepare data
    texts, labels = get_training_data()
    vocab = SimpleVocab()
    for text in texts:
        for word in text.lower().split():
            vocab.add(word)

    dataset = SentimentDataset(texts, labels, vocab)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # initialize model
    model = SentimentLSTM(vocab_size=vocab.size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
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
        scheduler.step()
        print(f"Epoch {epoch+1}/50 — Loss: {total_loss:.4f}")

    # save model and vocab
    torch.save({
        "model_state": model.state_dict(),
        "word2idx": vocab.word2idx
    }, "model/trained_model.pt")
    print("Model saved to model/trained_model.pt")

if __name__ == "__main__":
    train()