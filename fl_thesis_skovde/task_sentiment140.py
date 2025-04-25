from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .torchdatasetwrapper import TorchDatasetWrapper
from collections import Counter
import re
import numpy as np
from collections import OrderedDict
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# Globals
fds = None
word2idx = {}
which_dataset = "sentiment"

# Model
class Net(nn.Module):
    def __init__(self, vocab_size=20000, embed_dim=100, hidden_dim=128, num_classes=3):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(self.embedding(x))
        return self.fc(hidden[-1])

# Padding
def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

# Tokenization + Vocabulary
def get_transforms(train_data, seq_len=50, vocab_size=20000):
    global word2idx

    def clean_and_tokenize(text):
        if isinstance(text, list):
            text = " ".join(text)
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in stop_words]
        return tokens

    if not word2idx:
        counter = Counter()
        for example in train_data:
            text = example["text"]
            counter.update(clean_and_tokenize(text))
        most_common = counter.most_common(vocab_size - 2)
        word2idx = {"<PAD>": 0, "<UNK>": 1}
        word2idx.update({word: idx + 2 for idx, (word, _) in enumerate(most_common)})

    def tokenize(example):
        tokens = clean_and_tokenize(example["text"])
        token_ids = [word2idx.get(token, word2idx["<UNK>"]) for token in tokens]
        padded = padding_([token_ids], seq_len)[0]
        return torch.tensor(padded, dtype=torch.long), torch.tensor(example["label"], dtype=torch.long)

    return tokenize


# Load Data
def load_data(partition_id: int, num_partitions: int, alpha_partition: float):
    global fds, word2idx

    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions, partition_by="label", alpha=alpha_partition, seed=42
        )
        fds = FederatedDataset(
            dataset="mteb/tweet_sentiment_extraction",
            partitioners={"train": partitioner}
        )

    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    transform_fn = get_transforms(partition_train_test["train"])
    train_dataset = TorchDatasetWrapper(partition_train_test["train"], transform_fn)
    test_dataset = TorchDatasetWrapper(partition_train_test["test"], transform_fn)

    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32)
    return trainloader, testloader

# Train
def train(net, trainloader, epochs, lr, device):
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for _ in range(epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return loss.item()

# Test
def test(net, testloader, device):
    net.to(device)
    net.eval()
    total_loss, correct = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            total_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    return total_loss / len(testloader), accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)