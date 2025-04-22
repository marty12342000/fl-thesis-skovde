from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter, OrderedDict

# === Global variables ===
fds = None
word2idx = {}

# === Model ===
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_layers=1):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 3)  # 3 sentiment classes: pos/neg/neutral

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])

# === Vocabulary Builder ===
def build_vocab(dataset, min_freq=5):
    counter = Counter()
    for row in dataset["train"]:
        counter.update(row["x"].split())
    vocab = [word for word, freq in counter.items() if freq >= min_freq]
    vocab = ["<pad>", "<unk>"] + vocab
    return {word: idx for idx, word in enumerate(vocab)}

# === Collate Function ===
def collate_fn(batch):
    inputs = [item["text"] for item in batch]
    labels = torch.tensor([item["sentiment"] for item in batch], dtype=torch.long)
    padded_inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    return padded_inputs, labels

# === Data Loader ===
def load_sentiment140(partition_id: int, num_partitions: int, alpha_partition: float):
    global fds, word2idx

    # Initialize dataset and partitioner once
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions, partition_by="sentiment", alpha=alpha_partition, seed=42
        )
        fds = FederatedDataset(
            dataset="sentiment140",  # from Hugging Face
            partitioners={"train": partitioner},
        )

        # Build vocabulary only once
        raw_partition = fds.load_partition(0)
        raw_data = raw_partition.load_raw()["train"]
        word2idx = build_vocab(raw_data)

    # Load partition for this client
    partition = fds.load_partition(partition_id)

    # Tokenize each batch
    def tokenize(batch):
        inputs, labels = [], []
        for text, label in zip(batch["x"], batch["y"]):
            tokens = [word2idx.get(word, word2idx["<unk>"]) for word in text.split()]
            inputs.append(torch.tensor(tokens, dtype=torch.long))
            labels.append(torch.tensor(label, dtype=torch.long))
        return {"text": inputs, "sentiment": labels}

    # Apply tokenization
    transformed = partition.with_transform(tokenize)

    # Create DataLoaders
    trainloader = DataLoader(transformed["train"], batch_size=32, shuffle=True, collate_fn=collate_fn)
    testloader = DataLoader(transformed["test"], batch_size=32, collate_fn=collate_fn)
    return trainloader, testloader

# === Training ===
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

# === Testing ===
def test(net, testloader, device):
    net.to(device)
    net.eval()
    total_loss, correct = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            total_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return total_loss / len(testloader), accuracy

# === Federated Weights ===
def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)