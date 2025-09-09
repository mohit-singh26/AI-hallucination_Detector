import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


# Load your dataset JSON file (update path accordingly)
dataset_path = 'data.json'  # Replace with your actual JSON file path
data = pd.read_json(dataset_path)

# Show basic dataset info
print("Dataset shape:", data.shape)
print("Columns:", data.columns)
print("Sample entries:")
print(data.head())

# Assign dataset columns for processing
# Assuming 'src' as prompt and 'hyp' as response
data['prompt'] = data['src']
data['response'] = data['hyp']

# If you have a label column, use it here.
# Since no label column is in your dataset, create dummy labels for example purposes (must replace with real labels)
data['label'] = 1  # Replace this with actual labels for supervised learning

# Check for missing values
print("Missing values per column:")
print(data.isnull().sum())


def clean_text(text):
    text = str(text).lower()  # lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove punctuation except spaces and alphanumerics
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text


# Apply cleaning to prompt and response columns
data['prompt'] = data['prompt'].apply(clean_text)
data['response'] = data['response'].apply(clean_text)

# Verify cleaning
print(data[['prompt', 'response']].head())


class SimpleTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def build_vocab(self, texts):
        tokens = set()
        for text in texts:
            for word in text.split():
                tokens.add(word)
        self.vocab = {word: i + 1 for i, word in enumerate(sorted(tokens))}
        self.vocab['<PAD>'] = 0
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def __call__(self, text):
        return [self.vocab.get(w, 0) for w in text.split()]

    def decode(self, indices):
        return ' '.join(self.inverse_vocab.get(i, '<UNK>') for i in indices)


# Initialize tokenizer
tokenizer = SimpleTokenizer()

# Build vocab from both prompt and response columns
all_texts = list(data['prompt']) + list(data['response'])
tokenizer.build_vocab(all_texts)

# Check vocab size and sample tokens
print("Vocabulary size:", len(tokenizer.vocab))
print("Sample tokenized prompt:", tokenizer(data['prompt'].iloc[0]))


class SemanticSimilarityDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data.iloc[idx]['prompt']
        response = self.data.iloc[idx]['response']
        label = self.data.iloc[idx]['label']

        prompt_tokens = torch.tensor(self.tokenizer(prompt), dtype=torch.long)
        response_tokens = torch.tensor(self.tokenizer(response), dtype=torch.long)
        label = torch.tensor(label, dtype=torch.float)

        return {'prompt': prompt_tokens, 'response': response_tokens, 'label': label}


dataset = SemanticSimilarityDataset(data, tokenizer)


def collate_fn(batch):
    # Pad sequences in batch to max length

    def pad_seq(seqs):
        max_len = max(len(s) for s in seqs)
        padded = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            padded[i, :len(s)] = s
        return padded

    prompts = pad_seq([item['prompt'] for item in batch])
    responses = pad_seq([item['response'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    return prompts, responses, labels


dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Check one batch sample
for prompts, responses, labels in dataloader:
    print(prompts.shape, responses.shape, labels.shape)
    break


class SiameseSemanticSimilarityModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def encode(self, x):
        emb = self.embedding(x)
        _, (hidden, _) = self.lstm(emb)
        return hidden.squeeze(0)

    def forward(self, prompt, response):
        prompt_emb = self.encode(prompt)
        response_emb = self.encode(response)
        combined = torch.cat([prompt_emb, response_emb], dim=1)
        out = self.fc(combined)
        return torch.sigmoid(out).squeeze(1)


vocab_size = len(tokenizer.vocab)
model = SiameseSemanticSimilarityModel(vocab_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


def train(model, dataloader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for prompts, responses, labels in dataloader:
            prompts, responses, labels = prompts.to(device), responses.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(prompts, responses)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")


train(model, dataloader, optimizer, criterion, epochs=5)

def evaluate(model, dataloader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for prompts, responses, labels in dataloader:
            prompts, responses, labels = prompts.to(device), responses.to(device), labels.to(device)
            outputs = model(prompts, responses)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Evaluation accuracy: {correct / total:.4f}")

# If you have a separate validation or test set, use that here instead of the training dataloader.
evaluate(model, dataloader)

# Save model state dict and tokenizer vocab for future use
torch.save(model.state_dict(), 'semantic_similarity_model.pth')
import pickle
with open('tokenizer_vocab.pkl', 'wb') as f:
    pickle.dump(tokenizer.vocab, f)
print("Model and vocabulary saved for future inference.")


def preprocess_text(text):
    # Same preprocessing as used for training
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict(model, tokenizer, prompt, response):
    model.eval()
    prompt = preprocess_text(prompt)
    response = preprocess_text(response)
    prompt_tokens = torch.tensor(tokenizer(prompt), dtype=torch.long).unsqueeze(0).to(device)
    response_tokens = torch.tensor(tokenizer(response), dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        score = model(prompt_tokens, response_tokens).item()
    print(f"Semantic similarity score (0-1): {score:.4f}")
    return score

# Example usage
prompt_example = "Tell me about the capital of France."
response_example = "Paris is the capital of France."
predict(model, tokenizer, prompt_example, response_example)
