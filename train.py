import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import GPTModel  # Import the GPT model architecture

# Load preprocessed training data from tokenized data/train.pt
data = torch.load("tokenized data/train.pt")
X = data["X"]  # Input sequences, e.g., shape: (num_samples, seq_length)
y = data["y"]  # Target sequences, e.g., shape: (num_samples, seq_length)

# Create a TensorDataset and DataLoader
dataset = TensorDataset(X, y)
batch_size = 32  # Adjust based on available GPU memory
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model hyperparameters (must match those used during tokenizer setup)
vocab_size = 16000
embedding_dim = 256
max_position_embeddings = X.size(1)  # Use the sequence length from the dataset
n_layers = 4    # For your project, 4-6 transformer blocks is a good starting point
n_head = 4

# Instantiate the GPT model
model = GPTModel(vocab_size, embedding_dim, max_position_embeddings, n_layers, n_head)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Commonly used for next-token prediction
optimizer = optim.AdamW(model.parameters(), lr=5e-4)

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        # Move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass: get logits from the model
        logits = model(inputs)  # Expected shape: (batch_size, seq_length, vocab_size)
        
        # Reshape logits and targets for the loss computation
        logits = logits.view(-1, vocab_size)  # (batch_size * seq_length, vocab_size)
        targets = targets.view(-1)            # (batch_size * seq_length)
        
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if batch_idx % 300 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

# Save the trained model checkpoint
torch.save(model.state_dict(), "tokenized data/gpt_model.pt")
print("Model saved as tokenized data/gpt_model.pt")
