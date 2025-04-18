# inference.py
import torch
from model import GPTModel
from tokenizers import Tokenizer

# === CONFIGURATION ===
vocab_size = 16000
embedding_dim = 128  # Updated from 256 to 128
max_position_embeddings = 256  # Ensure this matches the training configuration
n_layers = 3  # Updated to match the training configuration
n_head = 4  # Ensure this matches the training configuration
model_path = "gpt_model.pt"
tokenizer_path = "token train/custom_tokenizer.json"

# === Load Tokenizer ===
tokenizer = Tokenizer.from_file(tokenizer_path)

# === Load Model ===
model = GPTModel(vocab_size, embedding_dim, max_position_embeddings, n_layers, n_head)
model.load_state_dict(torch.load(model_path, map_location="cpu"))  # or "cuda" if you have GPU
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_text(prompt, max_tokens=50):
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_tensor)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            input_tensor = torch.cat((input_tensor, next_token), dim=1)

    output_ids = input_tensor[0].tolist()
    return tokenizer.decode(output_ids)

# === Test ===
if __name__ == "__main__":
    prompt = input("Enter your prompt: ")
    result = generate_text(prompt, max_tokens=50)
    print("Generated text:")
    print(result)
