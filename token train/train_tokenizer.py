from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Initialize the BPE tokenizer
tokenizer = Tokenizer(models.BPE())

# Use whitespace-based token splitting
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Configure training parameters
trainer = trainers.BpeTrainer(
    vocab_size=16000,  # Good starting point for small models
    special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
)

# List of training files
files = [
    "data/cleaned_books_combined.txt",
    "data/all_conversations.txt",
    "data/clean_bb.txt"
    
    # You can add your big book here later
]

# Train the tokenizer on your files
tokenizer.train(files, trainer)

# Save the trained tokenizer to disk
tokenizer.save("token train/custom_tokenizer.json")

print("✅ Tokenizer trained and saved as token train/custom_tokenizer.json")
