import torch
import torch.nn as nn
import torch.nn.functional as F

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_position_embeddings):
        """
        Converts token IDs into continuous embeddings and adds positional information.
        """
        super(InputEmbeddings, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embeddings = nn.Embedding(max_position_embeddings, embedding_dim)
        
    def forward(self, input_ids):
        # input_ids: (batch_size, seq_length)
        batch_size, seq_length = input_ids.size()
        # Create position IDs: 0, 1, 2, ... seq_length-1 for each sample in the batch
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_length)
        
        token_embeds = self.token_embeddings(input_ids)             # (batch_size, seq_length, embedding_dim)
        pos_embeds = self.positional_embeddings(position_ids)         # (batch_size, seq_length, embedding_dim)
        
        # Sum token and positional embeddings
        return token_embeds + pos_embeds

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, n_head, dropout_rate=0.1):
        """
        A single Transformer block with multi-head self-attention and a feed-forward network.
        """
        super(TransformerBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.head_dim = embedding_dim // n_head  # Ensure embedding_dim is divisible by n_head
        
        # Linear layer for queries, keys, and values (combined)
        self.qkv_proj = nn.Linear(embedding_dim, embedding_dim * 3)
        # Output projection layer for attention
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Feed-Forward Network (FFN): 2 linear layers with GELU activation in between
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        
        # Layer normalization and dropout layers for stabilization and regularization
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # x: (batch_size, seq_length, embedding_dim)
        residual = x
        
        # Compute combined queries, keys, and values
        qkv = self.qkv_proj(x)  # Shape: (batch_size, seq_length, 3 * embedding_dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # Each: (batch_size, seq_length, embedding_dim)
        
        # Reshape for multi-head attention: 
        batch_size, seq_length, _ = q.size()
        q = q.view(batch_size, seq_length, self.n_head, self.head_dim).transpose(1, 2)  # (batch_size, n_head, seq_length, head_dim)
        k = k.view(batch_size, seq_length, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.n_head, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Multiply attention weights with values
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, n_head, seq_length, head_dim)
        # Concatenate heads and reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embedding_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        # Apply residual connection and layer normalization
        x = self.ln1(residual + attn_output)
        
        # Feed-forward network with residual connection and normalization
        residual_ffn = x
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        x = self.ln2(residual_ffn + ffn_output)
        
        return x

class OutputHead(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        """
        Output projection layer that maps transformer hidden states to vocabulary logits.
        """
        super(OutputHead, self).__init__()
        self.out_proj = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, hidden_states):
        # hidden_states: (batch_size, seq_length, embedding_dim)
        logits = self.out_proj(hidden_states)  # (batch_size, seq_length, vocab_size)
        return logits

class GPTModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_position_embeddings, n_layers, n_head, dropout_rate=0.1):
        """
        Assembles the GPT-style model with input embeddings, transformer blocks, and an output head.
        """
        super(GPTModel, self).__init__()
        self.embeddings = InputEmbeddings(vocab_size, embedding_dim, max_position_embeddings)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embedding_dim, n_head, dropout_rate) for _ in range(n_layers)]
        )
        self.output_head = OutputHead(embedding_dim, vocab_size)
        
    def forward(self, input_ids):
        # Get the combined token and positional embeddings
        x = self.embeddings(input_ids)
        # Pass through each transformer block sequentially
        for block in self.transformer_blocks:
            x = block(x)
        # Project the final hidden states to vocabulary logits
        logits = self.output_head(x)
        return logits

# For a quick test when running model.py directly:
if __name__ == "__main__":
    vocab_size = 16000
    embedding_dim = 256
    max_position_embeddings = 256  # Sequence length
    n_layers = 4  # You can choose between 4-6 for your project
    n_head = 4    # Ensure that embedding_dim is divisible by n_head
    
    model = GPTModel(vocab_size, embedding_dim, max_position_embeddings, n_layers, n_head)
    dummy_input = torch.randint(0, vocab_size, (8, max_position_embeddings))  # Batch of 8, each sequence is 256 tokens
    logits = model(dummy_input)
    print("Model output shape:", logits.shape)  # Expect (8, 256, 16000)
