import torch
import math
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# GPT-2 124M model configuration parameters (GPT-2 Small)
# This configuration matches the original GPT-2 Small model with 12 layers, 12 attention heads,
# and 768 hidden dimensions
config = {
    'vocab_size': 50257,
    'n_embd': 768,
    'n_layer': 12,
    'n_head': 12,
    'n_ctx': 1024,
    'dropout': 0.1,
    'bos_token_id': 50256,
    'eos_token_id': 50256,
    "qkv_bias": True,
}

class CausalSelfAttention(torch.nn.Module):
    """Causal Self-Attention module for transformer architecture
    
    Implements the multi-head attention mechanism with causal masking to ensure
    that predictions only depend on previous tokens in the sequence.
    """
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        # Query, Key, Value projection layer
        self.c_attn = torch.nn.Linear(config['n_embd'], 3 * config['n_embd'], bias=config['qkv_bias'])
        # Output projection layer
        self.c_proj = torch.nn.Linear(config['n_embd'], config['n_embd'])
        # Dropout layers
        self.attn_dropout = torch.nn.Dropout(config['dropout'])
        # Attention heads configuration
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        # Register lower triangular mask for causal attention
        self.register_buffer("bias", torch.tril(torch.ones(config['n_ctx'], config['n_ctx'])))
        
    def forward(self, x):
        B, T, C = x.size()  # Batch size, sequence length, embedding dimensionality
        
        # Split Q, K, V from combined projection
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        head_size = C // self.n_head
        
        # Reshape for multi-head attention (batch_size, num_heads, seq_len, head_size)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)
        
        # Calculate attention scores (scaled dot-product)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply causal mask to prevent attending to future tokens
        att = att.masked_fill(self.bias[None, None, :T, :T] == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Aggregate value vectors
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Reassemble all head outputs
        
        y = self.c_proj(y)
        return y

class MLP(torch.nn.Module):
    """Multi-Layer Perceptron module for transformer block
    
    Implements the feed-forward network with GELU activation used in each transformer block.
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc    = torch.nn.Linear(config['n_embd'], 4 * config['n_embd'])
        self.gelu    = torch.nn.GELU()
        self.c_proj  = torch.nn.Linear(4 * config['n_embd'], config['n_embd'])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(torch.nn.Module):
    """Transformer block containing attention and feed-forward layers
    
    Combines the causal self-attention mechanism with a feed-forward network
    and residual connections with layer normalization.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = torch.nn.LayerNorm(config['n_embd'])
        self.mlp = MLP(config)
        self.attn_dropout = torch.nn.Dropout(config['dropout'])
        self.mlp_dropout = torch.nn.Dropout(config['dropout'])

    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attn_dropout(self.attn(self.ln_1(x)))
        # Feed-forward network with residual connection
        x = x + self.mlp_dropout(self.mlp(self.ln_2(x)))
        return x

import argparse

class GPT2Small(torch.nn.Module):
    """GPT-2 Small model implementation (124M parameters)
    
    A custom implementation of the GPT-2 Small language model that can load
    pretrained weights from HuggingFace's transformer library.
    """
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # Token and position embedding tables
        self.token_embedding_table = torch.nn.Embedding(config['vocab_size'], config['n_embd'])
        self.position_embedding_table = torch.nn.Embedding(config['n_ctx'], config['n_embd'])
        self.dropout = torch.nn.Dropout(config['dropout'])
        # Transformer blocks
        self.blocks = torch.nn.Sequential(*[Block(config) for _ in range(config['n_layer'])])
        # Final layer normalization and language modeling head
        self.ln_f = torch.nn.LayerNorm(config['n_embd'])
        self.lm_head = torch.nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Load pretrained weights
        self.load_pretrained_weights()
        self.to(self.device)

    def load_pretrained_weights(self):
        """Load pretrained weights from HuggingFace's GPT-2 model
        
        Maps weights from the HuggingFace GPT-2 implementation to this custom model,
        handling differences in layer naming and weight shapes.
        """
        # Load pretrained model from HuggingFace
        pretrained_model = GPT2LMHeadModel.from_pretrained('gpt2')
        pretrained_state_dict = pretrained_model.state_dict()
        custom_state_dict = self.state_dict()
        
        # Weight name mapping between HuggingFace and custom implementation
        weight_map = {
            'transformer.wte.weight': 'token_embedding_table.weight',
            'transformer.wpe.weight': 'position_embedding_table.weight',
            'transformer.ln_f.weight': 'ln_f.weight',
            'transformer.ln_f.bias': 'ln_f.bias',
            'lm_head.weight': 'lm_head.weight'
        }
        
        # Add weight mappings for all transformer blocks
        for i in range(self.config['n_layer']):
            block_prefix = f'transformer.h.{i}.'
            custom_prefix = f'blocks.{i}.'
            
            # Layer normalization weights
            weight_map[block_prefix + 'ln_1.weight'] = custom_prefix + 'ln_1.weight'
            weight_map[block_prefix + 'ln_1.bias'] = custom_prefix + 'ln_1.bias'
            weight_map[block_prefix + 'ln_2.weight'] = custom_prefix + 'ln_2.weight'
            weight_map[block_prefix + 'ln_2.bias'] = custom_prefix + 'ln_2.bias'
            
            # Attention weights
            weight_map[block_prefix + 'attn.c_attn.weight'] = custom_prefix + 'attn.c_attn.weight'
            weight_map[block_prefix + 'attn.c_attn.bias'] = custom_prefix + 'attn.c_attn.bias'
            weight_map[block_prefix + 'attn.c_proj.weight'] = custom_prefix + 'attn.c_proj.weight'
            weight_map[block_prefix + 'attn.c_proj.bias'] = custom_prefix + 'attn.c_proj.bias'
            
            # MLP weights
            weight_map[block_prefix + 'mlp.c_fc.weight'] = custom_prefix + 'mlp.c_fc.weight'
            weight_map[block_prefix + 'mlp.c_fc.bias'] = custom_prefix + 'mlp.c_fc.bias'
            weight_map[block_prefix + 'mlp.c_proj.weight'] = custom_prefix + 'mlp.c_proj.weight'
            weight_map[block_prefix + 'mlp.c_proj.bias'] = custom_prefix + 'mlp.c_proj.bias'
        
        # Load and transpose weights as needed
        for hf_name, custom_name in weight_map.items():
            if hf_name in pretrained_state_dict:
                weight = pretrained_state_dict[hf_name]
                # Transpose attention and MLP weights to match PyTorch Linear layer dimensions
                # HuggingFace stores weights as (out_features, in_features), while PyTorch Linear expects (in_features, out_features)
                if 'c_attn.weight' in custom_name or 'c_fc.weight' in custom_name or 'c_proj.weight' in custom_name:
                    weight = weight.t()
                custom_state_dict[custom_name] = weight
        
        self.load_state_dict(custom_state_dict)
        print("Successfully loaded pretrained weights!")

    def forward(self, idx, targets=None):
        """Forward pass of the GPT-2 model
        
        Args:
            idx: Input token indices of shape (batch_size, seq_len)
            targets: Optional target token indices for training
        
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss if targets are provided, None otherwise
        """
        B, T = idx.size()
        
        # Token embeddings
        tok_emb = self.token_embedding_table(idx)
        
        # Position embeddings (dynamic truncation to context window size)
        pos = torch.arange(0, T, device=idx.device)
        pos_emb = self.position_embedding_table(pos[:min(T, self.config['n_ctx'])])
        if T > self.config['n_ctx']:
            pos_emb = torch.nn.functional.pad(pos_emb, (0, 0, 0, T - self.config['n_ctx']))
        
        # Combine embeddings and apply dropout
        x = self.dropout(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Calculate loss if targets are provided
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss

    def generate(self, input_text, max_new_tokens=50, temperature=0.7, top_k=None):
        """Generate text continuation from input text
        
        Args:
            input_text: Starting text for generation
            max_new_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (lower = more deterministic)
            top_k: If set, only sample from top k tokens
        
        Returns:
            Generated text continuation (excluding input text)
        """
        # Encode input text
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        
        # Use configured device
        input_ids = input_ids.to(self.device)
        
        # Generation mode
        self.eval()
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass (keep only last 1024 tokens to respect context window)
                inputs = generated[:, -self.config['n_ctx']:]
                logits, _ = self(inputs)
                
                # Get logits for last token and apply temperature scaling
                last_logits = logits[:, -1, :] / temperature
                
                # Top-k sampling
                if top_k is not None:
                    v, _ = torch.topk(last_logits, top_k)
                    last_logits[last_logits < v[:, [-1]]] = -float('Inf')
                
                # Convert logits to probability distribution
                probs = torch.nn.functional.softmax(last_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                # next_token = torch.argmax(probs, dim=-1, keepdim=True)
                
                # Concatenate new token to sequence
                generated = torch.cat((generated, next_token), dim=1)
                
                # Stop when EOS token is encountered
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode generated text and return only new tokens
        input_length = input_ids.shape[1]
        return self.tokenizer.decode(generated[0, input_length:], skip_special_tokens=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='GPT-2 Small Text Generation')
    parser.add_argument('--input-text', type=str, default="What's the capital of China?", help='Input text for generation')
    parser.add_argument('--max-new-tokens', type=int, default=30, help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for sampling')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling parameter')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    # Uncomment below code to fix output for reproducibility
    torch.manual_seed(123)
    gpt2_small = GPT2Small(config, device=args.device)
    
    generated_text = gpt2_small.generate(
        input_text=args.input_text,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )
    
    print(f"Input: {args.input_text}")
    print(f"Generated: {generated_text}")