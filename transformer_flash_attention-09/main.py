import argparse
import math
import torch
from pathlib import Path
import sys
# Add parent directory to system path for importing from transformer_gpt2_07
sys.path.append(str(Path(__file__).parent.parent))
from transformer_gpt2_07.main import GPT2Small, config
from flash_attention import FlashOrOriginalAttention

import torch
import math

import torch
import math
import torch.nn as nn

class GPT2SmallFlashAttention(GPT2Small):
    def __init__(self, config, device=None, enable=True):
        # Initialize parent GPT2Small class
        super().__init__(config, device)

        # Replace attention blocks with FlashAttention
        for block in self.blocks:
            # Create new FlashAttention module with same configuration
            flash_attn = FlashOrOriginalAttention(config, enable_flash = enable)

            # Copy weights from original attention module
            flash_attn.c_attn.weight.data = block.attn.c_attn.weight.data
            flash_attn.c_attn.bias.data = block.attn.c_attn.bias.data
            flash_attn.c_proj.weight.data = block.attn.c_proj.weight.data
            flash_attn.c_proj.bias.data = block.attn.c_proj.bias.data

            # Replace the attention module in the block
            block.attn = flash_attn
        if enable:
            print("Flash Attention enabled!")
        self.to(device)  # Move model to specified device


def generate(self, text, device=None, max_length=100, temperature=0.7):
    """
    Generate text using the GPT-2 model
    Args:
        text(str): Input prompt text
        device(str, optional): Device to run inference on(e.g., 'cpu' or 'cuda')
        max_length(int): Maximum length of generated text
        temperature(float): Sampling temperature for text generation

    Returns:
        str: Generated text
    """
    self.eval()  # Set model to evaluation mode

    # Tokenize input text
    inputs = self.tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=self.config['n_ctx']
    ).to(device)

    # Generate text with greedy sampling
    with torch.no_grad():
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        for _ in range(max_length - input_ids.shape[1]):
            # Get model outputs
            logits, _ = self(input_ids)
            # Take logits for the last token
            next_token_logits = logits[:, -1, :]
            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            # Update attention mask
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token)], dim=-1)

            # Stop if we reach EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break

    # Decode generated text
    generated_text = self.tokenizer.decode(
        input_ids[0], skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":
    torch.manual_seed(123)
    parser = argparse.ArgumentParser(
        description='Generate text with GPT2SmallFlashAttention')
    parser.add_argument('--text', type=str,
                        default='Hello, world!', help='Input text prompt')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., "cpu" or "cuda")')
    args = parser.parse_args()

    sample_text = args.text
    device = args.device if args.device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu")

    print(f"Generating text with prompt: '{sample_text}'")
    # Initialize model with Flash Attention using original config
    model = GPT2SmallFlashAttention(config, device=device)
    orig_model = GPT2SmallFlashAttention(config, device=device, enable=False)
    generated_text = model.generate(
        input_text=sample_text,
        max_new_tokens=30,
        temperature=0.8
    )
    orig_generated_text = orig_model.generate(
        input_text=sample_text,
        max_new_tokens=30,
        temperature=0.8
    )

    print("\nOrig Generated Text:")
    print(orig_generated_text)
    print("\nGenerated Text:")
    print(generated_text)
