import torch
import math
import torch.nn as nn
from pathlib import Path
import sys
# Add parent directory to system path for importing from transformer_gpt2_07
sys.path.append(str(Path(__file__).parent.parent))
from transformer_gpt2_07.main import CausalSelfAttention, config

class FlashOrOriginalAttention(CausalSelfAttention):
    """Causal Self-Attention with corrected Flash Attention implementation"""

    def __init__(self, config, enable_flash=True, state_dict=None, debug=False):
        super().__init__(config)
        self.dropout = config.get('dropout', 0.0)
        self.enable_flash = enable_flash
        self.debug = debug
        if state_dict is not None:
            self.load_state_dict(state_dict)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_size = C // self.n_head

        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        # Only use flash attention for long sequences
        # enable_flash = self.enable_flash and (T > 128)
        if self.enable_flash:
            y = self.flash_attention(q, k, v)
        else:
            y = self.original_attention(q, k, v)
        if self.debug:
            y_inner = self.original_attention(q, k, v)
            diff = (y - y_inner).abs().max()
            print(f"flash: {self.enable_flash}, Inner Max diff: {diff.item()}")

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        if self.debug:
            y_orig = super().forward(x)
            diff = (y_orig - y).abs().max()
            print(f"flash: {self.enable_flash}, Inner super Max diff: {diff.item()}")
        return y

    def original_attention(self, q, k, v):
        # Original attention implementation for reference
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Apply causal mask to prevent attending to future tokens
        att = att.masked_fill(self.bias[None, None, :q.size(-2), :q.size(-2)] == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        return att @ v

    def flash_attention(self, q, k, v):
        B, H, T, D = q.shape
        dtype = q.dtype
        BLOCK = min(128, T)  # Adaptive block size

        # Initialize output and statistics
        O = torch.zeros_like(v, dtype=torch.float32)
        L = torch.zeros((B, H, T), dtype=torch.float32, device=q.device)
        M = torch.full((B, H, T), -1e9, dtype=torch.float32, device=q.device)

        num_blocks = (T + BLOCK - 1) // BLOCK

        for i in range(num_blocks):
            start_i = i * BLOCK
            end_i = min((i + 1) * BLOCK, T)
            block_i = end_i - start_i

            # Load query block
            Q_i = q[:, :, start_i:end_i].float()

            # Initialize local state
            O_i = torch.zeros((B, H, block_i, D),
                              dtype=torch.float32, device=q.device)
            L_i = torch.zeros(
                (B, H, block_i), dtype=torch.float32, device=q.device)
            M_i = torch.full((B, H, block_i), -1e9,
                             dtype=torch.float32, device=q.device)

            for j in range(i + 1):  # Only j <= i (causal)
                start_j = j * BLOCK
                end_j = min((j + 1) * BLOCK, T)
                block_j = end_j - start_j

                # Load key/value block
                K_j = k[:, :, start_j:end_j].float()
                V_j = v[:, :, start_j:end_j].float()

                # Compute attention scores
                S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) / math.sqrt(D)

                # Apply causal mask to diagonal blocks
                if j == i:
                    mask = torch.tril(torch.ones(
                        block_i, block_j,
                        dtype=torch.bool, device=S_ij.device
                    ))
                    S_ij = S_ij.masked_fill(~mask, -1e9)

                # Update row max and exp sums
                m_ij = S_ij.max(dim=-1, keepdim=True).values
                m_new = torch.maximum(M_i.unsqueeze(-1), m_ij)

                # Compute exp scores
                exp_scores = torch.exp(S_ij - m_new)

                # Update normalization statistics
                l_ij = exp_scores.sum(dim=-1, keepdim=True)
                l_new = torch.exp(M_i.unsqueeze(-1) - m_new) * \
                    L_i.unsqueeze(-1) + l_ij

                # Update output
                O_i = torch.exp(M_i.unsqueeze(-1) - m_new) * O_i + \
                    torch.matmul(exp_scores, V_j)

                # Update local state
                M_i = m_new.squeeze(-1)
                L_i = l_new.squeeze(-1)

            # Finalize block output
            O_i = O_i / L_i.unsqueeze(-1)

            # Apply dropout if needed
            if self.dropout > 0 and self.training:
                # For simplicity, use standard dropout on the output
                O_i = self.attn_dropout(O_i)

            O[:, :, start_i:end_i] = O_i.to(dtype)

        return O.to(dtype)


if __name__ == "__main__":
    # Verify Flash Attention implementation
    # torch.manual_seed(42)
    # config = {'n_embd': 768, 'n_head': 12, 'dropout': 0.1, 'qkv_bias': True}
    torch.manual_seed(42)
    state_dict = {
        "c_attn.weight": torch.randn(768*3, 768),
        "c_attn.bias": torch.randn(768*3),
        "c_proj.weight": torch.randn(768, 768),
        "c_proj.bias": torch.randn(768),
        "bias": torch.tril(torch.ones(1024, 1024)),
    }
    orig_attn = FlashOrOriginalAttention(
        config, state_dict=state_dict, enable_flash=False, debug=True)
    attn = FlashOrOriginalAttention(config, state_dict=state_dict, debug=True)
    x = torch.randn(2, 512, 768)  # test batch size of 2

    # Ensure dropout is disabled for consistent comparison
    attn.eval()
    orig_attn.eval()
    with torch.no_grad():
        y_orig = orig_attn(x)
        y_flash = attn(x)

        # Compute max difference between original and flash attention outputs
        diff = (y_orig - y_flash).abs().max()
        print(f"Max diff: {diff.item()}")
