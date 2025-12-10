import torch
import math

def scaled_dot_product_attention_prefill(query, key, value, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    exp_weights = torch.exp(attn_weight)
    sum_exp_weights = exp_weights.sum(dim=-1, keepdim=True)
    softmax_weights = exp_weights / sum_exp_weights
    output = softmax_weights @ value
    return output


def scaled_dot_product_attention_decode(query, key, value, past_key, past_value, scale=None) -> torch.Tensor:
    # query: (batch, heads, L, dim)
    # key/value: (batch, heads_k, S_cur, dim)
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    if past_key is not None and past_value is not None:
        key = torch.cat([past_key, key], dim=-2)
        value = torch.cat([past_value, value], dim=-2)
    # Ensure key/value have same head count as query by repeating if necessary
    # query: (B, Hq, L, D), key: (B, Hk, S, D)
    B, Hq, _, D = query.shape
    _, Hk, S_total, _ = key.shape
    if Hq != Hk:
        repeat = Hq // Hk
        if repeat > 1:
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)

    # compute attention weights
    # attn_weight shape: (B, H, L, S)
    attn_weight = query @ key.transpose(-2, -1) * scale_factor

    # causal mask: match test/ops/self_attention.py logic
    # create mask of shape (L, S) with tril(diagonal = S - L)
    S_eff = key.size(-2)
    mask = torch.ones(L, S_eff, dtype=torch.bool, device=query.device).tril(diagonal=S_eff - L)
    # broadcast mask across batch and heads
    # positions not allowed are filled with -inf
    attn_bias = torch.zeros_like(attn_weight, dtype=attn_weight.dtype, device=attn_weight.device)
    attn_bias.masked_fill_(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    attn_weight = attn_weight + attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    output = attn_weight @ value
    return output
