import torch
import triton
import math
import triton.language as tl

# Import the local decode kernels we copied into llaisys
from llaisys.libllaisys.triton.kernels.scaled_dot_product_attention_decode import split_kv_kernel, combine_kv_splits_kernel

# Simple heuristic for S selection (kept same as hw2)
def get_optimal_s(batch_size: int, num_heads: int, seq_len: int, device: str = 'cuda') -> int:
    SEQ_LEN_THRESHOLD = 512
    if seq_len < SEQ_LEN_THRESHOLD:
        return 1
    if not torch.cuda.is_available():
        return 1
    properties = torch.cuda.get_device_properties(device)
    num_sms = properties.multi_processor_count
    total_tasks = batch_size * num_heads
    if total_tasks >= num_sms:
        return 1
    else:
        s_float = num_sms / total_tasks
        s_int = math.ceil(s_float)
        if s_int > 1:
            s_power_of_2 = 2**math.floor(math.log2(s_int))
            return min(4, s_power_of_2)
        else:
            return 1


def scaled_dot_product_attention_decode(q, k, v, past_k, past_v, scale=None):
    batch_size, num_heads, seq_len_q, emb_dim = q.shape
    _, num_heads_k, seq_len_k_v, _ = k.shape

    if scale is None:
        scale = 1 / math.sqrt(emb_dim)
    # past length and current length
    past_seq_len = past_k.shape[2]
    current_seq_len = k.shape[2]

    causal = True
    seq_len_k_v += past_seq_len
    # Align head counts: if query heads != key/value heads, repeat k/v and past_k/past_v
    if num_heads != num_heads_k:
        if num_heads % num_heads_k != 0:
            raise RuntimeError(f"Incompatible head counts: q heads={num_heads}, k heads={num_heads_k}")
        repeat = num_heads // num_heads_k
        # repeat along head dimension (dim=1)
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        if past_k is not None:
            past_k = past_k.repeat_interleave(repeat, dim=1)
        if past_v is not None:
            past_v = past_v.repeat_interleave(repeat, dim=1)
        # update num_heads_k now that we've repeated
        _, num_heads_k, _, _ = k.shape
    num_groups = num_heads // num_heads_k

    if scale is None:
        scale = 1. / math.sqrt(emb_dim)

    S = get_optimal_s(batch_size, num_heads, seq_len_k_v, q.device)

    split_logsumexp = torch.empty((batch_size, num_heads, S, seq_len_q), dtype=torch.float32, device="cuda")
    split_outputs = torch.empty((batch_size, num_heads, S, seq_len_q, emb_dim), dtype=torch.float16, device="cuda")

    def grid1(meta):
        return (triton.cdiv(seq_len_q, meta['BLOCK_SIZE_M']), S, num_heads * batch_size)

    M_binned = triton.next_power_of_2(seq_len_q)
    N_binned = triton.next_power_of_2(seq_len_k_v)

    split_kv_kernel[grid1](
        q, k, v, past_k, past_v,
        scale,
        split_logsumexp, split_outputs,
        num_heads, num_groups,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        past_k.stride(0), past_k.stride(1), past_k.stride(2), past_k.stride(3),
        past_v.stride(0), past_v.stride(1), past_v.stride(2), past_v.stride(3),
        split_outputs.stride(0), split_outputs.stride(1), split_outputs.stride(2), split_outputs.stride(3), split_outputs.stride(4),
        seq_len_q,
        seq_len_k_v,
        current_seq_len,
        S,
        EMB_DIM=emb_dim,
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64,
        M_BINNED=M_binned, N_BINNED=N_binned,
        IS_CAUSAL=causal,
    )

    if S == 1:
        return split_outputs.squeeze(2)

    final_l = torch.empty((batch_size, num_heads, seq_len_q), dtype=torch.float32, device="cuda")
    final_o = torch.empty_like(q)

    def grid2(meta):
        return (triton.cdiv(seq_len_q, meta['BLOCK_SIZE_M']), num_heads, batch_size)

    combine_kv_splits_kernel[grid2](
        split_outputs, split_logsumexp,
        final_o, final_l,
        num_heads,
        split_outputs.stride(0), split_outputs.stride(1), split_outputs.stride(2), split_outputs.stride(3), split_outputs.stride(4),
        final_o.stride(0), final_o.stride(1), final_o.stride(2), final_o.stride(3),
        seq_len_q, S,
        EMB_DIM=emb_dim,
        BLOCK_SIZE_M=64,
        M_BINNED=M_binned,
    )
    return final_o
