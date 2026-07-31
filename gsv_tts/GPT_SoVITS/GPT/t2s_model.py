from typing import List

import torch
import logging
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn.attention import sdpa_kernel
from ...Config import SDPBACKEND
from tqdm import tqdm

from .utils import sample
from .embedding import SinePositionalEmbedding, TokenEmbedding
from .t2s_base import (
    _T2SBlockBase, _T2STransformerBase, _BucketBase, Text2SemanticDecoderBase,
)


class T2SBlock(_T2SBlockBase):

    def process_prompt(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_mask: torch.Tensor
    ):
        B, L, _ = x.shape

        residual = x
        
        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k_cache[:, :, :L] = k
        v_cache[:, :, :L] = v

        with sdpa_kernel(SDPBACKEND):
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        x = x.transpose(1, 2).reshape(B, L, self.hidden_dim)
        x = self.out_proj(x)
        
        x = residual + x
        x = self.norm1(x)
        
        residual = x
        x = self.mlp(x)
        x = residual + x
        x = self.norm2(x)
        
        return x

    def decode_next_token(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_mask: torch.Tensor,
        kv_cache_len: torch.Tensor,
        batch_indices: torch.Tensor
    ):
        B, L, _ = x.shape

        residual = x

        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k_cache[batch_indices, :, kv_cache_len] = k.squeeze(2)
        v_cache[batch_indices, :, kv_cache_len] = v.squeeze(2)

        # kv_cache shape [batch_size, num_heads, kv_len, head_dim/num_heads]

        with sdpa_kernel(SDPBACKEND):
            x = F.scaled_dot_product_attention(q, k_cache, v_cache, attn_mask=attn_mask)
        
        x = x.transpose(1, 2).reshape(B, L, self.hidden_dim)
        x = self.out_proj(x)
        
        x = residual + x
        x = self.norm1(x)
        
        residual = x
        x = self.mlp(x)
        x = residual + x
        x = self.norm2(x)
        
        return x


class T2STransformer(_T2STransformerBase):

    def decode_next_token(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        kv_cache_len: torch.Tensor,
        attn_mask: torch.Tensor,
        batch_indices: torch.Tensor
    ):
        for i in range(self.num_blocks):
            x = self.blocks[i].decode_next_token(
                x, k_cache[i], v_cache[i], attn_mask, kv_cache_len, batch_indices
            )
        kv_cache_len += 1
        return x


class Bucket(_BucketBase):
    decode_attn_mask: torch.Tensor = None
    batch_indices: int = None

class Text2SemanticDecoder(Text2SemanticDecoderBase):
    """Standard SDPA implementation of GPT T2S decoder."""

    def _build_t2s_block(self, hidden_dim, num_heads):
        return T2SBlock(hidden_dim, num_heads)

    def __init__(self, config):
        super().__init__(config)
        # Re-wrap blocks with the correct T2STransformer subtype (base used
        # _T2STransformerBase; we need T2STransformer's decode_next_token).
        self.t2s_transformer = T2STransformer(
            self.num_layers, list(self.t2s_transformer.blocks))

    # Public aliases for methods called by inference code
    def process_batch_data(self, x, y, bert_feature, x_lens, y_lens):
        return self._process_batch_data(x, y, bert_feature, x_lens, y_lens)
    
    @torch.inference_mode()
    def initialize_runtime(self, dtype, device, gpt_cache):
        self.ar_text_position.extend_pe(torch.tensor(0.0, dtype=dtype, device=device).expand(1, 4000))
        self.ar_audio_position.extend_pe(torch.tensor(0.0, dtype=dtype, device=device).expand(1, 4000))

        for batch_size, max_kv_cache in gpt_cache:
            if batch_size in self.cuda_graph_buckets:
                for i, _max_kv_cache in enumerate(self.cuda_graph_buckets[batch_size]):
                    if _max_kv_cache > max_kv_cache:
                        self.cuda_graph_buckets[batch_size].insert(i, max_kv_cache)
                        break
                else:
                    self.cuda_graph_buckets[batch_size].append(max_kv_cache)
            else:
                self.cuda_graph_buckets[batch_size] = [max_kv_cache]

        # 这里采用了一种自适应的 KV Cache 管理策略

        # 检查是否使用 CUDA Graph（仅 CUDA 设备支持）
        use_cuda_graph = device.type == "cuda"

        if use_cuda_graph:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            stream_context = torch.cuda.stream(s)
        else:
            # 非 CUDA 设备使用默认流
            stream_context = torch.no_grad()

        with stream_context:
            max_btz = max(self.cuda_graph_buckets.keys())
            max_elem = max(btz * seqlen 
                                for btz in self.cuda_graph_buckets 
                                    for seqlen in self.cuda_graph_buckets[btz])
            max_numel = self.num_layers * max_elem * self.model_dim
            k_cache_root = torch.empty(max_numel, dtype=dtype, device=device)
            v_cache_root = torch.empty(max_numel, dtype=dtype, device=device)
            decode_attn_mask_root = torch.zeros(max_elem * self.num_head, dtype=torch.bool, device=device)

            KV_CACHE_LEN = torch.zeros((max_btz,), dtype=torch.int64, device=device)
            GRAPH_XY_POS = torch.zeros((max_btz, 1, self.embedding_dim), dtype=dtype, device=device)

            for batch_size in self.cuda_graph_buckets:
                BATCH_IDX = torch.arange(batch_size, dtype=torch.int64, device=device)
                for i in range(-1, -len(self.cuda_graph_buckets[batch_size])-1, -1):
                    max_kv_cache = self.cuda_graph_buckets[batch_size][i]

                    bucket = Bucket()

                    bucket.max_kv_cache = max_kv_cache
                    bucket.batch_size = batch_size

                    bucket.kv_cache_len = KV_CACHE_LEN[:batch_size]
                    bucket.graph_xy_pos = GRAPH_XY_POS[:batch_size]
                    bucket.batch_indices = BATCH_IDX

                    if i == -1:
                        numel_kv_cache = self.num_layers * batch_size * max_kv_cache * self.model_dim
                        numel_decode_attn_mask = batch_size * self.num_head * max_kv_cache
                        bucket.k_cache = k_cache_root[:numel_kv_cache].view(self.num_layers, batch_size, self.num_head, max_kv_cache, int(self.model_dim/self.num_head))
                        bucket.v_cache = v_cache_root[:numel_kv_cache].view(self.num_layers, batch_size, self.num_head, max_kv_cache, int(self.model_dim/self.num_head))
                        bucket.decode_attn_mask = decode_attn_mask_root[:numel_decode_attn_mask].view(batch_size, self.num_head, 1, max_kv_cache)
                    else:
                        max_bucket: Bucket = self.cuda_graph_buckets[batch_size][-1]
                        bucket.k_cache = max_bucket.k_cache[:, :, :, :max_kv_cache]
                        bucket.v_cache = max_bucket.v_cache[:, :, :, :max_kv_cache]
                        bucket.decode_attn_mask = max_bucket.decode_attn_mask[:, :, :, :max_kv_cache]

                    # 预热运行
                    for _ in range(3):
                        self.t2s_transformer.decode_next_token(
                            bucket.graph_xy_pos, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len, bucket.decode_attn_mask, bucket.batch_indices
                        )

                    bucket.kv_cache_len.fill_(0)

                    if use_cuda_graph:
                        torch.cuda.current_stream().synchronize()

                        bucket.cuda_graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(bucket.cuda_graph):
                            bucket.graph_xy_dec = self.t2s_transformer.decode_next_token(
                                bucket.graph_xy_pos, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len, bucket.decode_attn_mask, bucket.batch_indices
                            )

                    self.cuda_graph_buckets[batch_size][i] = bucket

        if use_cuda_graph:
            torch.cuda.current_stream().wait_stream(s)
    
    @torch.inference_mode()
    def infer(
        self,
        x: torch.LongTensor,
        y: torch.LongTensor,
        bert_feature: torch.LongTensor,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        initial_suppression_steps: int = 10,
        check_interval: int = 5,
    ):
        xy_pos, prompt_attn_mask = self.process_single_data(x, y, bert_feature)

        buckets = self.cuda_graph_buckets[1] # B = 1
        for bucket_i in range(len(buckets)):
            if buckets[bucket_i].max_kv_cache > xy_pos.shape[1]:
                break
        bucket: Bucket = buckets[bucket_i]
        max_bucket: Bucket = buckets[-1]

        max_bucket.kv_cache_len.fill_(0)

        pe_cache = self.ar_audio_position.alpha * self.ar_audio_position.pe
        pe_cache = pe_cache.transpose(0, 1)

        pre_tokens = y

        xy_dec = self.t2s_transformer.process_prompt(xy_pos, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len, prompt_attn_mask)
        logits = self.ar_predict_layer(xy_dec[:, -1])
        logits[:, self.suppressed_tokens] = -float("Inf")
        samples = sample(logits[:, :-1], pre_tokens, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0]
        pre_tokens = torch.concat([pre_tokens, samples], dim=1)
        y_emb = self.ar_audio_embedding(samples)
        pe_idx = min(max(0, bucket.kv_cache_len - x.shape[1]), pe_cache.shape[0] - 1)
        xy_pos = y_emb * self.ar_audio_position.x_scale + pe_cache[pe_idx]

        max_bucket.decode_attn_mask.fill_(False)
        bucket.decode_attn_mask[:, :, :, :bucket.kv_cache_len] = True
        
        for idx in tqdm(range(1, max_bucket.max_kv_cache - bucket.kv_cache_len + 1)):
            if bucket.kv_cache_len == bucket.max_kv_cache:
                bucket_i += 1
                if bucket_i >= len(buckets):
                    logging.warning(
                        f"Generated sequence exceeds max cache size "
                        f"({bucket.max_kv_cache}). Truncating."
                    )
                    break
                bucket: Bucket = buckets[bucket_i]

            bucket.decode_attn_mask[:, :, :, bucket.kv_cache_len] = True

            # 使用 CUDA Graph（如果可用）或普通执行
            if bucket.cuda_graph is not None:
                bucket.graph_xy_pos.copy_(xy_pos)
                bucket.cuda_graph.replay()
                xy_dec = bucket.graph_xy_dec
            else:
                xy_dec = self.t2s_transformer.decode_next_token(
                    xy_pos, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len, bucket.decode_attn_mask, bucket.batch_indices
                )

            logits = self.ar_predict_layer(xy_dec[:, -1])

            if idx < initial_suppression_steps:
                logits[:, self.suppressed_tokens] = -float("Inf")

            samples = sample(logits, pre_tokens, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0]
            
            pre_tokens = torch.concat([pre_tokens, samples], dim=1)

            if idx % check_interval == 0:
                if samples[0, 0] == self.EOS:
                    break

            y_emb = self.ar_audio_embedding(samples)
            pe_idx = min(max(0, bucket.kv_cache_len - x.shape[1]), pe_cache.shape[0] - 1)
        xy_pos = y_emb * self.ar_audio_position.x_scale + pe_cache[pe_idx]

        pre_tokens = pre_tokens[:, -idx:]
        eos_indices = (pre_tokens == self.EOS).nonzero(as_tuple=True)[1]
        if eos_indices.numel() > 0:
            first_eos_idx = eos_indices[0].item()
            return pre_tokens[:, :first_eos_idx].unsqueeze(0)
        else:
            return pre_tokens.unsqueeze(0)
        
    @torch.inference_mode()
    def infer_stream(
        self,
        x: torch.LongTensor,
        y: torch.LongTensor,
        bert_feature: torch.LongTensor,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        initial_suppression_steps: int = 10,
        stream_chunk: int = 25,
        boost_first_chunk: bool = True,
        debug: bool = True,
    ):
        xy_pos, prompt_attn_mask = self.process_single_data(x, y, bert_feature)

        buckets = self.cuda_graph_buckets[1] # B = 1
        for bucket_i in range(len(buckets)):
            if buckets[bucket_i].max_kv_cache > xy_pos.shape[1]:
                break
        bucket: Bucket = buckets[bucket_i]
        max_bucket: Bucket = buckets[-1]

        max_bucket.kv_cache_len.fill_(0)

        pe_cache = self.ar_audio_position.alpha * self.ar_audio_position.pe
        pe_cache = pe_cache.transpose(0, 1)

        pre_tokens = y

        xy_dec = self.t2s_transformer.process_prompt(xy_pos, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len, prompt_attn_mask)
        logits = self.ar_predict_layer(xy_dec[:, -1])
        logits[:, self.suppressed_tokens] = -float("Inf")
        samples = sample(logits[:, :-1], pre_tokens, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0]
        pre_tokens = torch.concat([pre_tokens, samples], dim=1)
        y_emb = self.ar_audio_embedding(samples)
        pe_idx = min(max(0, bucket.kv_cache_len - x.shape[1]), pe_cache.shape[0] - 1)
        xy_pos = y_emb * self.ar_audio_position.x_scale + pe_cache[pe_idx]

        max_bucket.decode_attn_mask.fill_(False)
        bucket.decode_attn_mask[:, :, :, :bucket.kv_cache_len] = True
        
        first_chunk = True
        pre_chunk = None
        for idx in tqdm(range(1, max_bucket.max_kv_cache - bucket.kv_cache_len + 1), disable=not debug):
            if bucket.kv_cache_len == bucket.max_kv_cache:
                bucket_i += 1
                if bucket_i >= len(buckets):
                    logging.warning(
                        f"Generated sequence exceeds max cache size "
                        f"({bucket.max_kv_cache}). Truncating."
                    )
                    break
                bucket: Bucket = buckets[bucket_i]
            
            bucket.decode_attn_mask[:, :, :, bucket.kv_cache_len] = True

            # 使用 CUDA Graph（如果可用）或普通执行
            if bucket.cuda_graph is not None:
                bucket.graph_xy_pos.copy_(xy_pos)
                bucket.cuda_graph.replay()
                xy_dec = bucket.graph_xy_dec
            else:
                xy_dec = self.t2s_transformer.decode_next_token(
                    xy_pos, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len, bucket.decode_attn_mask, bucket.batch_indices
                )

            logits = self.ar_predict_layer(xy_dec[:, -1])

            if idx < initial_suppression_steps:
                logits[:, self.suppressed_tokens] = -float("Inf")

            samples = sample(logits, pre_tokens, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0]
            
            if samples[0, 0] == self.EOS:
                break

            pre_tokens = torch.concat([pre_tokens, samples], dim=1)

            if idx % stream_chunk == 0:
                if not pre_chunk is None:
                    yield pre_chunk, False
                pre_chunk = pre_tokens[:, -idx:].unsqueeze(0)

                if boost_first_chunk:
                    if first_chunk:
                        first_chunk = False
                        yield pre_chunk, False
                        pre_chunk = None

            y_emb = self.ar_audio_embedding(samples)
            pe_idx = min(max(0, bucket.kv_cache_len - x.shape[1]), pe_cache.shape[0] - 1)
        xy_pos = y_emb * self.ar_audio_position.x_scale + pe_cache[pe_idx]

        yield pre_tokens[:, -idx:].unsqueeze(0), True
    
    @torch.inference_mode()
    def infer_batched(
        self,
        x: List[torch.LongTensor],
        y: List[torch.LongTensor],
        bert_feature: List[torch.LongTensor],
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        check_interval: int = 5,
    ):
        B, device = len(x), x[0].device
        
        for batch_size in sorted(self.cuda_graph_buckets):
            if batch_size >= B:
                break

        batch_indices = torch.arange(batch_size, dtype=torch.int64, device=device)
        actual_batch_size = min(B, batch_size)
        
        batch_x = pad_sequence(x[:batch_size], batch_first=True, padding_value=0)
        batch_y = pad_sequence(y[:batch_size], batch_first=True, padding_value=0)
        batch_bert_feature = pad_sequence(bert_feature[:batch_size], batch_first=True, padding_value=0)
        x_lens = torch.tensor([i.shape[0] for i in x[:batch_size]], device=device)
        y_lens = torch.tensor([i.shape[0] for i in y[:batch_size]], device=device)
        xy_lens = x_lens + y_lens

        xy_pos, last_token_mask, prompt_attn_mask = self.process_batch_data(
            batch_x,
            batch_y,
            batch_bert_feature,
            x_lens.unsqueeze(1),
            y_lens.unsqueeze(1),
        )

        buckets = self.cuda_graph_buckets[batch_size]
        for bucket_i in range(len(buckets)):
            if buckets[bucket_i].max_kv_cache > xy_pos.shape[1]:
                break
        bucket: Bucket = buckets[bucket_i]
        max_bucket: Bucket = buckets[-1]
            
        max_bucket.kv_cache_len.fill_(0)

        current_batch = actual_batch_size

        pe_cache = self.ar_audio_position.alpha * self.ar_audio_position.pe
        pe_cache = pe_cache.transpose(0, 1)

        pre_tokens = torch.zeros((batch_size, max_bucket.max_kv_cache), dtype=torch.int64, device=device)

        # prefill
        xy_dec = self.t2s_transformer.process_prompt(xy_pos, bucket.k_cache[:, :actual_batch_size], bucket.v_cache[:, :actual_batch_size], bucket.kv_cache_len[:actual_batch_size], prompt_attn_mask)
        logits = self.ar_predict_layer(xy_dec[last_token_mask])

        bucket.kv_cache_len[:actual_batch_size].copy_(xy_lens)

        samples = sample(logits[:, :-1], top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0]
        pre_tokens[batch_indices, bucket.kv_cache_len][:actual_batch_size] = samples.squeeze()
        y_emb = self.ar_audio_embedding(samples)
        xy_pos = y_emb * self.ar_audio_position.x_scale + pe_cache[bucket.kv_cache_len[:actual_batch_size]-x_lens]
        xy_pos = F.pad(xy_pos, (0, 0, 0, 0, 0, batch_size - actual_batch_size))
        x_lens = F.pad(x_lens, (0, batch_size - actual_batch_size))


        max_bucket.decode_attn_mask.fill_(False)
        indices = torch.arange(bucket.max_kv_cache, device=device)
        mask = indices[None, :] < bucket.kv_cache_len[:, None]
        bucket.decode_attn_mask.copy_(mask.view(batch_size, 1, 1, bucket.max_kv_cache))

        stop = False
        pred_semantic = []
        semantic_orig_idx = []
        batch_orig_idx = torch.linspace(0, batch_size-1, batch_size, dtype=torch.int64, device=device)
        decode_steps = torch.zeros(batch_size, dtype=torch.int64, device=device)
        ignore_batch = torch.ones(batch_size, dtype=torch.bool, device=device)
        ignore_batch[:actual_batch_size] = False
        while True:
            for idx in tqdm(range(1000)):
                decode_steps += 1

                bucket.decode_attn_mask[batch_indices, :, :, bucket.kv_cache_len] = True

                # 使用 CUDA Graph（如果可用）或普通执行
                if bucket.cuda_graph is not None:
                    bucket.graph_xy_pos.copy_(xy_pos)
                    bucket.cuda_graph.replay()
                    xy_dec = bucket.graph_xy_dec
                else:
                    xy_dec = self.t2s_transformer.decode_next_token(
                        xy_pos, bucket.k_cache, bucket.v_cache, bucket.kv_cache_len, bucket.decode_attn_mask, bucket.batch_indices
                    )

                logits = self.ar_predict_layer(xy_dec[:, -1])

                samples = sample(logits, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0] # 在想出更好的方案之前，暂时取消repetition_penalty
                
                pre_tokens[batch_indices, bucket.kv_cache_len] = samples.squeeze()

                if idx % check_interval == 0:
                    is_reached = bucket.kv_cache_len + check_interval >= bucket.max_kv_cache
                    is_eos_generated = samples[:, 0] == self.EOS
                    should_stop_seq = is_eos_generated | is_reached
                    finished = ~ignore_batch & should_stop_seq

                    if finished.any():
                        if is_reached.any():
                            bucket_i += 1
                            if bucket_i < len(buckets):
                                is_reached.fill_(False)
                                bucket = buckets[bucket_i]

                        should_stop_seq = is_eos_generated | is_reached
                        finished = ~ignore_batch & should_stop_seq

                        if finished.any():
                            finished_indices = finished.nonzero(as_tuple=True)[0]
                            for i in finished_indices:
                                generated_segment = pre_tokens[i, bucket.kv_cache_len[i]-decode_steps[i]+1 : bucket.kv_cache_len[i]]
                                eos_indices = (generated_segment == self.EOS).nonzero(as_tuple=True)[0]
                                if eos_indices.numel() > 0:
                                    first_eos_idx = eos_indices[0].item()
                                    generated_segment = generated_segment[:first_eos_idx]
                                pred_semantic.append(generated_segment.clone())
                                
                                semantic_orig_idx.append(batch_orig_idx[i].clone())
                                decode_steps[i] = 0

                                bucket.kv_cache_len[i].fill_(0)
                                max_kv_cache_len = bucket.kv_cache_len.max()
                                for bucket_i in range(len(buckets)):
                                    if buckets[bucket_i].max_kv_cache >= max_kv_cache_len + check_interval:
                                        break
                                bucket: Bucket = buckets[bucket_i]
                                
                                if current_batch == B:
                                    ignore_batch[i] = True
                                    if ignore_batch.all():
                                        stop = True
                                        break
                                else:
                                    single_x = x[current_batch]
                                    single_y = y[current_batch]
                                    single_bert_feature = bert_feature[current_batch]

                                    _xy_pos, prompt_attn_mask = self.process_single_data(
                                        single_x.unsqueeze(0),
                                        single_y.unsqueeze(0),
                                        single_bert_feature.unsqueeze(0),
                                    )

                                    xy_dec = self.t2s_transformer.process_prompt(_xy_pos, bucket.k_cache[:, i:i+1], bucket.v_cache[:, i:i+1], bucket.kv_cache_len[i:i+1], prompt_attn_mask)
                                    logits = self.ar_predict_layer(xy_dec[:, -1])

                                    x_lens[i].copy_(single_x.shape[0])
                                    bucket.kv_cache_len[i].copy_(single_x.shape[0] + single_y.shape[0])

                                    new_samples = sample(logits[:, :-1], top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0]
                                    samples[i:i+1] = new_samples

                                    max_bucket.decode_attn_mask[i:i+1].fill_(False)
                                    indices = torch.arange(bucket.max_kv_cache, device=device)
                                    mask = indices[None, :] < bucket.kv_cache_len[i:i+1, None]
                                    bucket.decode_attn_mask[i:i+1].copy_(mask.view(1, 1, 1, bucket.max_kv_cache))

                                    batch_orig_idx[i] = current_batch
                                    current_batch += 1
                            
                            if stop:
                                break

                y_emb = self.ar_audio_embedding(samples)
                xy_pos = y_emb * self.ar_audio_position.x_scale + pe_cache[bucket.kv_cache_len-x_lens]

            if stop:
                break

        semantic_orig_idx = torch.tensor(semantic_orig_idx, device=device)
        return pred_semantic, semantic_orig_idx
