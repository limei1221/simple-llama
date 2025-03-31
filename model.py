import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import os
import json
import pydantic


class LlamaConfig(pydantic.BaseModel):
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    max_position_embeddings: int = 2048
    pad_token_id: int = 0
    rms_norm_eps: float = 1e-6
    num_key_value_heads: Optional[int] = (
        None  # Will default to num_attention_heads ** 0.5 if not set
    )

    def to_dict(self):
        """Convert the config to a dictionary."""
        return self.model_dump()


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(
            torch.float32
        )  # torch.Size([batch_size, seq_length, hidden_size])
        variance = hidden_states.pow(2).mean(
            -1, keepdim=True
        )  # torch.Size([batch_size, seq_length, 1])
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, device=None):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # emb = torch.cat((freqs, freqs), dim=-1)
        emb = freqs.repeat_interleave(2, dim=1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_pairs(x):
    # Save original shape and flatten all dimensions except last
    orig_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1])

    # Reshape last dim into pairs
    pairs = x_flat.view(x_flat.shape[0], -1, 2)

    # Create output tensor and perform rotation
    rearranged = torch.empty_like(pairs)
    rearranged[..., 0] = -pairs[..., 1]
    rearranged[..., 1] = pairs[..., 0]

    # Restore original shape
    return rearranged.reshape(orig_shape)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_pairs(q) * sin)
    k_embed = (k * cos) + (rotate_pairs(k) * sin)
    return q_embed, k_embed


class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = (
            config.num_key_value_heads
            if config.num_key_value_heads is not None
            else int(self.num_heads**0.5)
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim, config.max_position_embeddings
        )

        self.k_cache = None
        self.v_cache = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        is_prefix: bool = False,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        # query_states.shape: torch.Size([batch_size, num_heads, seq_length, head_dim])
        # key_states.shape: torch.Size([batch_size, num_key_value_heads, seq_length, head_dim])
        # value_states.shape: torch.Size([batch_size, num_key_value_heads, seq_length, head_dim])

        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        # query_states.shape: torch.Size([batch_size, num_heads, seq_length, head_dim])
        # key_states.shape: torch.Size([batch_size, num_heads, seq_length, head_dim])
        # value_states.shape: torch.Size([batch_size, num_heads, seq_length, head_dim])

        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        # cos.shape: torch.Size([1, 1, seq_length, head_dim])
        # sin.shape: torch.Size([1, 1, seq_length, head_dim])
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        # query_states.shape: torch.Size([batch_size, num_heads, seq_length, head_dim])
        # key_states.shape: torch.Size([batch_size, num_heads, seq_length, head_dim])

        if use_cache:
            if is_prefix:
                self.k_cache = key_states
                self.v_cache = value_states
            else:
                key_states = torch.cat([self.k_cache, key_states], dim=2)
                value_states = torch.cat([self.v_cache, value_states], dim=2)
                self.k_cache = key_states
                self.v_cache = value_states

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        )

        if (not use_cache) or is_prefix:
            # Apply causal mask only during prefill
            seq_length = q_len
            causal_mask = torch.triu(
                torch.ones(
                    (seq_length, seq_length),
                    dtype=torch.bool,
                    device=attn_weights.device,
                ),
                diagonal=1,
            )
            # Convert to float and replace True with -inf
            causal_mask = causal_mask.float().masked_fill(
                causal_mask == 1, float("-inf")
            )

            # Add causal mask to attention weights
            attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            # Adjust attention mask to match the sequence length of attention weights
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)

            attention_mask = attention_mask.expand(-1, self.num_heads, -1, -1)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        is_prefix: bool = False,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            is_prefix=is_prefix,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        is_prefix: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
                is_prefix=is_prefix,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    def __init__(self, config, tokenizer=None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Share weights between embed_tokens and lm_head
        # self.lm_head.weight = self.model.embed_tokens.weight

        self.loss_fct = nn.CrossEntropyLoss()

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_built() else "cpu"
        )
        self.device = torch.device(device)
        print("Using device:", self.device)
        # Move model to specified device
        self.to(self.device)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        is_prefix: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            is_prefix=is_prefix,
        )
        logits = self.lm_head(hidden_states)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Add log_softmax before computing cross entropy
            loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        num_return_sequences: int = 1,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.LongTensor:
        """
        Generate text using the model with temperature and top-k sampling.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            max_length: Maximum length of the generated sequence
            temperature: Temperature for sampling (higher = more random)
            top_k: Number of highest probability tokens to consider for sampling
            top_p: Cumulative probability threshold for nucleus sampling
            do_sample: Whether to use sampling or greedy decoding
            pad_token_id: ID of the padding token
            eos_token_id: ID of the end-of-sequence token
            num_return_sequences: Number of sequences to generate per prompt
            attention_mask: Optional attention mask

        Returns:
            Generated token IDs of shape (batch_size * num_return_sequences, max_length)
        """
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = (
                self.config.eos_token_id
                if hasattr(self.config, "eos_token_id")
                else pad_token_id
            )

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Handle multiple sequences per prompt
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
            if attention_mask is not None:
                attention_mask = attention_mask.repeat(num_return_sequences, 1)

        # Initialize the generated sequence with input_ids
        generated = input_ids.clone()

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        # Clear KV cache in all attention layers
        for layer in self.model.layers:
            layer.self_attn.k_cache = None
            layer.self_attn.v_cache = None

        # Prefill phase - process the entire prompt
        outputs = self(
            input_ids=generated,
            attention_mask=attention_mask,
            use_cache=True,
            is_prefix=True,
        )

        # Add the attention mask to the generated sequence
        attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (batch_size * num_return_sequences, 1),
                        dtype=torch.bool,
                        device=device,
                    ),
                ],
                dim=1,
            )

        for _ in range(max_length - input_ids.shape[1]):
            # Get next token logits
            next_token_logits = outputs["logits"][:, -1, :] / temperature

            if do_sample:
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = (
                        next_token_logits
                        < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Process single new token with cached KV
            outputs = self(
                input_ids=next_token,
                attention_mask=attention_mask,
                use_cache=True,
                is_prefix=False,
            )

            # Append the next token to the generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Update attention mask
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (batch_size * num_return_sequences, 1),
                        dtype=torch.bool,
                        device=device,
                    ),
                ],
                dim=1,
            )

            # Check if all sequences have reached EOS token
            if (generated == eos_token_id).any(dim=1).all():
                break

        return generated
