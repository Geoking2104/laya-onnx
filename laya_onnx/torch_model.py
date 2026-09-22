"""Tiny ModernBERT-style DecisionModel used for export and parity tests."""

from __future__ import annotations

import math

import torch
from torch import nn


class EncoderConfig:
    def __init__(self, data: dict):
        self.hidden_size = int(data["hidden_size"])
        self.intermediate_size = int(data.get("intermediate_size", self.hidden_size * 4))
        self.vocab_size = int(data["vocab_size"])
        self.num_hidden_layers = int(data["num_hidden_layers"])
        self.num_attention_heads = int(data["num_attention_heads"])
        self.local_attention = int(data.get("local_attention", 128))
        self.global_attn_every_n_layers = int(data.get("global_attn_every_n_layers", 3))
        self.max_position_embeddings = int(data.get("max_position_embeddings", 512))
        self.layer_norm_eps = float(data.get("layer_norm_eps", 1e-5))

    @classmethod
    def from_dict(cls, data):
        return cls(data)


class EncoderLayer(nn.Module):
    def __init__(self, cfg: EncoderConfig, global_attn: bool):
        super().__init__()
        self.global_attn = global_attn
        self.local_attention = cfg.local_attention
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.in_proj = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=True)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        self.norm1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.norm2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.ff = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size),
            nn.GELU(),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size),
        )

    def forward(self, hidden, attention_mask):
        b, t, c = hidden.shape
        x = self.norm1(hidden)
        qkv = self.in_proj(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        pad = attention_mask.eq(0).unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(pad, torch.finfo(scores.dtype).min)
        if not self.global_attn:
            idx = torch.arange(t, device=hidden.device)
            dist = (idx[None, :] - idx[:, None]).abs()
            local = dist.gt(self.local_attention // 2).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(local, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, c)
        hidden = hidden + self.out_proj(ctx)
        hidden = hidden + self.ff(self.norm2(hidden))
        return hidden


class Encoder(nn.Module):
    def __init__(self, cfg: EncoderConfig, max_len: int):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos = nn.Embedding(max(max_len, cfg.max_position_embeddings), cfg.hidden_size)
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        layers = []
        for i in range(cfg.num_hidden_layers):
            global_attn = ((i + 1) % cfg.global_attn_every_n_layers) == 0 or i == 0
            layers.append(EncoderLayer(cfg, global_attn))
        self.layers = nn.ModuleList(layers)

    def forward(self, input_ids, attention_mask):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.embed(input_ids) + self.pos(positions)[None, :, :]
        hidden = self.norm(hidden)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
        return hidden


class DecisionModel(nn.Module):
    def __init__(self, encoder_cfg: dict, agent_cfg: dict, max_len: int = 512):
        super().__init__()
        if isinstance(encoder_cfg, EncoderConfig):
            enc = encoder_cfg
        else:
            enc = EncoderConfig(encoder_cfg)
        self.max_len = int(agent_cfg.get("max_len", max_len))
        self.encoder = Encoder(enc, self.max_len)
        hidden = enc.hidden_size
        layers = int(agent_cfg.get("head_layers", 2))
        blocks = []
        for _ in range(max(1, layers)):
            blocks.extend([nn.Linear(hidden, hidden), nn.GELU()])
        self.option_mlp = nn.Sequential(*blocks)
        self.option_out = nn.Linear(hidden, 1)
        self.act_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.qtype_embed = nn.Embedding(4, hidden)

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        hidden = self.encoder(input_ids.long(), attention_mask)
        gather_index = marker_pos.long().clamp(min=0, max=hidden.shape[1] - 1)
        gather_index = gather_index.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        option_h = hidden.gather(1, gather_index)
        option_h = option_h + self.qtype_embed(qtype.long().clamp(0, 3)).unsqueeze(1)
        logits = self.option_out(self.option_mlp(option_h)).squeeze(-1)
        invalid = marker_mask.eq(0) if marker_mask.dtype != torch.bool else ~marker_mask.bool()
        logits = logits.masked_fill(invalid, -1e4)
        pooled = (option_h * (~invalid).unsqueeze(-1).to(option_h.dtype)).sum(1)
        denom = (~invalid).sum(1).clamp(min=1).unsqueeze(-1).to(option_h.dtype)
        act = self.act_mlp(pooled / denom)
        return logits, act
