"""Scratch byte causal LM preparation; contains no pretrained model or fetcher."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

PAD, BOS, EOS, VOCAB_SIZE = 0, 1, 2, 259
LANGUAGES = ("ko", "ja", "en", "es")


@dataclass(frozen=True)
class ModelConfig:
    id: str = "scratch-byte-384x8-v1"
    vocab_size: int = VOCAB_SIZE
    hidden_dim: int = 384
    layers: int = 8
    heads: int = 6
    feedforward_dim: int = 1536
    sequence_length: int = 256

    def validate(self):
        if self.vocab_size != VOCAB_SIZE:
            raise ValueError("Only the fixed, unfitted 259-entry byte vocabulary is supported")
        for name in ("hidden_dim", "layers", "heads", "feedforward_dim", "sequence_length"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.heads:
            raise ValueError("Hidden dimension must divide evenly into attention heads")
        return self


class ByteTokenizer:
    """An immutable numeric byte map; never fits or reads another corpus."""

    vocab_size = VOCAB_SIZE

    @staticmethod
    def encode(text):
        if not isinstance(text, str):
            raise TypeError("Expected Unicode text")
        return [BOS, *(byte + 3 for byte in text.encode("utf-8", errors="strict")), EOS]

    @staticmethod
    def decode(ids):
        if any(type(value) is not int or not 0 <= value < VOCAB_SIZE for value in ids):
            raise ValueError("Invalid fixed byte token")
        return bytes(value - 3 for value in ids if value >= 3).decode("utf-8", errors="strict")


@dataclass(frozen=True)
class NativeDocument:
    """Review attestations are required metadata, not automatic language proof."""

    document_id: str
    language: str
    text: str
    source_url: str
    revision: str
    license_id: str
    attribution: str
    text_sha256: str
    near_duplicate_group: str
    rights_review: str
    foreign_passage_audit: str

    def validate(self, native_language):
        if native_language not in LANGUAGES or self.language != native_language:
            raise ValueError("Every document must belong to this one declared native language")
        if self.rights_review != "approved" or self.foreign_passage_audit != "passed":
            raise ValueError("Corpus rights and foreign-passage reviews must pass before encoding")
        if self.license_id not in ("CC-BY-SA-4.0", "CC-BY-4.0", "CC0-1.0"):
            raise ValueError("Unapproved source license; review before extending the allowlist")
        for name in ("document_id", "source_url", "revision", "attribution", "near_duplicate_group"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"Document {name} is required")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Document text must be nonempty")
        if hashlib.sha256(self.text.encode("utf-8", errors="strict")).hexdigest() != self.text_sha256:
            raise ValueError("Reviewed document text hash changed")
        return self


def document_split(document, seed=79003):
    """Keep externally reviewed near-duplicate groups in one 90/5/5 split."""
    if type(seed) is not int or not document.near_duplicate_group:
        raise ValueError("A fixed split seed and reviewed duplicate group are required")
    value = int.from_bytes(hashlib.sha256(f"{seed}:{document.near_duplicate_group}".encode()).digest()[:8], "big") % 100
    return "train" if value < 90 else "dev" if value < 95 else "test"


def document_window(documents, native_language, *, sequence_length=256, window=0):
    """Return independent document windows with no cross-document packing.

    Validate ALL records before encoding any text. No document or language is
    silently dropped. A production loader/sampler remains a future protocol.
    """
    documents = tuple(documents)
    if not documents or type(sequence_length) is not int or sequence_length < 1 or type(window) is not int or window < 0:
        raise ValueError("Supply documents and a valid fixed window")
    for document in documents:
        document.validate(native_language)
    rows = []
    for document in documents:
        encoded = ByteTokenizer.encode(document.text)
        start = window * sequence_length
        values = encoded[start:start + sequence_length + 1]
        if len(values) < 2:
            raise ValueError("Window has no next-token target; caller must log sampler decisions")
        rows.append(values + [PAD] * (sequence_length + 1 - len(values)))
    return torch.tensor(rows, dtype=torch.long)


class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.hidden_dim // config.heads
        self.qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(self, hidden, valid):
        batch, length, width = hidden.shape
        qkv = self.qkv(hidden).view(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()
        # SDPA bool True means an allowed key. Both future and PAD keys are
        # excluded. Every row has a valid first token, including padded queries.
        allowed = causal[None, None] & valid[:, None, None, :]
        states = F.scaled_dot_product_attention(query, key, value, attn_mask=allowed, dropout_p=0.0)
        return self.output(states.transpose(1, 2).contiguous().view(batch, length, width))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.hidden_dim)
        self.attention = CausalAttention(config)
        self.norm_feedforward = nn.LayerNorm(config.hidden_dim)
        self.feedforward = nn.Sequential(nn.Linear(config.hidden_dim, config.feedforward_dim), nn.GELU(),
                                         nn.Linear(config.feedforward_dim, config.hidden_dim))

    def forward(self, hidden, valid):
        hidden = hidden + self.attention(self.norm_attention(hidden), valid)
        hidden = hidden + self.feedforward(self.norm_feedforward(hidden))
        return hidden.masked_fill(~valid[:, :, None], 0)


class CausalLM(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config.validate()
        self.embedding = nn.Embedding(VOCAB_SIZE, config.hidden_dim, padding_idx=PAD)
        self.positions = nn.Embedding(config.sequence_length, config.hidden_dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, VOCAB_SIZE, bias=False)
        self.output.weight = self.embedding.weight

    def forward(self, ids):
        if (not isinstance(ids, torch.Tensor) or ids.dtype != torch.long or ids.ndim != 2
                or ids.shape[0] < 1 or not 1 <= ids.shape[1] <= self.config.sequence_length):
            raise ValueError("Expected a nonempty int64 BxT token matrix within the fixed context")
        if not bool(((ids >= 0) & (ids < VOCAB_SIZE)).all()):
            raise ValueError("Token IDs must use the fixed byte vocabulary")
        valid = ids != PAD
        if not bool(valid[:, 0].all()) or bool((~valid[:, :-1] & valid[:, 1:]).any()):
            raise ValueError("PAD must follow nonempty document content; only right padding is allowed")
        positions = torch.arange(ids.shape[1], device=ids.device)
        hidden = (self.embedding(ids) + self.positions(positions)).masked_fill(~valid[:, :, None], 0)
        for block in self.blocks:
            hidden = block(hidden, valid)
        hidden = self.norm(hidden).masked_fill(~valid[:, :, None], 0)
        return self.output(hidden)


def build_model(config=ModelConfig(), seed=79007):
    """Deterministic random initialization without consuming the caller's RNG."""
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Initialization seed must be an unsigned 32-bit integer")
    with torch.random.fork_rng(devices=[]):
        # Seed CPU state only. torch.manual_seed would also affect GPU RNGs,
        # which are deliberately outside this CPU construction scope.
        torch.random.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
        model = CausalLM(config)
        for module in model.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                nn.init.normal_(module.weight, mean=0., std=.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
        with torch.no_grad():
            model.embedding.weight[PAD].zero_()
        # Residual branches use a fixed depth-aware initialization scale.
        for block in model.blocks:
            nn.init.normal_(block.attention.output.weight, std=.02 / math.sqrt(2 * config.layers))
            nn.init.normal_(block.feedforward[-1].weight, std=.02 / math.sqrt(2 * config.layers))
    return model


def next_token_loss(model, token_windows):
    if (not isinstance(token_windows, torch.Tensor) or token_windows.dtype != torch.long or token_windows.ndim != 2
            or not 2 <= token_windows.shape[1] <= model.config.sequence_length + 1):
        raise ValueError("A token window must contain inputs plus at least one next-token target")
    valid = token_windows != PAD
    if (not bool(((token_windows >= 0) & (token_windows < VOCAB_SIZE)).all()) or not bool(valid[:, 0].all())
            or bool((~valid[:, :-1] & valid[:, 1:]).any())):
        raise ValueError("Whole input/target windows require valid IDs and right padding")
    targets = token_windows[:, 1:]
    if not bool((targets != PAD).any()):
        raise ValueError("No non-PAD next-token targets")
    logits = model(token_windows[:, :-1])
    return F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1), ignore_index=PAD)
