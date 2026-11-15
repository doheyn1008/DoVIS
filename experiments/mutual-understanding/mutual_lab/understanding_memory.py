"""Conditional G prototype: PNG object context for byte-state understanding.

Outside the frozen package; load as ``mutual_lab.understanding_memory``. This
adds engineered perception, parameters and a new understanding readout as one
bundle. It establishes neither word alignment nor role preservation after
learning. The selected F producer memory and all producer tensors stay fixed.
"""

from __future__ import annotations

import hashlib
import math
from types import MethodType


def _objects(self, pngs):
    from .neural import F, torch
    from .object_memory import component_descriptors
    if not isinstance(pngs, (list, tuple)) or not pngs:
        raise ValueError("Supply a nonempty PNG byte sequence")
    if any(p.requires_grad for p in self.visual_encoder.parameters()):
        raise RuntimeError("Understanding object context requires the frozen image encoder")
    for png in pngs:
        if type(png) is not bytes or not png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Understanding memory accepts PNG bytes only")
        key = hashlib.sha256(png).digest()
        if key not in self.understanding_object_cache:
            raw = component_descriptors(png)
            if raw.shape != (2, 34) or not bool(torch.isfinite(raw).all()) or raw.requires_grad:
                raise ValueError("Expected exactly two finite frozen component tokens, never a CNN grid")
            if not bool((raw[:, -2:].abs() <= 1 + 1e-6).all()):
                raise ValueError("Component centroids must use normalized y,x coordinates")
            self.understanding_object_cache[key] = torch.cat((F.normalize(raw[:, :32], dim=-1), raw[:, -2:]), -1).detach()
    return torch.stack([self.understanding_object_cache[hashlib.sha256(png).digest()] for png in pngs])


def _parts(self, texts, choices):
    from .neural import torch
    if (not isinstance(texts, (list, tuple)) or not texts or any(type(text) is not str for text in texts)
            or not isinstance(choices, (list, tuple)) or len(choices) != len(texts)):
        raise ValueError("Supply native text strings and one PNG choice sequence per text")
    count = len(choices[0])
    if count not in (2, 4) or any(len(row) != count for row in choices):
        raise ValueError("Comprehension requires consistently two or four image choices")
    pngs = [png for row in choices for png in row]
    if any(type(png) is not bytes or not png.startswith(b"\x89PNG\r\n\x1a\n") for png in pngs):
        raise ValueError("Comprehension choices must be PNG bytes, never Scene metadata or filenames")
    ids, lengths = self.tokenizer.batch(list(texts))
    visual = self.visual(pngs).reshape(len(texts), count, -1)
    objects = (self.understanding_objects(pngs).reshape(len(texts), count, 2, 34)
               if self.understanding.object_context else None)
    return self.understanding.logit_parts(ids, lengths, visual, objects)


def _logits(self, texts, choices):
    return self.comprehension_logit_parts(texts, choices)["total"]


def _grounding_loss(self, examples):
    from .neural import F, torch
    logits = self.comprehension_logits([row["text"] for row in examples], [row["choices"] for row in examples])
    answers = [row["answer"] for row in examples]
    if any(type(answer) is not int or not 0 <= answer < logits.shape[1] for answer in answers):
        raise ValueError("Comprehension answers must identify an available image choice")
    return F.cross_entropy(logits, torch.tensor(answers))


def _predict(self, text, pngs):
    from .neural import torch
    with torch.no_grad():
        return int(self.comprehension_logits([text], [pngs])[0].argmax())


def build_understanding_memory_model(profile, initialization_seed, bundle, condition="B1", *,
                                     understanding="objects", producer_memory="objects"):
    """Create G0(global) or G1(objects), fixing the producer memory beforehand."""
    if understanding not in ("global", "objects"):
        raise ValueError("Choose global or objects understanding")
    from .frozen_features import enforce_singleton_features
    from .neural import F, digest, nn, torch
    from .object_memory import build_object_memory_model
    model = build_object_memory_model(profile, initialization_seed, bundle, condition=condition, memory=producer_memory)
    enforce_singleton_features(model)
    base, dimension = model.understanding, profile.hidden_dim
    initial_base, initial_producer = digest(base.state_dict()), digest(model.producer.state_dict())

    class GroundedUnderstanding(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base
            self.object_context = understanding == "objects"
            if self.object_context:
                self.query = nn.Linear(dimension, dimension, bias=False)
                self.key = nn.Linear(34, dimension, bias=False)
                self.value = nn.Linear(34, dimension, bias=False)
                self.residual_output = nn.Linear(dimension, 1, bias=False)
                nn.init.zeros_(self.residual_output.weight)

        def logit_parts(self, ids, lengths, visual=None, objects=None):
            if visual is None:
                raise ValueError("Image-aware understanding requires comprehension_logits(texts, PNG choices)")
            if (ids.ndim != 2 or len(lengths) != ids.shape[0] or any(type(n) is not int or not 1 <= n <= ids.shape[1] for n in lengths)
                    or visual.ndim != 3 or visual.shape[0] != ids.shape[0] or visual.shape[1] not in (2, 4)
                    or visual.shape[-1] != dimension or not bool(torch.isfinite(visual).all())):
                raise ValueError("Invalid token lengths or finite global image vectors")
            if self.object_context:
                if (objects is None or objects.shape != (*visual.shape[:2], 2, 34)
                        or not bool(torch.isfinite(objects).all()) or objects.requires_grad):
                    raise ValueError("Understanding requires two finite frozen PNG component tokens per choice, never CNN grid cells")
            packed = nn.utils.rnn.pack_padded_sequence(self.base.embedding(ids), lengths,
                                                       batch_first=True, enforce_sorted=False)
            packed_states, hidden = self.base.gru(packed)  # Exactly one GRU pass per text batch.
            text = F.normalize(self.base.head(hidden[-1]), dim=-1)
            baseline = torch.einsum("bh,bkh->bk", text, visual) / .1
            residual = torch.zeros_like(baseline)
            if self.object_context:
                states, _ = nn.utils.rnn.pad_packed_sequence(packed_states, batch_first=True, total_length=ids.shape[1])
                scores = torch.einsum("bth,bksh->bkts", self.query(states), self.key(objects)) / math.sqrt(dimension)
                context = torch.einsum("bkts,bksh->bkth", torch.softmax(scores, dim=-1), self.value(objects))
                token_scores = self.residual_output(states[:, None] * context).squeeze(-1)
                valid = torch.arange(ids.shape[1], device=ids.device)[None, :] < torch.tensor(lengths, device=ids.device)[:, None]
                residual = (token_scores * valid[:, None]).sum(-1) / torch.tensor(lengths, device=ids.device)[:, None]
            total = baseline + residual
            if not bool(torch.isfinite(total).all()):
                raise ValueError("Non-finite image-aware comprehension logits")
            return {"base": baseline, "residual": residual, "total": total}

        def forward(self, ids, lengths, visual=None, objects=None):
            return self.logit_parts(ids, lengths, visual, objects)["total"]

    with torch.random.fork_rng():
        torch.manual_seed((initialization_seed + 190013) % 2**32)
        model.understanding = GroundedUnderstanding()
    model.understanding_object_cache = {}
    model.understanding_objects = MethodType(_objects, model)
    model.comprehension_logit_parts = MethodType(_parts, model)
    model.comprehension_logits = MethodType(_logits, model)
    model.grounding_loss = MethodType(_grounding_loss, model)
    model.predict = MethodType(_predict, model)
    extra = sum(parameter.numel() for name, parameter in model.understanding.named_parameters() if not name.startswith("base."))
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    model.development_spec = {**model.development_spec, "kind": "development_understanding_object_memory",
        "parameter_count": total_parameters, "trainable_parameter_count": trainable_parameters,
        "trainable_language_parameters": trainable_parameters, "understanding_extra_parameters": extra,
        "capacity_matched": False, "understanding_readout": "Base normalized final hidden cosine/0.1 plus optional object-context byte-state mean residual",
        "interpretation": "Understanding information, parameters and readout change together; producer perception is fixed"}
    model.understanding_memory_metadata = {
        "status": "conditional_unactivated_prototype", "excluded_from_research": True,
        "eligible_for_fresh_experiment": False, "understanding": understanding, "producer_memory": producer_memory,
        "initial_base_understanding_sha256": initial_base, "initial_producer_sha256": initial_producer,
        "understanding_extra_parameter_count": extra, "capacity_matched": False,
        "parameter_count": total_parameters, "trainable_parameter_count": trainable_parameters,
        "frozen_feature_policy": "per_png_singleton_v1", "memory_normalization": "l2_first32_raw_yx_v1", "coordinate_order": "y,x",
        "tokens": "Two raw unordered PNG component tokens, never inferred from duplicated producer rows; fixed L2 first32, raw y,x",
        "text": "One packed base GRU pass; same normalized base final hidden head; residual averages valid encoded byte/BMP states including BOS/EOS, excluding PAD",
        "readout": "Base cosine/0.1 plus mean zero-initialized scalar head of token_state times crossattended_object_context",
        "input_scope": "Native text and PNG bytes only, no Scene, class/role IDs, palette or fit statistics",
        "interpretation": "Engineered perception information + understanding parameters/readout bundle; no isolated attention effect, word alignment, role preservation or research qualification claim"}
    return model
