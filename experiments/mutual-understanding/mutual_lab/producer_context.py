"""Unactivated H: change the whole producer-global conditioning input only.

Both GRU initialization and its repeated per-token visual input change in H1.
All trainable tensors, G1 understanding, object attention memory and CNN remain
the same. This is an engineered PNG representation treatment, not a proof of
an isolated initial-state bottleneck or general native-language competence.
"""

import hashlib
from types import MethodType


def position_moments(objects):
    """Two unordered normalized 34D objects -> unit 128D fixed representation."""
    from mutual_lab.neural import F, torch
    if objects.ndim != 3 or objects.shape[1:] != (2, 34) or objects.requires_grad or not bool(torch.isfinite(objects).all()):
        raise ValueError("Expected exactly two finite frozen raw object tokens, never duplicated grid rows")
    if not bool((objects[:, :, -2:].abs() <= 1 + 1e-6).all()):
        raise ValueError("Raw component coordinates must be normalized y,x")
    mean = objects.mean(dim=1)
    y_weighted = (objects[:, :, -2:-1] * objects).mean(dim=1)
    x_weighted = (objects[:, :, -1:] * objects).mean(dim=1)
    joined = F.pad(torch.cat((mean, y_weighted, x_weighted), dim=-1), (0, 26))
    return F.normalize(joined, dim=-1)


def _moment_context(self, pngs):
    from mutual_lab.neural import torch
    if not isinstance(pngs, (list, tuple)) or not pngs:
        raise ValueError("Supply a nonempty PNG byte sequence")
    for png in pngs:
        if type(png) is not bytes or not png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Producer context accepts PNG bytes only")
        key = hashlib.sha256(png).digest()
        if key not in self.producer_context_cache:
            # Always N=1 arithmetic. These are the original two component
            # descriptors, not positions selected from repeated 24-cell memory.
            objects = self.understanding_objects([png])
            self.producer_context_cache[key] = position_moments(objects)[0].detach()
    return torch.stack([self.producer_context_cache[hashlib.sha256(png).digest()] for png in pngs])


def _producer_visual(self, pngs):
    # Warm exactly the same immutable object-moment table in BOTH conditions,
    # including after the core benchmark clears caches. H0 never feeds these
    # moments to its producer; its context is the unchanged CNN global vector.
    moments = self.producer_moments(pngs)
    return self.visual(pngs) if self.producer_context_source == "cnn" else moments


def _teacher_forced_logits(self, ids, pngs):
    if ids.ndim != 2 or len(pngs) != ids.shape[0]:
        raise ValueError("Token and PNG batches must have the same size")
    logits, _ = self.producer(ids, self.producer_visual(pngs), spatial=self.visual_grid(pngs))
    return logits


def _production_loss(self, examples):
    from mutual_lab.neural import F
    ids, _ = self.tokenizer.batch([example["text"] for example in examples])
    logits = self.teacher_forced_logits(ids[:, :-1], [example["image"] for example in examples])
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), ignore_index=0)


def _generate_with_trace(self, png):
    from mutual_lab.neural import torch
    with torch.no_grad():
        visual, spatial = self.producer_visual([png]), self.visual_grid([png])
        current, hidden, output, terminated = torch.tensor([[1]]), None, [], False
        for _ in range(self.tokenizer.maximum - 2):
            logits, hidden = self.producer(current, visual, hidden, spatial=spatial)
            token = int(logits[0, -1].argmax())
            if token == 2:
                terminated = True
                break
            output.append(token)
            current = torch.tensor([[token]])
        invalid = False
        if self.tokenizer.kind == "utf8":
            try:
                bytes(token - 3 for token in output if token >= 3).decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                invalid = True
        else:
            invalid = any(0xD800 <= token - 3 <= 0xDFFF for token in output)
        return {"text": self.tokenizer.decode(output), "token_ids": output, "terminated": terminated,
            "invalid_encoding": invalid, "special_tokens_in_body": sum(token < 3 for token in output),
            "output_units": len(output), "unit_limit": self.tokenizer.maximum - 2}


def build_producer_context_model(profile, initialization_seed, bundle, condition="B1", *, context="object_moments"):
    """H0(context=cnn), H1(context=object_moments), exactly the G1/F1 NN state."""
    if profile.hidden_dim != 128 or condition != "B1" or context not in ("cnn", "object_moments"):
        raise ValueError("H fixes H128, the common B1 prior, and cnn/object_moments producer conditioning")
    from mutual_lab.understanding_memory import build_understanding_memory_model
    model = build_understanding_memory_model(profile, initialization_seed, bundle, condition=condition,
        understanding="objects", producer_memory="objects")
    model.producer_context_source = context
    model.producer_context_cache = {}
    model.producer_moments = MethodType(_moment_context, model)
    model.producer_visual = MethodType(_producer_visual, model)
    model.teacher_forced_logits = MethodType(_teacher_forced_logits, model)
    model.production_loss = MethodType(_production_loss, model)
    model.generate_with_trace = MethodType(_generate_with_trace, model)
    model.development_spec = {**model.development_spec, "kind": "development_producer_global_context",
        "capacity_matched": True, "producer_global_context": context,
        "interpretation": "H matches NN parameter capacity; conditioning representation, basis/rank/scaling change as one input treatment"}
    model.producer_context_metadata = {"status": "unactivated_prototype", "context": context,
        "excluded_from_research": True, "eligible_for_fresh_experiment": False,
        "scope": "Whole producer-global conditioning: initial GRU hidden AND repeated per-token concatenated vector",
        "representation": "mean(object34),mean(y*object34),mean(x*object34)=102;26fixedzeros;unitL2=128",
        "object_input": "Exactly two raw normalized PNG component tokens; unordered, y,x coordinates, no fitted statistics",
        "conditions": "H0 feeds unchanged frozen CNN global128; H1 feeds frozen position-moment128",
        "nn_parameters_changed": False, "capacity_matched": True,
        "initial_logits_matched": context == "cnn",
        "understanding": "Unchanged G1 base/residual/globalCNN/object context and losses",
        "producer_memory": "Same normalized PNG-object attention memory in both conditions",
        "cache_policy": "N=1 per PNG, both arms materialize same moment table even after benchmark reset",
        "input_confounds": "Representation basis, effective rank, scaling and26fixedzero dimensions change together; trainable NN counts stay exact",
        "interpretation": "Engineered perception-input treatment; no isolated-init-state, general vision or human-language claim"}
    return model


def preflight_context(model, pngs):
    """Warm every supplied PNG in both arms without changing tensors/RNG."""
    from mutual_lab.neural import digest, torch
    from mutual_lab.frozen_features import feature_cache_manifest
    before, rng = digest(model.state_dict()), torch.random.get_rng_state().clone()
    with torch.no_grad():
        for png in pngs:
            model.visual([png]); model.visual_grid([png]); model.understanding_objects([png]); model.producer_visual([png])
    if digest(model.state_dict()) != before or not torch.equal(rng, torch.random.get_rng_state()):
        raise RuntimeError("H preflight changed NN tensors or RNG")
    return feature_cache_manifest(model)
