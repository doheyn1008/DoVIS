"""Unactivated PNG-only object-memory diagnostic, outside the frozen package.

Load with importlib using the name ``mutual_lab.object_memory`` to resolve the
existing factories without adding a file to the research package. This changes
only the attentive producer's memory provider. It is an engineered idealized
perception comparison, including feature scaling and segmentation assumptions;
it is not evidence for general vision or one isolated internal feature.
"""

from __future__ import annotations

import hashlib
import io
from types import MethodType

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
BACKGROUND_THRESHOLD = .02


def _png_bytes(png):
    if type(png) is not bytes or not png.startswith(PNG_MAGIC):
        raise ValueError("Object memory accepts PNG bytes only, never labels, scenes or filenames")


def _components(mask):
    """Find eight-connected foreground sets without ordering objects by position."""
    height, width = mask.shape
    remaining = set(int(index) for index in mask.reshape(-1).nonzero()[0])
    components = []
    while remaining:
        seed = remaining.pop()
        stack, pixels = [seed], []
        while stack:
            index = stack.pop()
            y, x = divmod(index, width)
            pixels.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dx or dy) or not (0 <= y + dy < height and 0 <= x + dx < width):
                        continue
                    neighbor = (y + dy) * width + x + dx
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
        components.append(pixels)
        if len(components) > 2:
            break
    if len(components) != 2 or remaining:
        raise ValueError("Object memory requires exactly two disconnected foreground components")
    return components


def _occupancy(mask, np):
    """Exact area overlap with an 8-row/3-column bounding-box grid, even at 1px."""
    height, width = mask.shape
    def overlaps(size, cells):
        boundaries = np.linspace(0, size, cells + 1, dtype=np.float64)
        pixels = np.arange(size, dtype=np.float64)
        return np.maximum(0, np.minimum(boundaries[1:, None], pixels[None] + 1)
                          - np.maximum(boundaries[:-1, None], pixels[None]))
    weights_y, weights_x = overlaps(height, 8), overlaps(width, 3)
    return ((weights_y @ mask @ weights_x.T) / ((height / 8) * (width / 3))).reshape(-1)


def component_descriptors(png):
    """Return two unordered 34D continuous descriptors using only this PNG.

    RGB mean(3), image-area fraction(1), image-normalized bbox width/height(2),
    bbox mask occupancy8x3(24), image-normalized central momentsxx/yy(2), and
    actual centroidyx mapped to[-1,1](2). No palette, shape IDs, fitted image
    statistics, renderer centers, object ranks or Scene metadata are consulted.
    """
    _png_bytes(png)
    from .neural import Image, np, torch
    with Image.open(io.BytesIO(png)) as image:
        if image.convert("RGBA").getchannel("A").getextrema() != (255, 255):
            raise ValueError("Object memory requires an opaque PNG")
        pixels = np.asarray(image.convert("RGB"), dtype=np.float64) / 255
    height, width, _ = pixels.shape
    foreground = np.max(np.abs(pixels - pixels[:1, :1]), axis=-1) > BACKGROUND_THRESHOLD
    rows = []
    for points in _components(foreground):
        y, x = np.asarray(points, dtype=np.int64).T
        ymin, xmin, ymax, xmax = int(y.min()), int(x.min()), int(y.max()), int(x.max())
        box_height, box_width = ymax - ymin + 1, xmax - xmin + 1
        mask = np.zeros((box_height, box_width), dtype=np.float64)
        mask[y - ymin, x - xmin] = 1
        nx, ny = (x + .5) / width, (y + .5) / height
        row = np.concatenate((pixels[y, x].mean(axis=0),
            [len(points) / (height * width), box_width / width, box_height / height],
            _occupancy(mask, np), [((nx - nx.mean()) ** 2).mean(), ((ny - ny.mean()) ** 2).mean()],
            [2 * ny.mean() - 1, 2 * nx.mean() - 1]))
        rows.append(row)
    descriptors = torch.tensor(np.stack(rows), dtype=torch.float32)
    if descriptors.shape != (2, 34) or not bool(torch.isfinite(descriptors).all()):
        raise ValueError("Object memory produced invalid component descriptors")
    return descriptors


def _visual_grid(self, pngs):
    from .neural import F, torch
    if not isinstance(pngs, (list, tuple)) or not pngs:
        raise ValueError("Supply a nonempty sequence of PNG bytes")
    if any(parameter.requires_grad for parameter in self.visual_encoder.parameters()):
        raise RuntimeError("The inherited spatial CNN must remain frozen")
    for png in pngs:
        _png_bytes(png)
        key = hashlib.sha256(png).digest()
        if key not in self.object_memory_cache:
            if self.object_memory_source == "cnn":
                raw = self.object_memory_cnn_grid([png])[0]
            else:
                # Equal duplication preserves the two-object softmax mass.
                raw = component_descriptors(png).repeat_interleave(12, dim=0)
            # A common, data-independent scale rule in both F0 and F1. The
            # normalized CNN arm is a NEW baseline, distinct from old D1.
            self.object_memory_cache[key] = torch.cat((F.normalize(raw[:, :32], dim=-1), raw[:, 32:]), -1).detach()
    return torch.stack([self.object_memory_cache[hashlib.sha256(png).digest()] for png in pngs])


def build_object_memory_model(profile, initialization_seed, bundle, condition="B1", *, memory="objects"):
    """Keep all attentive tensors/buffers and change only the PNG memory method.

    Understanding retains its original global CNN path and training flags.
    This prototype is tested with producer-only updates; a future full-L1
    comparison would need the existing independently clipped branches.
    """
    if condition != "B1":
        raise ValueError("This conditional prototype fixes the inherited pretrained B1 prior")
    if memory not in ("cnn", "objects"):
        raise ValueError("Choose the fixed cnn or objects memory provider")
    from .attentive_spatial import build_attentive_spatial_model
    model = build_attentive_spatial_model(profile, initialization_seed, bundle, condition=condition)
    model.object_memory_source = memory
    model.object_memory_cnn_grid = model.visual_grid
    model.object_memory_cache = {}
    model.visual_grid = MethodType(_visual_grid, model)
    model.object_memory_metadata = {
        "status": "unactivated_prototype", "memory_source": memory, "excluded_from_research": True,
        "eligible_for_fresh_experiment": False, "engineered_idealized_perception": memory == "objects",
        "coordinate_order": "y,x", "memory_normalization": "l2_first32_raw_yx_v1",
        "input": "Opaque PNG bytes only, original resolution; no labels, Scene, filenames or fit statistics",
        "segmentation": ("Eight-connected foreground; max RGB distance from top-left background >0.02; exactly2 disconnected unoccluded components assumed; invalid counts raise"
                         if memory == "objects" else "No component segmentation; inherited frozen B1 CNN grid"),
        "descriptor": ("RGB3+area1+bboxwidthheight2+maskoccupancy8x3+centralmomentsxx/yy2+actualcentroidyx2=34"
                       if memory == "objects" else "Existing32 CNN feature channels plus2 fixed grid coordinates"),
        "normalization": "BOTH arms apply per-token L2 normalization to first32channels, no fitted statistics; final centroid/gridyx remains[-1,1] in inherited CNN row/column order; object raw features use RGB/255, fixed image dimensions and bbox occupancy",
        "baseline": "F0 normalized CNN is a NEW baseline, distinct from the old unnormalized D1 memory; F1 changes source, partition and descriptors as one bundle",
        "memory": ("Two unordered tokens, each repeated12times, no rank labels or positional sorting"
                   if memory == "objects" else "Existing24 CNN cells with fixed spatial coordinates"),
        "parameter_change": "None; existing producer, understanding, CNN and buffers are retained exactly",
        "interpretation": "Whole engineered memory-provider comparison includes feature scaling and world assumptions; not general vision, single-feature causality or research qualification"}
    return model
