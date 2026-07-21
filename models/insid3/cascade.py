"""Recursive instance discovery — zoom until the exemplar is re-identified.

Breadth-first, batched, connected-components fixed point. See
``docs/INSID3-Instance-recursive-redesign.md`` for the rationale. In brief, for each crop:

1. **Re-embed** (batched per level) and **gate** by cosine to the exemplar bank → foreground.
2. **Connected components** propose tighter crops:
   - ≥ 2 components  → enqueue each (tighter child crops);
   - 1 component whose bbox does **not** fill the crop → enqueue the tighter crop (keep zooming);
   - 1 component whose bbox **fills** the crop → *converged* (no tighter crop extractable).
3. A converged crop is **accepted as a leaf** iff its **CLS token** re-identifies the exemplar
   (``cos(CLS_crop, CLS_exemplar) ≥ cls_threshold``). A converged-but-rejected large crop is a
   seamless clump → optional watershed split; otherwise accepted as a single instance (the
   DINO-only ceiling) or discarded.

The exemplar defines the target *scale and granularity* — annotate a small morphology as the
exemplar and the same machinery discovers it. Only leaves are returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from scipy.ndimage import generate_binary_structure, label

from models.insid3 import border, clustering, features as featlib, gate as gatelib, individuation, merge
from models.insid3.pipeline import InSID3Params

_CONN8 = generate_binary_structure(2, 2)   # 8-connectivity: don't over-split single instances


@dataclass
class DiscoveredInstance:
    mask: np.ndarray                       # (H, W) uint8, ORIGINAL image coordinates
    box: tuple[int, int, int, int]         # (y0, y1, x0, x1) crop that isolated it
    depth: int                             # BFS level at which it converged
    cls_score: float                       # cos(CLS_crop, CLS_exemplar) — re-identification


@dataclass
class CascadeStats:
    n_embeds: int = 0
    max_depth: int = 0
    leaves: int = 0
    discarded: int = 0
    level_sizes: list[int] = field(default_factory=list)   # frontier size per BFS level


@dataclass
class _Region:
    box: tuple[int, int, int, int]
    depth: int


def _mask_bbox(mask, pad_frac):
    ys, xs = np.where(mask)
    h, w = mask.shape
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    py, px = int((y1 - y0) * pad_frac), int((x1 - x0) * pad_frac)
    return max(0, y0 - py), min(h, y1 + py), max(0, x0 - px), min(w, x1 + px)


def _extract_features(patches: torch.Tensor, mode: str, k: int) -> torch.Tensor:
    """Reduce one exemplar's masked patch features (M, D) to prototypes per ``mode``.

    ``"all"`` keeps every masked patch; ``"mean"`` returns the single mean; ``"cluster"``
    returns ``k`` (spherical) k-means centroids — a denoised, multi-modal prototype set.
    """
    if mode == "all" or patches.shape[0] <= max(k, 1):
        return patches
    if mode == "mean":
        return featlib.l2_normalize(patches.mean(dim=0, keepdim=True), dim=1)
    if mode == "cluster":
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(patches.detach().cpu().numpy())
        centers = torch.from_numpy(km.cluster_centers_).to(patches.device, patches.dtype)
        return featlib.l2_normalize(centers, dim=1)
    raise ValueError(f"unknown prototype_mode {mode!r}")


def _exemplar_bank(backbone, image, exemplar_masks, params, *,
                   pad_frac=0.15, prototype_mode="all", n_prototypes=4):
    """Build the prototype bank, cropping **each** exemplar separately.

    Per exemplar: crop to its bbox (+pad) → embed the crop → keep the patches inside the
    (crop-space) mask → reduce them via ``_extract_features``. All exemplars' prototypes are
    concatenated into one ``(M, D)`` bank; the CLS tokens are averaged into one ``(D,)`` vector.
    Everything stays in crop coordinates — nothing is mapped back to the root image.
    """
    feats_list, cls_list = [], []
    masks = [m.astype(bool) for m in exemplar_masks if m.any()]
    if not masks:
        raise ValueError("All exemplar masks are empty.")

    boxes = [_mask_bbox(m, pad_frac) for m in masks]
    crops = [image[y0:y1, x0:x1] for (y0, y1, x0, x1) in boxes]
    embedded = featlib.embed_batch(backbone, crops, chunk=8, standardize=params.standardize)

    for (feat, cls), m, (y0, y1, x0, x1) in zip(embedded, masks, boxes):
        mask_grid = featlib.resize_mask_to_grid(m[y0:y1, x0:x1], feat.shape[:2])
        if not mask_grid.any():
            # Exemplar too small to land on even its own crop grid → use all crop patches.
            mask_grid = np.ones(feat.shape[:2], dtype=bool)
        patches = feat[torch.from_numpy(mask_grid).to(feat.device)]   # (M_i, D)
        feats_list.append(_extract_features(patches, prototype_mode, n_prototypes))
        cls_list.append(cls)

    bank = torch.cat(feats_list, dim=0)                               # (M, D)
    bank_cls = featlib.l2_normalize(torch.stack(cls_list).mean(dim=0), dim=0)
    return bank, bank_cls


def _gate_to_bank(feat, bank, threshold):
    """Foreground grid: patches whose max cosine to the exemplar bank clears ``threshold``."""
    hp, wp, d = feat.shape
    sims = feat.reshape(hp * wp, d) @ bank.T                      # (P, M)
    return (sims.max(dim=1).values >= threshold).reshape(hp, wp).cpu().numpy()


def _grid_bbox(comp):
    rows, cols = np.where(comp)
    return rows.min(), rows.max() + 1, cols.min(), cols.max() + 1


def _child_box(comp, box, pad_frac):
    """Pixel bbox (in original coords) of a grid component within ``box``, padded."""
    y0, y1, x0, x1 = box
    ch, cw = y1 - y0, x1 - x0
    hp, wp = comp.shape
    rmin, rmax, cmin, cmax = _grid_bbox(comp)
    py0 = y0 + int(np.floor(rmin / hp * ch)); py1 = y0 + int(np.ceil(rmax / hp * ch))
    px0 = x0 + int(np.floor(cmin / wp * cw)); px1 = x0 + int(np.ceil(cmax / wp * cw))
    pady, padx = int(round((py1 - py0) * pad_frac)), int(round((px1 - px0) * pad_frac))
    return (max(y0, py0 - pady), min(y1, py1 + pady),
            max(x0, px0 - padx), min(x1, px1 + padx))


def _box_area(box):
    return (box[1] - box[0]) * (box[3] - box[2])


def _split_clump(feat, comp_grid, params):
    """Watershed fallback for a seamless clump → list of sub-component grids (≥1)."""
    labels = clustering.agglomerative_oversegment(feat, comp_grid, params.cluster_tau)
    indiv = individuation.individuate(
        feat, comp_grid, labels, mode=params.marker_mode, alpha=params.elevation_alpha,
        beta=params.elevation_beta, marker_min_distance=params.marker_min_distance,
    )
    merged = merge.merge_instances(
        feat, indiv.instances, indiv.feature_boundary,
        similarity_threshold=params.merge_similarity, boundary_threshold=params.merge_boundary,
    )
    return [merged == i for i in np.unique(merged) if i != 0]


def discover_instances(
    backbone,
    image: np.ndarray,
    exemplar_masks: list[np.ndarray],
    negative_masks: list[np.ndarray] | None = None,
    params: InSID3Params | dict | None = None,
    *,
    exemplar_image: np.ndarray | None = None,
    max_depth: int = 8,
    min_crop: int = 64,
    pad_frac: float = 0.08,
    shrink_stop: float = 0.9,
    cls_threshold: float = 0.5,
    clump_area_factor: float = 1.5,
    min_instance_area: int = 16,
    embed_batch_size: int = 8,
    max_total_embeds: int = 512,
    split_mode: str = "watershed",
    border_mode: str = "proto",
    prototype_mode: str = "all",
    n_prototypes: int = 4,
    observer=None,
) -> tuple[list[DiscoveredInstance], CascadeStats]:
    """Discover instances of the exemplar class by recursive, batched zoom-in.

    Parameters
    ----------
    shrink_stop:
        A single component **converges** when its tighter child crop is no longer meaningfully
        smaller than the current crop — i.e. ``area(child) / area(crop) >= shrink_stop``. (Using
        an area ratio, not bbox-vs-crop, because the padded child crop never fills its parent.)
    observer:
        Optional ``callable(info: dict)`` invoked once per processed region for tracing /
        visualization. ``info`` has ``level, depth, box, decision, n_components, cls_score,
        feat, fg, comp_labels, children``.
    cls_threshold:
        Min ``cos(CLS_crop, CLS_exemplar)`` to accept a converged crop as a re-identified
        instance.
    clump_area_factor:
        A converged, CLS-rejected component larger than this × exemplar area is treated as a
        clump and split (``split_mode``); smaller ones are accepted as single instances.
    embed_batch_size / max_total_embeds:
        Crops per DINOv3 forward, and the global embed budget (safety cap).
    max_depth / min_crop:
        Recursion bounds — depth cap, and the resolution floor (stop zooming below this crop
        size, where re-embedding only interpolates).
    border_mode:
        Per-leaf border refinement against the exemplar prototype: ``"proto"`` (optimal
        threshold maximizing prototype cosine), ``"otsu"`` (mode split), or ``"static"`` (keep
        the gate threshold). See :mod:`models.insid3.border`.
    prototype_mode / n_prototypes:
        How each exemplar's masked patches are reduced into the bank — ``"all"`` (every patch),
        ``"mean"`` (one mean vector), or ``"cluster"`` (``n_prototypes`` k-means centroids, a
        denoised multi-modal prototype set). Each exemplar is cropped and embedded separately.
    exemplar_image:
        If given, the exemplar (and ``exemplar_masks``) live in *this* image while ``image`` is
        a **different target** — the in-context / cross-image setting. The bank is built from
        ``exemplar_image`` and every target crop (including the root) is gated against it.
        ``None`` (default) = intra-image: exemplar and instances share ``image``, and the root
        uses the accurate in-image gate. NB: cross-image matching carries a DINOv3 positional
        bias (INSID3 Sec. 3.1) that is not yet corrected here.
    """
    if not isinstance(params, InSID3Params):
        params = InSID3Params.from_dict(params)

    H, W = image.shape[:2]
    same_image = exemplar_image is None
    exemplar_area = float(np.logical_or.reduce([m.astype(bool) for m in exemplar_masks]).sum())
    leaves: list[DiscoveredInstance] = []
    stats = CascadeStats()

    bank, bank_cls = _exemplar_bank(backbone, image if same_image else exemplar_image,
                                    exemplar_masks, params,
                                    prototype_mode=prototype_mode, n_prototypes=n_prototypes)
    proto = featlib.l2_normalize(bank.mean(dim=0), dim=0)   # exemplar prototype for border refine
    stats.n_embeds += 1

    def emit(region, comp_grid, score, feat):
        # Optimal-threshold border refinement (per-instance, replaces the static gate threshold).
        if border_mode != "static":
            comp_grid, _, _ = border.refine(feat, comp_grid, proto, mode=border_mode)
        y0, y1, x0, x1 = region.box
        mask_local = cv2.resize(comp_grid.astype(np.uint8), (x1 - x0, y1 - y0),
                                interpolation=cv2.INTER_NEAREST)
        if int(mask_local.sum()) < min_instance_area:
            stats.discarded += 1
            return
        full = np.zeros((H, W), np.uint8)
        full[y0:y1, x0:x1] = mask_local
        leaves.append(DiscoveredInstance(full, region.box, region.depth, score))
        stats.leaves += 1

    frontier = [_Region((0, H, 0, W), 0)]
    level_idx = 0
    while frontier and stats.n_embeds < max_total_embeds:
        stats.level_sizes.append(len(frontier))
        crops = [image[r.box[0]:r.box[1], r.box[2]:r.box[3]] for r in frontier]
        embedded = featlib.embed_batch(backbone, crops, chunk=embed_batch_size,
                                       standardize=params.standardize)
        stats.n_embeds += len(crops)

        nxt: list[_Region] = []
        for r, (feat, cls) in zip(frontier, embedded):
            stats.max_depth = max(stats.max_depth, r.depth)

            # Root, intra-image: use the accurate in-image gate -- but only if the exemplar is
            # large enough to land on the full-image grid. A *small* exemplar covers < 1 patch
            # there, so we fall back to the bank gate (the bank comes from the exemplar's own
            # object-centered crop, which is robust to small objects).
            use_inimage_gate = (
                r.depth == 0 and same_image
                and featlib.stack_exemplar_patches(feat, exemplar_masks).shape[0] > 0
            )
            if use_inimage_gate:
                fg = gatelib.semantic_gate(feat, exemplar_masks, negative_masks,
                                           params.gate_threshold).foreground
            else:                             # children, cross-image root, or sub-patch exemplar
                fg = _gate_to_bank(feat, bank, params.gate_threshold)
            labels, n = (label(fg, structure=_CONN8) if fg.any()
                         else (np.zeros_like(fg, dtype=int), 0))

            floor = (r.box[1] - r.box[0]) <= min_crop or (r.box[3] - r.box[2]) <= min_crop
            decision, children, score = "empty", [], None

            if n == 0:
                pass
            elif r.depth >= max_depth or floor:          # forced convergence: emit components
                decision = "leaf-cap"
                score = float(cls @ bank_cls)
                for cid in range(1, n + 1):
                    emit(r, labels == cid, score, feat)
            elif n >= 2:                                  # multiple instances → tighter crops
                decision = "split"
                children = [_child_box(labels == cid, r.box, pad_frac) for cid in range(1, n + 1)]
            else:                                         # single component
                comp = labels == 1
                child = _child_box(comp, r.box, pad_frac)
                if _box_area(child) / max(_box_area(r.box), 1) < shrink_stop:
                    decision, children = "zoom", [child]  # strictly tighter → keep zooming
                else:                                     # converged — no tighter crop
                    score = float(cls @ bank_cls)
                    comp_area = float(comp.sum()) / comp.size * _box_area(r.box)
                    if score >= cls_threshold or comp_area <= clump_area_factor * exemplar_area:
                        decision = "leaf"; emit(r, comp, score, feat)
                    elif split_mode != "none":
                        subs = _split_clump(feat, comp, params)
                        if len(subs) >= 2:
                            decision = "clump-split"
                            children = [_child_box(s, r.box, pad_frac) for s in subs]
                        else:
                            decision = "leaf"; emit(r, comp, score, feat)  # unsplittable → accept
                    else:
                        decision = "leaf"; emit(r, comp, score, feat)

            if observer is not None:
                observer(dict(level=level_idx, depth=r.depth, box=r.box, decision=decision,
                              n_components=int(n), cls_score=score, feat=feat, fg=fg,
                              comp_labels=labels, children=list(children)))
            nxt += [_Region(cb, r.depth + 1) for cb in children]

        frontier = nxt
        level_idx += 1

    return leaves, stats
