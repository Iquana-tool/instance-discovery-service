"""INSID3-Instance: training-free intra-image instance discovery from frozen DINOv3 features.

This package adapts the INSID3 (CVPR 2026) in-context segmentation recipe to *instance*
suggestion **within a single image**. INSID3 itself produces one mask for a prompted
concept; it does not individuate instances. Here we keep its building blocks -- DINOv3
features, agglomerative over-clustering, prototype/backward-correspondence matching, and
self-similarity grouping -- and add the one thing INSID3 lacks: an *individuation prior*
that splits the gated class region into separate instances.

The pipeline is intentionally split into small, pure-ish stage functions so the
accompanying notebook can visualize each step. See ``pipeline.run`` for the orchestration
and ``InSID3Result`` for the bag of intermediates.

Stages
------
1. :mod:`features`       -- DINOv3 dense features + exemplar prototypes.
2. :mod:`gate`           -- semantic gate: which patches are the exemplar class.
3. :mod:`clustering`     -- agglomerative over-segmentation into part-level atoms.
4. :mod:`individuation`  -- marker-controlled watershed (the corrected watershed step).
5. :mod:`merge`          -- region-adjacency merge by intra-image self-similarity.
"""

from models.insid3.pipeline import InSID3Params, InSID3Result, run

__all__ = ["InSID3Params", "InSID3Result", "run"]
