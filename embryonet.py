#!/usr/bin/env python3
"""Optional second opinion: EmbryoNet, for the wells the geometric detector
is not confident about.

The plain detector in process.py finds the specimen as the peak of smoothed
gradient energy. That is right for the overwhelming majority of wells and needs
no model, no GPU and no download. It is NOT right when the specimen is small,
sits against the frame edge, or is faint -- there the argmax can settle on the
zero-padding artifact at exactly the kernel radius.

A LOW CONFIDENCE SCORE IS NOT EVIDENCE OF AN EMPTY WELL. It only means this
particular heuristic could not commit. Treating it as "nothing there" was
wrong: on AQV10 the flagged wells contained embryos, and only B03 is actually
empty. So the flagged wells get a second opinion here rather than a verdict.

This is deliberately tiny. It loads the SavedModel the previous pipeline
already ships and asks it for one box per frame -- nothing else from that
pipeline is carried over.

    from embryonet import EmbryoNet
    net = EmbryoNet(model_dir)          # raises if TF or the model is missing
    yx  = net.detect(image)             # (cy, cx) or None

Everything is lazy: importing this module costs nothing, and TensorFlow is only
imported when a model is actually constructed. A run with no doubtful wells
never touches it.
"""
from __future__ import annotations

import os

import numpy as np

DEFAULT_MODEL_DIR = "/g/aulehla/Tiago/twinnet_clean/models/segmentation_model/saved_model"


class EmbryoNetUnavailable(RuntimeError):
    """TensorFlow missing, or the model directory is not where we looked."""


class EmbryoNet:
    """One loaded SavedModel, reused across wells."""

    def __init__(self, model_dir: str = None, min_score: float = 0.1):
        self.model_dir = model_dir or os.environ.get("EMBRYONET_DIR", DEFAULT_MODEL_DIR)
        self.min_score = float(min_score)
        if not os.path.isdir(self.model_dir):
            raise EmbryoNetUnavailable(f"no model at {self.model_dir}")
        try:
            import tensorflow as tf                              # noqa: F401
        except ImportError as e:
            raise EmbryoNetUnavailable(
                "tensorflow is not installed in this environment. On the "
                "cluster: module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1"
            ) from e
        import tensorflow as tf
        for g in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except RuntimeError:
                pass
        self._tf = tf
        self._fn = tf.saved_model.load(self.model_dir).signatures["serving_default"]
        self.on_gpu = bool(tf.config.list_physical_devices("GPU"))

    def detect(self, img):
        """(cy, cx) in full-resolution pixels for the best box, or None.

        Takes an ARRAY, not a path, so the caller reuses the frame it has
        already read rather than going back to the filesystem.
        """
        a = np.asarray(img)
        if a.ndim == 3:
            a = a[..., 0]
        # the model wants uint8 RGB; scale by this frame's own max, as the
        # previous pipeline did, so the input distribution matches training
        if a.dtype != np.uint8:
            a8 = (a.astype(np.float32) / max(1, float(a.max())) * 255).astype(np.uint8)
        else:
            a8 = a
        rgb = np.repeat(a8[:, :, None], 3, axis=2)
        det = self._fn(self._tf.convert_to_tensor(rgb[None, ...], dtype=self._tf.uint8))
        scores = det["detection_scores"].numpy()[0]
        boxes = det["detection_boxes"].numpy()[0]
        if len(scores) == 0 or float(scores[0]) < self.min_score:
            return None
        h, w = a.shape[:2]
        ymin, xmin, ymax, xmax = boxes[0]
        return (int((ymin + ymax) / 2 * h), int((xmin + xmax) / 2 * w))


def try_load(model_dir=None, min_score=0.1, log=print):
    """EmbryoNet or None, never raising -- the fallback is optional by design."""
    try:
        net = EmbryoNet(model_dir, min_score)
        log(f"  EmbryoNet loaded from {net.model_dir} "
            f"({'GPU' if net.on_gpu else 'CPU'})")
        return net
    except EmbryoNetUnavailable as e:
        log(f"  EmbryoNet not available ({e}); keeping the geometric centres "
            f"for the doubtful wells")
        return None
