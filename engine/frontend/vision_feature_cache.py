"""Byte-bounded cache for one immutable vision tower; callers serialize access."""

from collections import OrderedDict
from dataclasses import dataclass
from itertools import accumulate

import numpy as np


@dataclass(frozen=True)
class CachedImage:
    key: bytes
    pixels: int
    grid: tuple[int, int, int]


class VisionFeatureCache:
    def __init__(self, max_bytes):
        self.max_bytes = max(0, max_bytes)
        self.nbytes = 0
        self.entries = OrderedDict()
        self.source_images = {}
        self.sources_by_feature = {}

    def find_source(self, digest):
        return self.source_images.get(digest)

    def remember_source(self, digest, record):
        # Keep one source digest per feature, with the same lifetime.
        if record.key in self.entries:
            previous = self.sources_by_feature.get(record.key)
            if previous != digest:
                self.source_images.pop(previous, None)
            self.sources_by_feature[record.key] = digest
            self.source_images[digest] = record

    def _put(self, key, value):
        if len(value) > self.max_bytes:
            return
        while self.nbytes + len(value) > self.max_bytes:
            key_to_evict, evicted = self.entries.popitem(last=False)
            source = self.sources_by_feature.pop(key_to_evict, None)
            self.source_images.pop(source, None)
            self.nbytes -= len(evicted)
        self.entries[key] = value
        self.nbytes += len(value)

    def encode(self, pixels, grid, keys, merge, run_tower, *, prepared=None):
        """Run only missing images, then concatenate BF16 bytes in prompt order."""
        if not self.max_bytes:
            return run_tower(pixels, grid).tobytes()
        patch_counts = np.prod(grid, axis=1, dtype=np.int64)
        sparse = prepared is not None
        prepared = list(range(len(keys))) if prepared is None else prepared
        available = patch_counts[prepared]
        if (np.any(patch_counts <= 0) or np.any(patch_counts % merge)
                or len(pixels) != available.sum()):
            if sparse:
                raise ValueError("new image pixels do not match the image grids")
            return run_tower(pixels, grid).tobytes()

        features = {}
        missing = {}
        for index, key in enumerate(keys):
            if key in self.entries:
                self.entries.move_to_end(key)
                features[key] = self.entries[key]
            elif key not in missing:
                missing[key] = index

        if missing:
            indices = list(missing.values())
            if indices == prepared:
                encoded = run_tower(pixels, grid[indices])
            else:
                # Pixels are packed in contiguous per-image rows. The tower
                # attends within each image, with independent positions.
                offsets = list(accumulate(map(int, available), initial=0))
                chunks = {i: pixels[offsets[j]:offsets[j + 1]] for j, i in enumerate(prepared)}
                selected = np.concatenate([chunks[i] for i in indices])
                encoded = run_tower(selected, grid[indices])
            feature_counts = [int(patch_counts[i]) // merge for i in indices]
            if encoded.ndim != 2 or encoded.shape[1] == 0 or len(encoded) != sum(feature_counts):
                raise ValueError("vision feature rows do not match the image grids")
            offset = 0
            for key, count in zip(missing, feature_counts):
                feature = encoded[offset:offset + count].tobytes()
                features[key] = feature
                self._put(key, feature)
                offset += count

        return b"".join(features[key] for key in keys)
