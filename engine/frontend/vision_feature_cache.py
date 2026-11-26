"""Byte-bounded cache for one immutable vision tower; callers serialize access."""

from collections import OrderedDict
from itertools import accumulate

import numpy as np


class VisionFeatureCache:
    def __init__(self, max_bytes):
        self.max_bytes = max(0, max_bytes)
        self.nbytes = 0
        self.entries = OrderedDict()

    def _put(self, key, value):
        if len(value) > self.max_bytes:
            return
        while self.nbytes + len(value) > self.max_bytes:
            _, evicted = self.entries.popitem(last=False)
            self.nbytes -= len(evicted)
        self.entries[key] = value
        self.nbytes += len(value)

    def encode(self, pixels, grid, keys, merge, run_tower):
        """Run only missing images, then concatenate BF16 bytes in prompt order."""
        if not self.max_bytes:
            return run_tower(pixels, grid).tobytes()
        patch_counts = np.prod(grid, axis=1, dtype=np.int64)
        if (np.any(patch_counts <= 0) or np.any(patch_counts % merge)
                or len(pixels) != patch_counts.sum()):
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
            if len(indices) == len(keys):
                encoded = run_tower(pixels, grid)
            else:
                # Pixels are packed in contiguous per-image rows. The tower
                # attends within each image, with independent positions.
                offsets = list(accumulate(map(int, patch_counts), initial=0))
                selected = np.concatenate([pixels[offsets[i]:offsets[i + 1]] for i in indices])
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
