import unittest

import numpy as np

from vision_feature_cache import VisionFeatureCache


class VisionFeatureCacheTests(unittest.TestCase):
    def setUp(self):
        self.images = {
            "a": (np.full((4, 2), 11), [1, 2, 2]),
            "b": (np.full((8, 2), 22), [1, 2, 4]),
            "c": (np.full((16, 2), 33), [1, 4, 4]),
            "d": (np.full((4, 2), 44), [1, 2, 2]),
        }
        self.batches = []

    def tower(self, pixels, grid):
        self.batches.append((pixels[:, 0].tolist(), grid.tolist()))
        return pixels[::4].astype("<u2")

    def encode(self, cache, *keys):
        images = [self.images[key] for key in keys]
        pixels = np.concatenate([pixels for pixels, _ in images])
        grid = np.array([grid for _, grid in images])
        return cache.encode(pixels, grid, keys, 4, self.tower)

    def test_warm_and_partial_hits_preserve_prompt_order(self):
        live, cold = VisionFeatureCache(64), VisionFeatureCache(0)
        expected = self.encode(cold, "b", "a")
        self.assertEqual(self.encode(live, "b", "a"), expected)
        self.batches.clear()
        self.assertEqual(self.encode(live, "b", "a"), expected)
        self.assertEqual(self.batches, [])
        expected = self.encode(cold, "a", "c", "b", "c")
        self.batches.clear()
        self.assertEqual(self.encode(live, "a", "c", "b", "c"), expected)
        self.assertEqual(self.batches, [([33] * 16, [[1, 4, 4]])])

    def test_most_recent_hit_survives_eviction(self):
        cache = VisionFeatureCache(12)
        self.encode(cache, "a", "b")
        self.encode(cache, "a")
        self.encode(cache, "d")
        self.assertEqual(list(cache.entries), ["a", "d"])
        self.assertEqual(cache.nbytes, 8)

    def test_oversize_feature_does_not_evict_reusable_features(self):
        cache = VisionFeatureCache(12)
        self.encode(cache, "a", "b")
        self.encode(cache, "c")
        self.assertEqual(list(cache.entries), ["a", "b"])
        self.assertEqual(cache.nbytes, 12)
        self.batches.clear()
        self.encode(cache, "c")
        self.assertEqual(len(self.batches), 1)

    def test_disabled_cache_encodes_every_row(self):
        cache = VisionFeatureCache(0)
        self.assertEqual(self.encode(cache, "a", "a"), self.encode(cache, "a", "a"))
        self.assertEqual(self.batches, [([11] * 8, [[1, 2, 2]] * 2)] * 2)
        self.assertFalse(cache.entries)

    def test_incompatible_pixel_layout_bypasses_cache(self):
        cache = VisionFeatureCache(64)
        self.encode(cache, "a")
        self.batches.clear()
        pixels = np.full((8, 2), 99)
        result = cache.encode(pixels, np.array([[1, 2, 2]]), ["a"], 4, self.tower)
        self.assertEqual(result, pixels[::4].astype("<u2").tobytes())
        self.assertEqual(len(self.batches), 1)
        self.assertEqual(cache.nbytes, 4)

    def test_cache_is_scoped_to_one_tower(self):
        first, second = VisionFeatureCache(64), VisionFeatureCache(64)
        self.encode(first, "a")
        self.batches.clear()
        self.encode(second, "a")
        self.assertEqual(len(self.batches), 1)


if __name__ == "__main__":
    unittest.main()
