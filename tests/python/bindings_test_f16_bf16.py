import unittest

import numpy as np

import hnswlib

# Dims chosen to hit the 16-wide, 4-wide, and residual kernel dispatch paths.
DIMS = [4, 8, 15, 16, 17, 32, 100, 128]


def snap_f16(x):
    """Round to values exactly representable in float16."""
    return np.float16(x).astype(np.float32)


def snap_bf16(x):
    """Truncate to values exactly representable in bfloat16."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return (u & 0xFFFF0000).view(np.float32)


SPACES = {
    'l2_f16': ('l2', snap_f16, 5e-3),
    'ip_f16': ('ip', snap_f16, 1e-3),
    'cosine_f16': ('cosine', snap_f16, 1e-2),
    'l2_bf16': ('l2', snap_bf16, 1e-3),
    'ip_bf16': ('ip', snap_bf16, 1e-3),
    'cosine_bf16': ('cosine', snap_bf16, 3e-2),
}


def reference_distances(metric, data, queries):
    data = np.float64(data)
    queries = np.float64(queries)
    if metric == 'l2':
        return np.sum((queries[:, None, :] - data[None, :, :]) ** 2, axis=2)
    if metric == 'ip':
        return 1.0 - queries @ data.T
    if metric == 'cosine':
        qn = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        dn = data / np.linalg.norm(data, axis=1, keepdims=True)
        return 1.0 - qn @ dn.T
    raise ValueError(metric)


class F16BF16TestCase(unittest.TestCase):
    def test_distances_match_reference(self):
        rng = np.random.default_rng(99)
        num_elements, k = 200, 10

        for space, (metric, snap, rtol) in SPACES.items():
            for dim in DIMS:
                with self.subTest(space=space, dim=dim):
                    # snapped inputs are exactly representable in the storage
                    # format, so the reference sees the same values the index
                    # stores (cosine re-normalizes internally, hence the
                    # looser tolerance there)
                    data = snap(rng.standard_normal((num_elements, dim)))
                    queries = snap(rng.standard_normal((20, dim)))

                    bf = hnswlib.BFIndex(space=space, dim=dim)
                    bf.init_index(max_elements=num_elements)
                    bf.add_items(data)

                    labels, distances = bf.knn_query(queries, k=k)
                    ref = reference_distances(metric, data, queries)
                    for qi in range(len(queries)):
                        expected = ref[qi][labels[qi]]
                        np.testing.assert_allclose(
                            distances[qi], expected, rtol=rtol, atol=rtol)

    def test_hnsw_agrees_with_bruteforce(self):
        rng = np.random.default_rng(77)
        num_elements, dim, k = 2000, 128, 10

        for space, (metric, snap, rtol) in SPACES.items():
            with self.subTest(space=space):
                data = snap(rng.standard_normal((num_elements, dim)))
                if metric == 'ip':
                    data = snap(data / np.linalg.norm(np.float64(data), axis=1, keepdims=True))
                queries = data[:50]

                index = hnswlib.Index(space=space, dim=dim)
                index.init_index(max_elements=num_elements, ef_construction=200, M=16)
                index.add_items(data)
                index.set_ef(100)

                bf = hnswlib.BFIndex(space=space, dim=dim)
                bf.init_index(max_elements=num_elements)
                bf.add_items(data)

                labels_hnsw, _ = index.knn_query(queries, k=k)
                labels_bf, _ = bf.knn_query(queries, k=k)
                hits = sum(len(set(h) & set(b)) for h, b in zip(labels_hnsw, labels_bf))
                recall = hits / (len(queries) * k)
                self.assertGreater(recall, 0.9, f"{space}: recall {recall}")


if __name__ == "__main__":
    unittest.main()
