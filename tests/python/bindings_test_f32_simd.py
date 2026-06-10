import unittest

import numpy as np

import hnswlib

# Dims chosen to hit every distance-function dispatch path:
# scalar (<=4), 4-wide (dim % 4), 16-wide (dim % 16), and both residual paths.
DIMS = [3, 4, 8, 12, 15, 16, 17, 20, 30, 33, 64, 100, 128, 130]


def reference_distances(space, data, queries):
    if space == 'l2':
        return np.sum((queries[:, None, :] - data[None, :, :]) ** 2, axis=2)
    if space == 'ip':
        return 1.0 - queries @ data.T
    if space == 'cosine':
        qn = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        dn = data / np.linalg.norm(data, axis=1, keepdims=True)
        return 1.0 - qn @ dn.T
    raise ValueError(space)


class F32SimdTestCase(unittest.TestCase):
    def test_distances_match_numpy_reference(self):
        rng = np.random.default_rng(123)
        num_elements = 200
        k = 10

        for space in ['l2', 'ip', 'cosine']:
            for dim in DIMS:
                with self.subTest(space=space, dim=dim):
                    data = rng.standard_normal((num_elements, dim)).astype(np.float32)
                    queries = rng.standard_normal((20, dim)).astype(np.float32)

                    bf = hnswlib.BFIndex(space=space, dim=dim)
                    bf.init_index(max_elements=num_elements)
                    bf.add_items(data)

                    labels, distances = bf.knn_query(queries, k=k)

                    ref = reference_distances(space, np.float64(data), np.float64(queries))
                    for qi in range(len(queries)):
                        expected = ref[qi][labels[qi]]
                        np.testing.assert_allclose(
                            distances[qi], expected, rtol=1e-4, atol=1e-5)
                        # returned neighbors must actually be the k smallest
                        kth_best = np.sort(ref[qi])[k - 1]
                        self.assertLessEqual(distances[qi][-1], kth_best + 1e-3)

    def test_hnsw_agrees_with_bruteforce(self):
        rng = np.random.default_rng(321)
        num_elements = 2000
        k = 10

        for space in ['l2', 'ip', 'cosine']:
            for dim in [16, 17, 100, 128]:
                with self.subTest(space=space, dim=dim):
                    data = rng.standard_normal((num_elements, dim)).astype(np.float32)
                    if space == 'ip':
                        # keep self-similarity meaningful for ip recall
                        data /= np.linalg.norm(data, axis=1, keepdims=True)
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
                    correct = sum(
                        len(set(h) & set(b)) for h, b in zip(labels_hnsw, labels_bf)
                    )
                    recall = correct / (len(queries) * k)
                    self.assertGreater(recall, 0.9)


if __name__ == "__main__":
    unittest.main()
