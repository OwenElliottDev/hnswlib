import os
import pickle
import unittest

import numpy as np

import hnswlib


def matryoshka_data(rng, n, dim):
    """Random data with exponentially decaying per-dimension variance, so
    leading dimensions carry most of the signal (as in MRL embeddings)."""
    scales = np.exp(-np.arange(dim) / (dim / 8))
    return (rng.standard_normal((n, dim)) * scales).astype(np.float32)


def l2_sqr(a, b):
    return np.sum((a - b) ** 2, axis=-1)


class MrlTestCase(unittest.TestCase):
    def test_validation(self):
        for bad_scan_dim in [-1, 128, 256]:
            with self.assertRaises(RuntimeError):
                hnswlib.Index(space='l2', dim=128, mrl_scan_dim=bad_scan_dim)

        for unsupported_space in ['l2_f16', 'cosine_bf16', 'geodegrees']:
            with self.assertRaises(RuntimeError):
                hnswlib.Index(space=unsupported_space, dim=128, mrl_scan_dim=16)

        # rerank_size is only valid on an MRL index
        index = hnswlib.Index(space='l2', dim=16)
        index.init_index(max_elements=10)
        index.add_items(np.zeros((5, 16), dtype=np.float32))
        with self.assertRaises(ValueError):
            index.knn_query(np.zeros((1, 16), dtype=np.float32), k=1, rerank_size=10)

    def test_scan_dim_distances_without_rerank(self):
        rng = np.random.default_rng(1)
        dim, scan_dim, num_elements = 64, 16, 500
        data = matryoshka_data(rng, num_elements, dim)
        queries = matryoshka_data(rng, 10, dim)

        index = hnswlib.Index(space='l2', dim=dim, mrl_scan_dim=scan_dim)
        index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        index.add_items(data)
        index.set_ef(100)

        labels, distances = index.knn_query(queries, k=5)
        for q, lab_row, dist_row in zip(queries, labels, distances):
            expected = l2_sqr(q[:scan_dim], data[lab_row][:, :scan_dim])
            np.testing.assert_allclose(dist_row, expected, rtol=1e-4)

    def test_rerank_distances_are_full_dim(self):
        rng = np.random.default_rng(2)
        dim, scan_dim, num_elements = 128, 32, 1000
        data = matryoshka_data(rng, num_elements, dim)
        queries = matryoshka_data(rng, 20, dim)

        index = hnswlib.Index(space='l2', dim=dim, mrl_scan_dim=scan_dim)
        index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        index.add_items(data)
        index.set_ef(100)

        labels, distances = index.knn_query(queries, k=10, rerank_size=200)
        for q, lab_row, dist_row in zip(queries, labels, distances):
            expected = l2_sqr(q, data[lab_row])
            np.testing.assert_allclose(dist_row, expected, rtol=1e-4)
            # results must be sorted by full-dim distance
            self.assertTrue(np.all(np.diff(dist_row) >= 0))

    def test_rerank_improves_recall(self):
        rng = np.random.default_rng(3)
        dim, scan_dim, num_elements, k = 256, 32, 5000, 10
        data = matryoshka_data(rng, num_elements, dim)
        queries = matryoshka_data(rng, 50, dim)

        index = hnswlib.Index(space='l2', dim=dim, mrl_scan_dim=scan_dim)
        index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        index.add_items(data)
        index.set_ef(150)

        bf = hnswlib.BFIndex(space='l2', dim=dim)
        bf.init_index(max_elements=num_elements)
        bf.add_items(data)
        labels_bf, _ = bf.knn_query(queries, k=k)

        def recall(labels_test):
            correct = sum(
                len(set(t) & set(b)) for t, b in zip(labels_test, labels_bf)
            )
            return correct / (len(queries) * k)

        labels_scan, _ = index.knn_query(queries, k=k)
        labels_rerank, _ = index.knn_query(queries, k=k, rerank_size=1000)

        recall_scan = recall(labels_scan)
        recall_rerank = recall(labels_rerank)

        self.assertGreater(recall_rerank, recall_scan)
        self.assertGreater(recall_rerank, 0.9)

    def test_cosine_rerank(self):
        rng = np.random.default_rng(4)
        dim, scan_dim, num_elements = 128, 32, 1000
        data = matryoshka_data(rng, num_elements, dim)
        queries = matryoshka_data(rng, 10, dim)

        index = hnswlib.Index(space='cosine', dim=dim, mrl_scan_dim=scan_dim)
        index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        index.add_items(data)
        index.set_ef(100)

        labels, distances = index.knn_query(queries, k=5, rerank_size=100)
        data_norm = data / np.linalg.norm(data, axis=1, keepdims=True)
        for q, lab_row, dist_row in zip(queries, labels, distances):
            qn = q / np.linalg.norm(q)
            expected = 1.0 - data_norm[lab_row] @ qn
            np.testing.assert_allclose(dist_row, expected, rtol=1e-3, atol=1e-5)

    def test_rerank_size_smaller_than_k(self):
        rng = np.random.default_rng(5)
        dim, scan_dim = 64, 16
        data = matryoshka_data(rng, 200, dim)

        index = hnswlib.Index(space='l2', dim=dim, mrl_scan_dim=scan_dim)
        index.init_index(max_elements=200)
        index.add_items(data)
        index.set_ef(50)

        labels, distances = index.knn_query(data[:5], k=10, rerank_size=3)
        self.assertEqual(labels.shape, (5, 10))

    def test_pickle_and_save_load_roundtrip(self):
        rng = np.random.default_rng(6)
        dim, scan_dim, num_elements = 64, 16, 500
        data = matryoshka_data(rng, num_elements, dim)
        queries = matryoshka_data(rng, 10, dim)

        index = hnswlib.Index(space='l2', dim=dim, mrl_scan_dim=scan_dim)
        index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        index.add_items(data)
        index.set_ef(100)

        labels, distances = index.knn_query(queries, k=5, rerank_size=100)

        # pickle round trip
        restored = pickle.loads(pickle.dumps(index))
        self.assertEqual(restored.mrl_scan_dim, scan_dim)
        labels_p, distances_p = restored.knn_query(queries, k=5, rerank_size=100)
        np.testing.assert_array_equal(labels, labels_p)
        np.testing.assert_allclose(distances, distances_p, rtol=1e-6)

        # save / load round trip
        path = "mrl_test_index.bin"
        try:
            index.save_index(path)
            loaded = hnswlib.Index(space='l2', dim=dim, mrl_scan_dim=scan_dim)
            loaded.load_index(path, max_elements=num_elements)
            loaded.set_ef(100)
            labels_l, distances_l = loaded.knn_query(queries, k=5, rerank_size=100)
            np.testing.assert_array_equal(labels, labels_l)
            np.testing.assert_allclose(distances, distances_l, rtol=1e-6)
        finally:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
