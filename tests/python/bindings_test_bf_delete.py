import unittest

import numpy as np

import hnswlib


class BFDeleteTestCase(unittest.TestCase):
    def test_delete_then_self_query(self):
        """Regression test: BFIndex.removePoint read the erased map iterator
        after erase (use-after-free), corrupting results after deletions."""
        dim, n = 4, 10
        rng = np.random.default_rng(0)
        data = rng.standard_normal((n, dim)).astype(np.float32) * 5

        bf = hnswlib.BFIndex(space='l2', dim=dim)
        bf.init_index(max_elements=n)
        bf.add_items(data, np.arange(n))

        for label in range(5):
            bf.delete_vector(label)

        labels, dists = bf.knn_query(data[5:], k=1)
        np.testing.assert_array_equal(labels.flatten(), np.arange(5, 10))
        np.testing.assert_allclose(dists.flatten(), 0, atol=1e-6)

        # refill the freed capacity and check everything again
        new = rng.standard_normal((5, dim)).astype(np.float32) * 5
        bf.add_items(new, np.arange(10, 15))
        labels, dists = bf.knn_query(np.vstack([data[5:], new]), k=1)
        np.testing.assert_array_equal(labels.flatten(), np.arange(5, 15))
        np.testing.assert_allclose(dists.flatten(), 0, atol=1e-6)

    def test_delete_last_element(self):
        """Removing the element in the last slot must not leave a phantom
        label mapping behind."""
        dim = 4
        data = np.arange(12, dtype=np.float32).reshape(3, dim)

        bf = hnswlib.BFIndex(space='l2', dim=dim)
        bf.init_index(max_elements=3)
        bf.add_items(data, np.array([0, 1, 2]))

        bf.delete_vector(2)  # occupies the last slot
        labels, _ = bf.knn_query(data[:2], k=1)
        np.testing.assert_array_equal(labels.flatten(), [0, 1])

        # the freed slot must be reusable without corrupting other labels
        bf.add_items(data[2:3] + 100, np.array([3]))
        labels, dists = bf.knn_query(np.vstack([data[:2], data[2:3] + 100]), k=1)
        np.testing.assert_array_equal(labels.flatten(), [0, 1, 3])
        np.testing.assert_allclose(dists.flatten(), 0, atol=1e-6)

    def test_churn_ground_truth_matches_numpy(self):
        """Sliding-window delete/insert churn must keep BFIndex results exact."""
        dim, n, churn, rounds, nq, k = 16, 2000, 100, 3, 50, 10
        rng = np.random.default_rng(42)
        total = n + rounds * churn + nq
        pool = rng.standard_normal((total, dim)).astype(np.float32)
        queries = pool[-nq:]
        pool = pool[:-nq]

        bf = hnswlib.BFIndex(space='l2', dim=dim)
        bf.init_index(max_elements=n)
        bf.add_items(pool[:n], np.arange(n))

        oldest, next_label = 0, n
        for _ in range(rounds):
            for label in range(oldest, oldest + churn):
                bf.delete_vector(label)
            oldest += churn
            bf.add_items(pool[next_label:next_label + churn],
                         np.arange(next_label, next_label + churn))
            next_label += churn

        bf_labels, _ = bf.knn_query(queries, k=k)

        live_labels = np.arange(oldest, next_label)
        live = np.float64(pool[live_labels])
        for qi in range(nq):
            d = np.sum((live - np.float64(queries[qi])) ** 2, axis=1)
            true_set = set(live_labels[np.argsort(d)[:k]])
            self.assertEqual(len(true_set & set(bf_labels[qi])), k,
                             f"query {qi}: BF results diverge from exact knn")


if __name__ == "__main__":
    unittest.main()
