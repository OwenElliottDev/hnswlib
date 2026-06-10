import unittest

import numpy as np

import hnswlib


def build_index(data, labels, max_elements=None, allow_replace_deleted=False, ef=100):
    index = hnswlib.Index(space='l2', dim=data.shape[1])
    index.init_index(max_elements=max_elements or len(data), M=16, ef_construction=200,
                     allow_replace_deleted=allow_replace_deleted)
    index.add_items(data, labels)
    index.set_ef(ef)
    return index


class RemoveItemTestCase(unittest.TestCase):
    def test_removed_items_are_gone(self):
        rng = np.random.default_rng(1)
        n, dim = 1000, 32
        data = rng.standard_normal((n, dim)).astype(np.float32)
        index = build_index(data, np.arange(n))

        removed = set(range(0, 300))
        for label in removed:
            index.remove_item(label)

        labels, _ = index.knn_query(data, k=5)
        self.assertFalse(removed & set(labels.flatten()), "removed labels must never be returned")

        with self.assertRaises(RuntimeError):
            index.remove_item(0)  # already removed
        with self.assertRaises(RuntimeError):
            index.remove_item(99999)  # never existed
        with self.assertRaises(RuntimeError):
            index.unmark_deleted(0)  # label released, not tombstoned

    def test_remaining_elements_stay_searchable(self):
        rng = np.random.default_rng(2)
        n, dim, k = 5000, 32, 10
        data = rng.standard_normal((n, dim)).astype(np.float32)
        index = build_index(data, np.arange(n))

        for label in range(0, n, 3):  # remove every third element (~33%)
            index.remove_item(label)

        live_labels = np.array([l for l in range(n) if l % 3 != 0])
        live_data = data[live_labels]

        # exact ground truth over the live set
        bf = hnswlib.BFIndex(space='l2', dim=dim)
        bf.init_index(max_elements=len(live_labels))
        bf.add_items(live_data, live_labels)

        queries = rng.standard_normal((200, dim)).astype(np.float32)
        true_labels, _ = bf.knn_query(queries, k=k)
        found, _ = index.knn_query(queries, k=k)
        recall = sum(len(set(f) & set(t)) for f, t in zip(found, true_labels)) / (len(queries) * k)
        self.assertGreater(recall, 0.95, f"recall after heavy removal: {recall}")

        # self-recall of live elements
        labels_self, _ = index.knn_query(live_data, k=1)
        self.assertGreater((labels_self.flatten() == live_labels).mean(), 0.99)

    def test_slot_reuse_with_replace_deleted(self):
        rng = np.random.default_rng(3)
        n, dim = 2000, 32
        data = rng.standard_normal((n + 500, dim)).astype(np.float32)
        index = build_index(data[:n], np.arange(n), allow_replace_deleted=True)

        for label in range(500):
            index.remove_item(label)
        # capacity is full of dead slots: must be reusable without resize
        index.add_items(data[n:], np.arange(n, n + 500), replace_deleted=True)
        self.assertEqual(index.get_max_elements(), n)

        labels, dists = index.knn_query(data[n:], k=1)
        self.assertGreater((labels.flatten() == np.arange(n, n + 500)).mean(), 0.99)

    def test_remove_all_then_reuse(self):
        rng = np.random.default_rng(4)
        n, dim = 200, 16
        data = rng.standard_normal((n + 50, dim)).astype(np.float32)
        index = build_index(data[:n], np.arange(n), max_elements=n + 50,
                            allow_replace_deleted=True)

        # removes every element including the entry point (repeatedly reassigned)
        for label in range(n):
            index.remove_item(label)

        new_labels = np.arange(n, n + 50)
        index.add_items(data[n:], new_labels, replace_deleted=True)
        labels, dists = index.knn_query(data[n:], k=1)
        self.assertGreater((labels.flatten() == new_labels).mean(), 0.99)
        np.testing.assert_allclose(dists.flatten()[labels.flatten() == new_labels], 0, atol=1e-5)

    def test_interleaved_churn(self):
        rng = np.random.default_rng(5)
        n, dim, churn, rounds, k = 3000, 32, 150, 10, 10
        total = n + rounds * churn
        pool = rng.standard_normal((total, dim)).astype(np.float32)
        queries = rng.standard_normal((100, dim)).astype(np.float32)

        index = build_index(pool[:n], np.arange(n), allow_replace_deleted=True)
        bf = hnswlib.BFIndex(space='l2', dim=dim)
        bf.init_index(max_elements=n)
        bf.add_items(pool[:n], np.arange(n))

        oldest, next_label = 0, n
        for _ in range(rounds):
            for label in range(oldest, oldest + churn):
                index.remove_item(label)
                bf.delete_vector(label)
            oldest += churn
            index.add_items(pool[next_label:next_label + churn],
                            np.arange(next_label, next_label + churn), replace_deleted=True)
            bf.add_items(pool[next_label:next_label + churn],
                         np.arange(next_label, next_label + churn))
            next_label += churn

        true_labels, _ = bf.knn_query(queries, k=k)
        found, _ = index.knn_query(queries, k=k)
        recall = sum(len(set(f) & set(t)) for f, t in zip(found, true_labels)) / (len(queries) * k)
        self.assertGreater(recall, 0.95, f"recall after churn: {recall}")
        # half the index was churned; no removed label may surface
        removed = set(range(oldest))
        self.assertFalse(removed & set(found.flatten()))


if __name__ == "__main__":
    unittest.main()
