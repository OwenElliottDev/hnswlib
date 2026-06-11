import unittest

import numpy as np

import hnswlib

EARTH_MEAN_RADIUS_KM = 6371.0088


def haversine_km(a, b):
    """Reference great-circle distance in km, a/b are (lat, lon) in degrees."""
    lat1, lon1 = np.radians(a[..., 0]), np.radians(a[..., 1])
    lat2, lon2 = np.radians(b[..., 0]), np.radians(b[..., 1])
    h = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_MEAN_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def random_coords(rng, n):
    lat = rng.uniform(-90, 90, n)
    lon = rng.uniform(-180, 180, n)
    return np.stack([lat, lon], axis=1).astype(np.float32)


class GeoDegreesTestCase(unittest.TestCase):
    def test_dim_must_be_two(self):
        for bad_dim in [1, 3, 128]:
            with self.assertRaises(RuntimeError):
                hnswlib.Index(space='geodegrees', dim=bad_dim)
            with self.assertRaises(RuntimeError):
                hnswlib.BFIndex(space='geodegrees', dim=bad_dim)

    def test_distance_matches_haversine(self):
        rng = np.random.default_rng(42)
        num_elements = 500

        data = random_coords(rng, num_elements)
        queries = random_coords(rng, 20)

        bf = hnswlib.BFIndex(space='geodegrees', dim=2)
        bf.init_index(max_elements=num_elements)
        bf.add_items(data)

        labels, distances = bf.knn_query(queries, k=5)
        for q, lab_row, dist_row in zip(queries, labels, distances):
            expected = haversine_km(q[np.newaxis, :], data[lab_row])
            np.testing.assert_allclose(dist_row, expected, rtol=1e-3, atol=0.5)

    def test_known_distances(self):
        # London (51.5074, -0.1278) to Paris (48.8566, 2.3522) is ~343 km
        bf = hnswlib.BFIndex(space='geodegrees', dim=2)
        bf.init_index(max_elements=10)
        bf.add_items(np.array([[48.8566, 2.3522]], dtype=np.float32), np.array([0]))
        labels, distances = bf.knn_query(np.array([[51.5074, -0.1278]], dtype=np.float32), k=1)
        self.assertEqual(labels[0][0], 0)
        self.assertAlmostEqual(distances[0][0], 343.5, delta=2.0)

        # identical points have zero distance
        bf2 = hnswlib.BFIndex(space='geodegrees', dim=2)
        bf2.init_index(max_elements=10)
        bf2.add_items(np.array([[51.5074, -0.1278]], dtype=np.float32), np.array([0]))
        labels, distances = bf2.knn_query(np.array([[51.5074, -0.1278]], dtype=np.float32), k=1)
        self.assertAlmostEqual(distances[0][0], 0.0, delta=1e-3)

    def test_hnsw_recall_against_bruteforce(self):
        rng = np.random.default_rng(7)
        num_elements = 2000
        k = 10

        data = random_coords(rng, num_elements)
        queries = random_coords(rng, 50)

        index = hnswlib.Index(space='geodegrees', dim=2)
        index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        index.add_items(data)
        index.set_ef(100)

        bf = hnswlib.BFIndex(space='geodegrees', dim=2)
        bf.init_index(max_elements=num_elements)
        bf.add_items(data)

        labels_hnsw, _ = index.knn_query(queries, k=k)
        labels_bf, _ = bf.knn_query(queries, k=k)

        correct = sum(
            len(set(h) & set(b)) for h, b in zip(labels_hnsw, labels_bf)
        )
        recall = correct / (len(queries) * k)
        self.assertGreater(recall, 0.95)


if __name__ == "__main__":
    unittest.main()
