import hnswlib
import numpy as np

dim = 16
num_elements = 10000

data = np.float32(np.random.random((num_elements, dim)))

data1 = data[:num_elements // 2]
data2 = data[num_elements // 2:]

p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip

p.init_index(max_elements=num_elements//2, ef_construction=100, M=16)

# Controlling the recall by setting ef:
# higher ef leads to better accuracy, but slower search
p.set_ef(10)

# Set number of threads used during batch search/construction
# By default using all available cores
p.set_num_threads(4)

print("Adding first batch of %d elements" % (len(data1)))
p.add_items(data1)

# Query the elements for themselves and measure recall:
labels, distances = p.knn_query(data1, k=1)
print("Recall for the first batch:", np.mean(labels.reshape(-1) == np.arange(len(data1))), "\n")

labels, distances = p.knn_semantic_filter_query(data1, data2, 0.5, k=1)
print("Recall for the first batch:", np.mean(labels.reshape(-1) == np.arange(len(data1))), "\n")


