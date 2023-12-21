import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for sentences
sentences = [
    "Advancements in AI are transforming the tech industry.",
    "Renewable energy sources are essential for sustainable development.",
    "Quantum computing has the potential to revolutionize data processing.",
    "The exploration of Mars could reveal new insights about our solar system.",
    "Machine learning algorithms are becoming increasingly efficient.",
    "Blockchain technology is changing the way digital transactions are secured.",
    "The human genome project offers unprecedented insights into genetic makeup.",
    "Autonomous vehicles could redefine the future of transportation.",
    "Virtual reality is creating new opportunities in gaming and education.",
    "3D printing is revolutionizing manufacturing and design processes.",
    "Nanotechnology is leading to breakthroughs in medicine and materials science.",
    "The study of neural networks contributes to our understanding of AI.",
    "Robotic automation is being integrated into various industries.",
    "Wireless technology is key to the growth of the Internet of Things.",
    "Climate change research is vital for environmental preservation."
]
sentence_embeddings = model.encode(sentences)

# Generate embeddings for positive query terms
positive_queries = [
    "artificial intelligence",
    "sustainable energy",
    "quantum technology",
    "space exploration",
    "machine learning",
    # Add more positive queries as needed
]
positive_query_embeddings = model.encode(positive_queries)

# Generate embeddings for negative queries
negative_queries = [
    "classical music compositions",
    "19th-century literature",
    "medieval history",
    "ocean biodiversity",
    "culinary recipes",
    # Add more queries as needed
]
negative_query_embeddings = model.encode(negative_queries)

# Initialize and configure HNSWlib
dim = sentence_embeddings.shape[1]
num_elements = len(sentence_embeddings)

p = hnswlib.Index(space='cosine', dim=dim)
p.init_index(max_elements=num_elements, ef_construction=100, M=16)
p.set_ef(10)
p.set_num_threads(4)

# Adding sentence embeddings to the index
print("Adding sentence embeddings")
p.add_items(sentence_embeddings)

# Query the elements using positive query terms
print("Querying with positive query terms")
labels, distances = p.knn_query(positive_query_embeddings, k=10)
print("Distances for positive queries:", distances)

# Perform semantic filter query
print("Performing semantic filter query")
labels, distances = p.knn_semantic_filter_query(positive_query_embeddings, negative_query_embeddings, 0.9, k=10)
print("Distances for semantic filter query:", distances)
