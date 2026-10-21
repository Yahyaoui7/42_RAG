from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "How do I run a Docker container?",
    "Start a container using Docker.",
    "How do I cook pasta?",
]

embeddings = model.encode(sentences)

print(embeddings.shape)
print(embeddings[0])

similarities = model.similarity(embeddings, embeddings)
print(similarities)
