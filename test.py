import bm25s

documents = [
    "Python is a programming language",
    "Python is used for machine learning",
    "Java is another programming language",
]

tokens = bm25s.tokenize(documents)
print(tokens)

retriever = bm25s.BM25()
retriever.index(tokens)

# Inspect the BM25 object
print("BM25 object:")
print(retriever)

print("\nInternal attributes:")
for key, value in vars(retriever).items():
    print(f"\n{key}:")
    print(value)
