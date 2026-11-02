from fastapi import FastAPI, HTTPException
from .retriever import Searcher
from .caching import Caching
from .prompts import Prompt
from .generate import Generate

app = FastAPI()


@app.get("/search")
def get_search(query: str, k: int = 10):
    """Return the top-k sources for a single query."""

    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if k <= 0:
        raise HTTPException(
            status_code=400, detail="k must be greater than zero"
        )

    try:
        searcher = Searcher(k=k)
        results = searcher.get_resources(query)

        return {
            "query": query,
            "results": [
                {
                    "file_path": chunk.file_path,
                    "start": chunk.first_character_index,
                    "end": chunk.last_character_index,
                    "text": chunk.text,
                }
                for chunk in results
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/answer")
def get_answer(query: str, k: int = 10):

    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if k <= 0:
        raise HTTPException(
            status_code=400, detail="k must be greater than zero."
        )

    try:
        cache = Caching(top_k=k)
        cached = cache.get_answer_from_cache(query)

        if cached:
            return {"answer": cached}

        searcher = Searcher(k=k)
        sources = searcher.get_resources(query)

        prompt = Prompt().build_prompt(query, sources)

        generator = Generate()
        answer_text = generator.generate_answer(prompt)

        cache.save_query_and_answer(query, answer_text)

        return {"answer": answer_text}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating answer: {e}"
        )
