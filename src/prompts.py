from src.models import MinimalSource

from pathlib import Path

MAX_CONTEXT_CHARS: int = 6000


class Prompt():

    def load_chunk_text(self, chunk: MinimalSource) -> str:
        try:
            with open(
                chunk.file_path,
                "r",
                encoding="utf-8",
                errors="ignore",
            ) as f:
                content = f.read()
        except OSError:
            return ""

        return content[
            chunk.first_character_index : chunk.last_character_index
        ]

    def truncate_contexts(self, contexts: list[str], budget: int):
        lengths = 0
        result = []
        for chunk in contexts:
            lengths += len(chunk)
            result.append(chunk)
            if lengths >= budget:
                return result
        return result

    def build_prompt(
        self,
        question: str,
        contexts: list[MinimalSource],
    ) -> str:
        context_texts = [self.load_chunk_text(chunk) for chunk in contexts]

        chunks = self.truncate_contexts(
            context_texts,
            MAX_CONTEXT_CHARS,
        )

        retrieved_context = "\n\n".join(chunks)

        return f"""Context:
        {retrieved_context}

        Question:
        {question}"""
