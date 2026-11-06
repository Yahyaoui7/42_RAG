from __future__ import annotations

from src.models import MinimalSource

MAX_CONTEXT_CHARS: int = 15000


class Prompt:
    """Prompt builder for grounding question answers in retrieved context."""

    def load_chunk_text(self, chunk: MinimalSource) -> str:
        """Read source text from file using slice offsets."""
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
            chunk.first_character_index:chunk.last_character_index
        ]

    def truncate_contexts(
        self, contexts: list[str], budget: int
    ) -> list[str]:
        """Truncate context snippets so total characters stay within budget."""
        lengths = 0
        result: list[str] = []
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
        """Construct prompt combining retrieved contexts and user question."""
        context_texts = [
            chunk.text if chunk.text else self.load_chunk_text(chunk)
            for chunk in contexts
        ]

        chunks = self.truncate_contexts(
            context_texts,
            MAX_CONTEXT_CHARS,
        )

        retrieved_context = "\n\n".join(chunks)

        return f"""Context:
{retrieved_context}

Question:
{question}"""
