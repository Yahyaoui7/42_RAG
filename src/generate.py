from __future__ import annotations

from typing import Any
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from src.models import (
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    MinimalAnswer,
)
from src.caching import Caching
from src.prompts import Prompt
from src.config import QWEN_MODEL_ID, QWEN_MAX_NEW_TOKENS


class Generate:
    """Generate answers using Qwen causal language model."""

    def __init__(self) -> None:
        """Initialize tokenizer and model."""
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
        )
        self.system_prompt = (
            "You are a helpful assistant for the vLLM codebase.\n\n"
            "Answer the user's question using only the provided context.\n\n"
            "Rules:\n"
            "- Give a direct, clear answer to the question.\n"
            "- Use relevant details from the context when they help.\n"
            "- Do not invent facts or use outside knowledge.\n"
            "- If the context does not contain enough information to answer,\n"
            'say "I don\'t know."\n'
            "- Keep the answer concise and technically accurate."
        )

    def generate_answer(self, prompt: str) -> str:
        """Generate answer text from formatted context prompt."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=QWEN_MAX_NEW_TOKENS,
                do_sample=False,
            )

        input_length = model_inputs.input_ids.shape[1]
        output_ids = generated_ids[0][input_length:]

        decoded = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
        )
        if isinstance(decoded, list):
            return " ".join(decoded).strip()
        return str(decoded).strip()

    def answer_dataset(
        self,
        student_result: StudentSearchResults,
        save_result_path: Path,
    ) -> StudentSearchResultsAndAnswer:
        """Process entire search results dataset and write answered results."""
        answered_results: list[MinimalAnswer] = []
        cache = Caching(student_result.k)

        for retrieve in tqdm(student_result.search_results):
            answer = cache.get_answer_from_cache(retrieve.question)
            if not answer:
                prompt = Prompt().build_prompt(
                    retrieve.question,
                    retrieve.retrieved_sources,
                )
                answer = self.generate_answer(prompt)
                cache.save_query_and_answer(retrieve.question, answer)

            minimal = MinimalAnswer(
                question_id=retrieve.question_id,
                question=retrieve.question,
                retrieved_sources=retrieve.retrieved_sources,
                answer=answer,
            )
            answered_results.append(minimal)

        result = StudentSearchResultsAndAnswer(
            search_results=answered_results,
            k=student_result.k,
        )

        Path(save_result_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_result_path).write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return result
