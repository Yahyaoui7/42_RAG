from src.models import (
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    MinimalAnswer,
)
from src.prompts import Prompt
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from pathlib import Path

from src.config import QWEN_MODEL_ID, QWEN_MAX_NEW_TOKENS


class Generate:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
        )
        self.system_prompt = """You are a helpful assistant for the vLLM codebase.

        Answer the user's question using only the provided context.

        Rules:
        - Give a direct, clear answer to the question.
        - Use relevant details from the context when they help answer the question.
        - Do not invent facts or use outside knowledge.
        - If the context does not contain enough information to answer,
        say "I don't know."
        - Keep the answer concise and technically accurate."""

    def generate_answer(self, prompt) -> str:
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

        return self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
        ).strip()

    def answer_dataset(
        self,
        student_result: StudentSearchResults,
        save_directory: Path,
    ) -> StudentSearchResultsAndAnswer:

        answered_results: list[MinimalAnswer] = []

        for retrieve in tqdm(student_result.search_results):
            prompt = Prompt().build_prompt(
                retrieve.question,
                retrieve.retrieved_sources,
            )

            answer = self.generate_answer(prompt)

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

        Path(save_directory).mkdir(parents=True, exist_ok=True)
        (Path(save_directory) / Path(student_result).name).write_text(
            result.model_dump_json(indent=2)
        )
