from src.models import MinimalSource
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from src.config import QWEN_MODEL_ID, QWEN_MAX_NEW_TOKENS

MAX_CONTEXT_CHARS: int = 50000


class Queen_model:
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

        self.user_prompt = f"""Context:
        {retrieved_context}

        Question:
        {question}"""

    def generate_answer(self) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
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


if __name__ == "__main__":
    data = {
        "file_path": "data/raw/vllm-0.10.1/vllm/engine/multiprocessing/client.py",
        "first_character_index": 1658,
        "last_character_index": 2238,
    }

    obje = MinimalSource.model_validate(data)
    text = load_chunk_text(obje)

    print("Path:", obje.file_path)
    print(
        "Expected length:",
        obje.last_character_index - obje.first_character_index,
    )
    print("Actual length:", len(text))
    print("Extracted chunk:")
    print(text)
