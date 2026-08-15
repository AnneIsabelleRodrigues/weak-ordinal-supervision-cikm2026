from typing import List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)

from ..config.settings import ModelConfig, RetrievedDocument

from ..config.env import HUGGINGFACE_TOKEN
from utils import _free_cuda


class MedGemmaGenerator:

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        token = HUGGINGFACE_TOKEN

        self.processor = AutoProcessor.from_pretrained(
            config.llm_model,
            trust_remote_code=True,
            token=token,
        )

        quant_config = None
        if config.use_quantization:
            quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True, )

        self.model = AutoModelForCausalLM.from_pretrained(
            config.llm_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quant_config,
            trust_remote_code=True,
            token=token,
            offload_folder="offload",
        )
        self.model.eval()

    def generate(
            self,
            system_instruction: str,
            query: str,
            context_docs: Optional[List[RetrievedDocument]] = None,
            max_new_tokens: int = 2000,
    ) -> str:

        if not query:
            return "Empty query."

        messages = []

        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_instruction}]
        })

        user_content = []

        if context_docs and len(context_docs) > 0:
            rag_text = "\n\n".join(
                f"[Document {i + 1}]\n{doc.text}" for i, doc in enumerate(context_docs)
            )

            user_content.append({
                "type": "text",
                "text": f"Documents retrieved from RAG:\n{rag_text}\n"
            })

        user_content.append({
            "type": "text",
            "text": query
        })

        messages.append({
            "role": "user",
            "content": user_content
        })

        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            decoded = self.processor.decode(generation[0][input_len:], skip_special_tokens=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return decoded

        except Exception as exc:
            return f"Generation failed: {exc}"
        finally:
            _free_cuda()