from typing import Optional, Dict, Callable

import google.generativeai as genai
from mistralai import Mistral
from openai import OpenAI

from ..config.env import MISTRAL_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
from constants import MODEL_NAMES


#mistral_client = Mistral(api_key=MISTRAL_API_KEY)
#genai.configure(api_key=GEMINI_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def build_messages(prompt, system_instruction):
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    return messages

def _ask_openai(prompt, system_instruction, model_name):
    response = openai_client.chat.completions.create(
        model=model_name,
        messages=build_messages(prompt, system_instruction)
    )
    return response.choices[0].message.content


def _ask_mistral(prompt, system_instruction, model_name):
    response = mistral_client.chat.complete(
        model=model_name,
        messages=build_messages(prompt, system_instruction)
    )
    return response.choices[0].message.content


def _ask_gemini(prompt, system_instruction, model_name):
    model = genai.GenerativeModel(
        model_name,
        system_instruction=system_instruction if system_instruction else None
    )
    response = model.generate_content(prompt)
    return response.text


PROVIDERS: Dict[str, Callable] = {
    "gpt": _ask_openai,
    "mistral": _ask_mistral,
    "gemini": _ask_gemini,
}

def ask_llm(provider: str,
            prompt: str,
            system_instruction: Optional[str] = None):

    provider = provider.lower()

    if provider not in PROVIDERS:
        raise ValueError(f"Provider '{provider}' não suportado")

    fn = PROVIDERS[provider]

    model_name = MODEL_NAMES[provider]

    try:
        return fn(prompt, system_instruction, model_name)
    except Exception as e:
        print(f"[{provider.upper()} ERROR]: {e}")
        return None