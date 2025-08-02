import requests
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any


class OllamaLLM(LLM):
    model_name: str = "mistral"
    api_url: str = "http://localhost:11434/api/generate"
    temperature: float = 0.7
    max_tokens: int = 512

    @property
    def _llm_type(self) -> str:
        return "ollama"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

    @property
    def _default_params(self) -> Mapping[str, Any]:
        return self._identifying_params

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False
        }

        if stop:
            payload["stop"] = stop

        response = requests.post(self.api_url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("response", result.get("completion", str(result)))
