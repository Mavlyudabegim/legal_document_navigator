"""
llm.py  —  one small class that talks to whichever LLM you choose

Providers (all speak the OpenAI-compatible API, so one client works for all):
  "groq"   : Groq API (fast, free tier). Needs GROQ_API_KEY.   pip install openai
  "openai" : OpenAI API. Needs OPENAI_API_KEY.                  pip install openai
  "ollama" : free local model via Ollama (https://ollama.com).  pip install openai

Thinking: gpt-oss models "think" before answering. reasoning_effort=None uses the model's
default ("medium"), which gave the best accuracy in our ContractNLI evaluation (78.3%).
"low" is faster but less careful; "high" was too slow (~3 min per answer on a laptop).
"""
from __future__ import annotations

import os

PROVIDERS = {
    #  name      default model           base_url                          env var with the key
    "groq":   ("openai/gpt-oss-20b",  "https://api.groq.com/openai/v1",   "GROQ_API_KEY"),
    "openai": ("gpt-4o-mini",         None,                               "OPENAI_API_KEY"),
    "ollama": ("gpt-oss:20b",         "http://localhost:11434/v1",        None),
}


class LLM:
    def __init__(self, provider: str = "groq", model: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 1500,
                 reasoning_effort: str | None = None):    # None = model default (medium)
        if provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {list(PROVIDERS)}")
        self.provider = provider
        default_model, self.base_url, self.key_env = PROVIDERS[provider]
        self.model = model or os.environ.get("LLM_MODEL") or default_model
        self.temperature, self.max_tokens = temperature, max_tokens
        self.reasoning_effort = reasoning_effort
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            if self.key_env:
                api_key = os.environ.get(self.key_env)
                if not api_key:
                    raise RuntimeError(f"{self.key_env} is not set. "
                                       f"Run: export {self.key_env}=your_key")
            else:
                api_key = "ollama"                     # Ollama needs no key, any text works
            self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        return self._client

    def complete(self, system: str, user: str) -> str:
        extra = {}
        # only reasoning models accept this setting; others would reject the request
        if self.reasoning_effort and "gpt-oss" in self.model:
            extra["reasoning_effort"] = self.reasoning_effort
        resp = self._get_client().chat.completions.create(
            model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            **extra,
        )
        return (resp.choices[0].message.content or "").strip()