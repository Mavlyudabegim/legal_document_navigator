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
import re

def _setting(name: str) -> str | None:
    """Environment variable (.env locally) first, then Streamlit secrets (Streamlit Cloud)."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st
        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:
        return None


PROVIDERS = {
    #  name      default model           base_url                          env var with the key
    "groq":   ("openai/gpt-oss-20b",  "https://api.groq.com/openai/v1",   "GROQ_API_KEY"),
    "openai": ("gpt-4o-mini",         None,                               "OPENAI_API_KEY"),
    "ollama": ("gpt-oss:20b",         "http://localhost:11434/v1",        None),
    # other OpenAI-compatible hosts with free tiers (check their current limits)
    "cerebras":   ("gpt-oss-120b",          "https://api.cerebras.ai/v1",   "CEREBRAS_API_KEY"),
    "openrouter": ("openai/gpt-oss-120b",   "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
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
        # usage counters, for latency/cost reports (eval/contractnli.py)
        self.calls = self.prompt_tokens = self.completion_tokens = self.length_hits = 0
        self.last_finish_reason = None           # "length" = ran out of max_tokens

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            if self.key_env:
                api_key = _setting(self.key_env)
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
        self.calls += 1
        usage = getattr(resp, "usage", None)
        if usage is not None:                    # thinking tokens are counted as completion tokens
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        self.last_finish_reason = resp.choices[0].finish_reason
        self.length_hits += self.last_finish_reason == "length"
        text = resp.choices[0].message.content or ""
        # some local models (e.g. Qwen3, DeepSeek-R1) put their thinking in the answer as
        # <think>...</think>; remove it so the answer starts with "Yes."/"No." as the prompt asks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)   # opening tag already cut off
        return text.strip()

    def usage(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "length_hits": self.length_hits}