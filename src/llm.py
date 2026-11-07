#!/usr/bin/env python3
"""Ollama client with deterministic options, caching and call accounting."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache" / "llm"
LOG_PATH = ROOT / "results" / "llm_calls.jsonl"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if not BASE_URL.startswith("http"):
    BASE_URL = "http://" + BASE_URL


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ns: int = 0
    prompt_eval_duration_ns: int = 0
    eval_duration_ns: int = 0
    load_duration_ns: int = 0
    cached: bool = False
    model: str = ""
    tag: str = ""
    error: str | None = None
    raw: dict = field(default_factory=dict)


class OllamaClient:
    def __init__(self, model: str = "qwen7b:latest",
                 num_ctx: int = 6144, temperature: float = 0.0,
                 seed: int = 42, top_p: float = 1.0,
                 repeat_penalty: float = 1.05, use_cache: bool = True,
                 timeout: int = 600):
        self.model = model
        self.base_options = {
            "num_ctx": num_ctx,
            "temperature": temperature,
            "seed": seed,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
        }
        self.use_cache = use_cache
        self.timeout = timeout
        self.cooldown = float(os.environ.get("LLM_COOLDOWN", "0.0"))

    def _post(self, url: str, payload: dict):
        # A fresh request object avoids stale pooled connections in long runs.
        return requests.post(url, json=payload, timeout=self.timeout,
                             proxies={"http": None, "https": None})

    def _cache_key(self, system: str, prompt: str, options: dict, fmt: str | None) -> str:
        payload = json.dumps(
            {"model": self.model, "system": system, "prompt": prompt,
             "options": options, "format": fmt}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def generate(self, prompt: str, system: str = "", num_predict: int = 320,
                 fmt: str | None = None, tag: str = "", use_cache: bool | None = None,
                 extra_options: dict | None = None) -> LLMResult:
        options = dict(self.base_options)
        options["num_predict"] = num_predict
        if extra_options:
            options.update(extra_options)
        use_cache = self.use_cache if use_cache is None else use_cache
        key = self._cache_key(system, prompt, options, fmt)

        if use_cache:
            cache_file = CACHE_DIR / f"{key}.json"
            if cache_file.exists():
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                res = LLMResult(**data)
                res.cached = True
                res.tag = tag
                return res

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [],
            "stream": False,
            "options": options,
        }
        if system:
            payload["messages"].append({"role": "system", "content": system})
        payload["messages"].append({"role": "user", "content": prompt})
        if fmt:
            payload["format"] = fmt

        t0 = time.time()
        try:
            r = self._post(f"{BASE_URL}/api/chat", payload)
            r.raise_for_status()
            d = r.json()
            msg = d.get("message", {})
            res = LLMResult(
                text=msg.get("content", "") or "",
                prompt_tokens=int(d.get("prompt_eval_count", 0) or 0),
                completion_tokens=int(d.get("eval_count", 0) or 0),
                total_duration_ns=int(d.get("total_duration", 0) or 0),
                prompt_eval_duration_ns=int(d.get("prompt_eval_duration", 0) or 0),
                eval_duration_ns=int(d.get("eval_duration", 0) or 0),
                load_duration_ns=int(d.get("load_duration", 0) or 0),
                cached=False,
                model=self.model,
                tag=tag,
                raw={k: v for k, v in d.items() if k not in ("message",)},
            )
        except Exception as e:
            res = LLMResult(text="", model=self.model, tag=tag, error=f"{type(e).__name__}: {e}",
                            total_duration_ns=int((time.time() - t0) * 1e9))

        record = {
            "tag": tag, "model": self.model, "options": options, "format": fmt,
            "system_chars": len(system), "prompt_chars": len(prompt),
            "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
            "wall_seconds": round(time.time() - t0, 3),
            "total_duration_s": round(res.total_duration_ns / 1e9, 3),
            "cached": False, "error": res.error, "cache_key": key,
            "text_chars": len(res.text),
        }
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if not res.cached and self.cooldown > 0 and res.error is None:
            time.sleep(self.cooldown)
        if use_cache and res.error is None:
            payload_to_save = res.__dict__.copy()
            payload_to_save["raw"] = {}
            (CACHE_DIR / f"{key}.json").write_text(
                json.dumps(payload_to_save, ensure_ascii=False), encoding="utf-8")
        return res

    def embed(self, texts: list[str], model: str = "nomic-embed-text") -> list[list[float]]:
        r = self._post(f"{BASE_URL}/api/embed",
                       {"model": model, "input": texts})
        r.raise_for_status()
        if self.cooldown > 0:
            time.sleep(min(0.2, self.cooldown / 2))
        return r.json()["embeddings"]


def warm_up(client: OllamaClient) -> LLMResult:
    return client.generate("Reply with the single word: ready", num_predict=8, tag="warmup",
                           use_cache=False)


if __name__ == "__main__":
    c = OllamaClient()
    print(json.dumps(warm_up(c).__dict__, ensure_ascii=False, indent=2))
