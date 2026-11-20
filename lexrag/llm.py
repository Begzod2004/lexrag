"""LLM provayderlari (Anthropic Claude, OpenAI, OpenAI-mos vLLM/Qwen) + fallback + JSON ajratish."""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field

import requests

from .config import LLMEndpoint

log = logging.getLogger("lexrag.llm")


class LLMError(Exception):
    def __init__(self, code: str, message: str, retryable: bool = True):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    ms: int = 0


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    models: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, r: LLMResult):
        with self._lock:
            self.input_tokens += r.input_tokens
            self.output_tokens += r.output_tokens
            if r.model not in self.models:
                self.models.append(r.model)


_tls = threading.local()


def _session() -> requests.Session:
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        _tls.s = s
    return s


def _call_anthropic(ep: LLMEndpoint, system: str, user: str, max_tokens: int, temperature: float) -> LLMResult:
    t0 = time.time()
    body = {
        "model": ep.model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    try:
        r = _session().post(
            f"{(ep.base_url or 'https://api.anthropic.com').rstrip('/')}/v1/messages",
            json=body,
            headers={"x-api-key": ep.api_key or "", "anthropic-version": "2023-06-01", "content-type": "application/json"},
            timeout=ep.timeout,
        )
    except requests.Timeout as e:
        raise LLMError("LLM_TIMEOUT", f"{ep.label}: {e}") from e
    except requests.RequestException as e:
        raise LLMError("LLM_ERROR", f"{ep.label}: {e}") from e
    if r.status_code != 200:
        retry = r.status_code in (408, 409, 429, 500, 502, 503, 504, 529)
        raise LLMError("LLM_ERROR", f"{ep.label}: HTTP {r.status_code} {r.text[:300]}", retryable=retry)
    j = r.json()
    text = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
    u = j.get("usage", {})
    return LLMResult(text, ep.model, u.get("input_tokens", 0), u.get("output_tokens", 0), int((time.time() - t0) * 1000))


def _call_openai(ep: LLMEndpoint, system: str, user: str, max_tokens: int, temperature: float, seed: int | None) -> LLMResult:
    t0 = time.time()
    body: dict = {
        "model": ep.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
    }
    if seed is not None:
        body["seed"] = seed
    if ep.provider == "qwen":
        body["max_tokens"] = max_tokens
        # Qwen3: "thinking" rejimini o'chirish (vLLM)
        body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        body["max_completion_tokens"] = max_tokens
    headers = {"Content-Type": "application/json"}
    if ep.api_key:
        headers["Authorization"] = f"Bearer {ep.api_key}"
    url = f"{(ep.base_url or 'https://api.openai.com/v1').rstrip('/')}/chat/completions"
    try:
        r = _session().post(url, json=body, headers=headers, timeout=ep.timeout)
        if r.status_code == 400 and "temperature" in r.text and ep.provider == "openai":
            body.pop("temperature", None)  # ba'zi reasoning modellar temperature qabul qilmaydi
            r = _session().post(url, json=body, headers=headers, timeout=ep.timeout)
    except requests.Timeout as e:
        raise LLMError("LLM_TIMEOUT", f"{ep.label}: {e}") from e
    except requests.RequestException as e:
        raise LLMError("LLM_ERROR", f"{ep.label}: {e}") from e
    if r.status_code != 200:
        retry = r.status_code in (408, 409, 429, 500, 502, 503, 504)
        raise LLMError("LLM_ERROR", f"{ep.label}: HTTP {r.status_code} {r.text[:300]}", retryable=retry)
    j = r.json()
    text = j["choices"][0]["message"].get("content") or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    u = j.get("usage", {}) or {}
    return LLMResult(text, ep.model, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), int((time.time() - t0) * 1000))


class LLM:
    """Bir nechta endpoint: birinchisi asosiy, xato bo'lsa keyingisiga o'tadi."""

    def __init__(self, endpoints: list[LLMEndpoint]):
        self.endpoints = endpoints

    @property
    def available(self) -> bool:
        return bool(self.endpoints)

    @property
    def label(self) -> str:
        return self.endpoints[0].model if self.endpoints else "none"

    def complete(self, system: str, user: str, max_tokens: int = 1200, temperature: float = 0.0, seed: int | None = None, usage: Usage | None = None) -> LLMResult:
        if not self.endpoints:
            raise LLMError("LLM_ERROR", "LLM sozlanmagan (ANTHROPIC_API_KEY / OPENAI_API_KEY / QWEN_BASE_URL)", retryable=False)
        last: LLMError | None = None
        for ep in self.endpoints:
            for attempt in range(2):
                try:
                    if ep.provider == "anthropic":
                        res = _call_anthropic(ep, system, user, max_tokens, temperature)
                    else:
                        res = _call_openai(ep, system, user, max_tokens, temperature, seed)
                    if usage is not None:
                        usage.add(res)
                    return res
                except LLMError as e:
                    last = e
                    log.warning("LLM xato (%s, urinish %d): %s", ep.label, attempt + 1, e)
                    if not e.retryable:
                        break
                    time.sleep(1.0 + attempt * 2)
        assert last is not None
        raise last

    def complete_json(self, system: str, user: str, max_tokens: int = 1200, temperature: float = 0.0, seed: int | None = None, usage: Usage | None = None) -> tuple[dict, LLMResult]:
        res = self.complete(system, user, max_tokens, temperature, seed, usage)
        try:
            return extract_json(res.text), res
        except ValueError:
            # bir marta qayta so'raymiz
            res2 = self.complete(system, user + "\n\nReturn ONLY one valid JSON object, no prose.", max_tokens, temperature, seed, usage)
            return extract_json(res2.text), res2


def extract_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        v = json.loads(t)
        if isinstance(v, dict):
            return v
    except Exception:
        pass
    # birinchi muvozanatlangan {...}
    start = t.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = t[start : i + 1]
                    try:
                        v = json.loads(cand)
                        if isinstance(v, dict):
                            return v
                    except Exception:
                        cand2 = re.sub(r",\s*([}\]])", r"\1", cand)
                        try:
                            v = json.loads(cand2)
                            if isinstance(v, dict):
                                return v
                        except Exception:
                            pass
                    break
        start = t.find("{", start + 1)
    raise ValueError(f"JSON topilmadi: {text[:200]}")
