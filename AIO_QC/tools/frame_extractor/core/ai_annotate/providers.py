# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Talking to vision models — OpenAI-compatible services, Azure OpenAI, Gemini, Anthropic, AWS Bedrock (Qt-free).

Standard library HTTP only (no extra packages). One call:

    chat(cfg, system, parts) -> {"text", "usage": {"input", "output"}, "seconds"}
        parts: [("text", str) | ("image", bytes, mime) | ("cache", None)]   ("cache" marks the end of the part that
        is the same in every request — references and rules — for services that cache prompt prefixes)

Retries 429 / 5xx / time-outs / dropped connections with growing waits (honours Retry-After); other errors say
what went wrong in plain words. The key is read from its environment variable at call time and never logged.
"""
from __future__ import annotations

import base64
import json
import random
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

from AIO_QC.tools.frame_extractor.core.ai_annotate import keys
from AIO_QC.tools.frame_extractor.core.ai_annotate.profiles import PROVIDERS
from AIO_QC.core.diag import ignored as _ignored


class ProviderError(Exception):
    def __init__(self, message, status=0, retryable=False, wait=None):
        super().__init__(message)
        self.status, self.retryable, self.wait = status, retryable, wait


def kind(cfg) -> str:
    return PROVIDERS.get(cfg["connection"].get("provider") or "custom", PROVIDERS["custom"])[1]


def api_key(cfg) -> str:
    name = cfg["connection"].get("api_key_env") or ""
    return keys.get(name) if name else ""


def _openai_base(url: str) -> str:
    u = (url or "").rstrip("/")
    if u.endswith("/chat/completions"):
        u = u[: -len("/chat/completions")]
    if not u.endswith("/v1") and "/v1/" not in u + "/" and "compatible-mode" not in u and "/openai" not in u[-8:]:
        u += "/v1"
    return u


def _request(url, body=None, headers=None, timeout=120, method=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method or ("POST" if body is not None else "GET"),
                                 headers={"Content-Type": "application/json", "User-Agent": "AIO-QC",
                                          **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")
            j = json.loads(detail)
            err = j.get("error", j)
            detail = err.get("message") if isinstance(err, dict) else str(err)
        except Exception as _exc:
            _ignored(_exc)
        wait = None
        ra = e.headers.get("Retry-After") if e.headers else None
        if ra:
            try:
                wait = float(ra)
            except ValueError:
                wait = None
        what = {400: "the service did not accept the request", 401: "the key was not accepted",
                403: "this key may not use that model", 404: "not found (check the address and the model name)",
                413: "the request is too large (lower the image size)", 429: "too many requests (rate limit)"
                }.get(e.code, "the service returned an error")
        raise ProviderError(f"{e.code}: {what}" + (f" — {str(detail)[:300]}" if detail else ""), e.code,
                            e.code in (408, 409, 425, 429) or e.code >= 500, wait) from None
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
        reason = getattr(e, "reason", e)
        raise ProviderError(f"could not reach {urllib.parse.urlsplit(url).netloc or url} ({reason})", 0, True) from None
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError:
        raise ProviderError("the service answered with something that is not JSON", 0, True) from None


def _b64(b):
    return base64.b64encode(b).decode("ascii")


# ------------------------------------------------------------------------------------------------ per API
def _openai_call(cfg, system, parts, params):
    con = cfg["connection"]
    content = []
    for p in parts:
        if p[0] == "text":
            content.append({"type": "text", "text": p[1]})
        elif p[0] == "image":
            content.append({"type": "image_url", "image_url": {"url": f"data:{p[2]};base64,{_b64(p[1])}"}})
    body = {"model": cfg["vlm"]["model"], "messages": ([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": content}]}
    body.update(params)
    if cfg["vlm"].get("json_mode") and "response_format" not in body:
        body["response_format"] = {"type": "json_object"}
    key = api_key(cfg)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    if kind(cfg) == "azure":
        base = con["base_url"].rstrip("/")
        url = (f"{base}/openai/deployments/{urllib.parse.quote(cfg['vlm']['model'])}/chat/completions"
               f"?api-version={con.get('api_version') or '2024-10-21'}")
        headers = {"api-key": key}
    else:
        url = _openai_base(con["base_url"]) + "/chat/completions"
    try:
        j = _request(url, body, headers, con.get("timeout_s", 120))
    except ProviderError as e:     # some servers refuse response_format: try once more without it
        if e.status == 400 and "response_format" in body and "response_format" not in params:
            body.pop("response_format")
            j = _request(url, body, headers, con.get("timeout_s", 120))
        else:
            raise
    try:
        msg = j["choices"][0]["message"]
        text = msg.get("content")
        if isinstance(text, list):
            text = "".join(t.get("text", "") for t in text if isinstance(t, dict))
    except (KeyError, IndexError, TypeError):
        raise ProviderError(f"unexpected answer: {str(j)[:200]}", 0, True) from None
    u = j.get("usage") or {}
    return text or "", {"input": int(u.get("prompt_tokens") or 0), "output": int(u.get("completion_tokens") or 0)}


def _gemini_call(cfg, system, parts, params):
    con = cfg["connection"]
    gparts = []
    for p in parts:
        if p[0] == "text":
            gparts.append({"text": p[1]})
        elif p[0] == "image":
            gparts.append({"inline_data": {"mime_type": p[2], "data": _b64(p[1])}})
    gen = {}
    for a, b in (("temperature", "temperature"), ("max_tokens", "maxOutputTokens"), ("top_p", "topP")):
        if a in params:
            gen[b] = params[a]
    gen.update({k: v for k, v in params.items() if k not in ("temperature", "max_tokens", "top_p",
                                                               "response_format")})
    if cfg["vlm"].get("json_mode"):
        gen["responseMimeType"] = "application/json"
    body = {"contents": [{"role": "user", "parts": gparts}], "generationConfig": gen}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    base = (con.get("base_url") or "https://generativelanguage.googleapis.com").rstrip("/")
    model = cfg["vlm"]["model"]
    model = model if model.startswith("models/") else f"models/{model}"
    url = f"{base}/v1beta/{model}:generateContent"
    j = _request(url, body, {"x-goog-api-key": api_key(cfg)}, con.get("timeout_s", 120))
    try:
        cand = j["candidates"][0]
        text = "".join(p.get("text", "") for p in cand["content"]["parts"])
    except (KeyError, IndexError, TypeError):
        reason = (j.get("promptFeedback") or {}).get("blockReason") or str(j)[:200]
        raise ProviderError(f"no answer ({reason})", 0, False) from None
    u = j.get("usageMetadata") or {}
    return text, {"input": int(u.get("promptTokenCount") or 0), "output": int(u.get("candidatesTokenCount") or 0)}


def _anthropic_call(cfg, system, parts, params):
    con = cfg["connection"]
    content = []
    for p in parts:
        if p[0] == "text":
            content.append({"type": "text", "text": p[1]})
        elif p[0] == "image":
            content.append({"type": "image", "source": {"type": "base64", "media_type": p[2], "data": _b64(p[1])}})
        elif p[0] == "cache" and content:
            content[-1]["cache_control"] = {"type": "ephemeral"}
    body = {"model": cfg["vlm"]["model"], "max_tokens": int(params.get("max_tokens", 3000)),
            "messages": [{"role": "user", "content": content}]}
    for k, v in params.items():
        if k not in ("max_tokens", "response_format"):
            body[k] = v
    if system:
        body["system"] = system
    base = (con.get("base_url") or "https://api.anthropic.com").rstrip("/")
    url = base + ("/messages" if base.endswith("/v1") else "/v1/messages")
    j = _request(url, body, {"x-api-key": api_key(cfg), "anthropic-version": "2023-06-01"}, con.get("timeout_s", 120))
    text = "".join(b.get("text", "") for b in j.get("content") or [] if b.get("type") == "text")
    u = j.get("usage") or {}
    return text, {"input": int(u.get("input_tokens") or 0) + int(u.get("cache_read_input_tokens") or 0),
                  "output": int(u.get("output_tokens") or 0)}


def _bedrock_call(cfg, system, parts, params):
    try:
        import boto3
    except ImportError:
        raise ProviderError("AWS Bedrock needs the boto3 package (pip install boto3)", 0, False) from None
    con = cfg["connection"]
    client = boto3.client("bedrock-runtime", region_name=con.get("region") or None)
    content = []
    for p in parts:
        if p[0] == "text":
            content.append({"text": p[1]})
        elif p[0] == "image":
            content.append({"image": {"format": p[2].split("/")[-1].replace("jpg", "jpeg"), "source": {"bytes": p[1]}}})
    inf = {"maxTokens": int(params.get("max_tokens", 3000))}
    if "temperature" in params:
        inf["temperature"] = params["temperature"]
    try:
        r = client.converse(modelId=cfg["vlm"]["model"], messages=[{"role": "user", "content": content}],
                            system=[{"text": system}] if system else [], inferenceConfig=inf)
    except Exception as e:
        name = type(e).__name__
        raise ProviderError(f"Bedrock: {name}: {e}", 0, "Throttl" in name or "Timeout" in name) from None
    text = "".join(c.get("text", "") for c in r["output"]["message"]["content"])
    u = r.get("usage") or {}
    return text, {"input": int(u.get("inputTokens") or 0), "output": int(u.get("outputTokens") or 0)}


_CALLS = {"openai": _openai_call, "azure": _openai_call, "gemini": _gemini_call, "anthropic": _anthropic_call,
          "bedrock": _bedrock_call}


def chat(cfg, system, parts, cancelled=lambda: False, local=None) -> dict:
    """One answer, retried when it makes sense. `local` is the env worker for provider 'local'."""
    k = kind(cfg)
    params = dict((cfg.get("vlm") or {}).get("params") or {})
    retries = max(0, int(cfg["connection"].get("retries", 5)))
    last = None
    for attempt in range(retries + 1):
        if cancelled():
            raise ProviderError("stopped", 0, False)
        t0 = time.monotonic()
        try:
            if k == "local":
                if local is None:
                    raise ProviderError("the local model is not loaded", 0, False)
                text, usage = local.vlm(cfg, system, parts)
            else:
                text, usage = _CALLS[k](cfg, system, parts, params)
            return {"text": text, "usage": usage, "seconds": round(time.monotonic() - t0, 2), "attempts": attempt + 1}
        except ProviderError as e:
            last = e
            if not e.retryable or attempt == retries:
                raise
            wait = e.wait if e.wait is not None else min(60.0, 1.5 * 2 ** attempt) * (0.75 + random.random() / 2)
            end = time.monotonic() + wait
            while time.monotonic() < end:
                if cancelled():
                    raise ProviderError("stopped", 0, False)
                time.sleep(min(0.25, end - time.monotonic()) if end > time.monotonic() else 0)
    raise last


def list_models(cfg, timeout=20) -> list[str]:
    """Model names the service offers (empty when it does not say)."""
    k = kind(cfg)
    con = cfg["connection"]
    key = api_key(cfg)
    if k == "openai":
        j = _request(_openai_base(con["base_url"]) + "/models", None,
                     {"Authorization": f"Bearer {key}"} if key else {}, timeout)
        return sorted({m.get("id", "") for m in j.get("data") or [] if m.get("id")})
    if k == "gemini":
        base = (con.get("base_url") or "https://generativelanguage.googleapis.com").rstrip("/")
        j = _request(f"{base}/v1beta/models?pageSize=1000", None, {"x-goog-api-key": key}, timeout)
        return sorted(m["name"].split("/", 1)[-1] for m in j.get("models") or []
                      if "generateContent" in (m.get("supportedGenerationMethods") or []))
    if k == "anthropic":
        base = (con.get("base_url") or "https://api.anthropic.com").rstrip("/")
        j = _request(base + ("/models" if base.endswith("/v1") else "/v1/models"), None,
                     {"x-api-key": key, "anthropic-version": "2023-06-01"}, timeout)
        return sorted(m.get("id", "") for m in j.get("data") or [])
    if k == "bedrock":
        try:
            import boto3
            c = boto3.client("bedrock", region_name=con.get("region") or None)
            return sorted(m["modelId"] for m in c.list_foundation_models(byInputModality="IMAGE")["modelSummaries"])
        except Exception as e:
            raise ProviderError(f"Bedrock: {e}") from None
    return []


def test_connection(cfg) -> dict:
    """{"ok", "message", "models", "model_listed", "ms"} — reachable, key accepted, model offered."""
    t0 = time.monotonic()
    name = cfg["connection"].get("api_key_env") or ""
    if name and not api_key(cfg) and kind(cfg) not in ("bedrock", "local"):
        return {"ok": False, "message": f"No key: the environment variable {name} is not set.", "models": [],
                "model_listed": None, "ms": 0}
    try:
        models = list_models(cfg)
    except ProviderError as e:
        return {"ok": False, "message": str(e), "models": [], "model_listed": None,
                "ms": int(1000 * (time.monotonic() - t0))}
    want = cfg["vlm"].get("model", "")
    listed = None if not models or not want else (want in models or want.split("/")[-1] in
                                                 {m.split("/")[-1] for m in models})
    return {"ok": True, "message": f"Connected · {len(models)} models offered" if models else "Connected",
            "models": models, "model_listed": listed, "ms": int(1000 * (time.monotonic() - t0))}
