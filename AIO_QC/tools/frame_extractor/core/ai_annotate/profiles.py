# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Providers (where a model runs) and model profiles (how a model family writes boxes) — Qt-free.

A profile is chosen from the model's name; "box_format: auto" in a project means "the profile's". When a model is
not listed, or a family changes its convention, the app's "Detect box format" asks the model to box an object in an
example image whose boxes are known and keeps the format that matches best (calibrate in runner).
"""
from __future__ import annotations

import re

# key: (title, api kind, default address, default key variable, note)
PROVIDERS = {
    "litellm": ("LiteLLM proxy", "openai", "http://localhost:4000", "LITELLM_API_KEY",
                "Your team's proxy in front of many models (OpenAI-compatible)."),
    "openai": ("OpenAI", "openai", "https://api.openai.com/v1", "OPENAI_API_KEY", "GPT-4o, GPT-4.1, GPT-5 …"),
    "azure": ("Azure OpenAI", "azure", "https://YOUR-RESOURCE.openai.azure.com", "AZURE_OPENAI_API_KEY",
              "The model box takes your deployment name."),
    "gemini": ("Google Gemini", "gemini", "https://generativelanguage.googleapis.com", "GEMINI_API_KEY",
               "Gemini's own API (best for its built-in box detection)."),
    "anthropic": ("Anthropic Claude", "anthropic", "https://api.anthropic.com", "ANTHROPIC_API_KEY",
                  "Claude models."),
    "bedrock": ("AWS Bedrock", "bedrock", "", "",
                "Uses your AWS credentials (aws configure); needs the boto3 package. Region below."),
    "openrouter": ("OpenRouter", "openai", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                   "Many providers' models through one key."),
    "dashscope": ("Alibaba Qwen (DashScope)", "openai", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                  "DASHSCOPE_API_KEY", "Qwen-VL models."),
    "together": ("Together AI", "openai", "https://api.together.xyz/v1", "TOGETHER_API_KEY", "Open models (Llama, Qwen …)."),
    "groq": ("Groq", "openai", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "Fast Llama vision models."),
    "mistral": ("Mistral", "openai", "https://api.mistral.ai/v1", "MISTRAL_API_KEY", "Pixtral, Mistral Medium …"),
    "xai": ("xAI Grok", "openai", "https://api.x.ai/v1", "XAI_API_KEY", "Grok vision models."),
    "vllm": ("vLLM server", "openai", "http://localhost:8000/v1", "", "An open model you serve yourself (no key)."),
    "ollama": ("Ollama (this computer)", "openai", "http://localhost:11434/v1", "",
               "Models pulled with ollama (no key)."),
    "lmstudio": ("LM Studio (this computer)", "openai", "http://localhost:1234/v1", "", "No key."),
    "custom": ("Other OpenAI-compatible service", "openai", "", "", "Any address that speaks the OpenAI API."),
    "local": ("Model in my Python environment", "local", "", "",
              "A Hugging Face vision model run by your Python environment (Settings ▸ Models); no API, no key."),
}

# (pattern on the model name, family, box format, notes, extra)
_PROFILES = [
    (r"gemini", "Gemini", "yxyx_1000", "Built-in box detection ([ymin, xmin, ymax, xmax], 0–1000).", {}),
    (r"qwen3[-_.]?vl|qwen3-vl|qwen-vl-max-latest", "Qwen3-VL", "xyxy_1000", "Built-in grounding, 0–1000.", {}),
    (r"qwen2(\.5)?[-_.]?vl|qwen-vl", "Qwen2.5-VL", "xyxy_px",
     "Built-in grounding in pixels of the image it sees — the image is sent at a size the model keeps (multiples "
     "of 28).", {"round_to": 28, "max_side": 1288}),
    (r"internvl", "InternVL", "xyxy_1000", "Grounding in 0–1000.", {}),
    (r"glm-?4(\.\d)?v|glm-4\.5v|glm-?4v", "GLM-4V", "xyxy_1000", "Grounding in 0–1000.", {}),
    (r"deepseek[-_]?vl", "DeepSeek-VL2", "xyxy_1000", "Grounding in 0–1000.", {}),
    (r"minicpm", "MiniCPM-V", "xyxy_1000", "Grounding in 0–1000.", {}),
    (r"kimi[-_]?vl", "Kimi-VL", "xyxy_1000", "Boxes as asked (0–1000).", {}),
    (r"molmo", "Molmo", "xyxy_1000", "Points rather than boxes: best for questions, or with a detector.", {}),
    (r"pixtral|mistral", "Mistral / Pixtral", "xyxy_1000", "Boxes as asked; approximate — a detector + check gives "
                                                          "tighter boxes.", {}),
    (r"llama|llava", "Llama vision", "xyxy_1000", "Boxes as asked; approximate.", {}),
    (r"claude", "Claude", "xyxy_1000", "Boxes as asked; good at reading and checking, approximate boxes — "
                                       "a detector + check gives tighter boxes.", {}),
    (r"gpt|o\d|chatgpt", "GPT", "xyxy_1000", "Boxes as asked; good at reading and checking, approximate boxes — "
                                             "a detector + check gives tighter boxes.", {}),
    (r"grok", "Grok", "xyxy_1000", "Boxes as asked; approximate.", {}),
    (r"paligemma", "PaliGemma", "yxyx_1000", "Local: <loc> tokens (read for you).", {"local_kind": "paligemma"}),
    (r"florence", "Florence-2", "xyxy_1000", "Local: use it as the detector (text prompts).", {}),
    (r"moondream", "Moondream", "xyxy_norm", "Local: its own detect call.", {"local_kind": "moondream"}),
]


def profile(model: str) -> dict:
    m = (model or "").lower().split("/")[-1]
    for pat, fam, fmt, note, extra in _PROFILES:
        if re.search(pat, m):
            return {"family": fam, "box_format": fmt, "note": note, **extra}
    return {"family": "Other", "box_format": "xyxy_1000",
            "note": "Not a known family: boxes are asked as [xmin, ymin, xmax, ymax] 0–1000. Use “Detect box format” "
                    "to make sure."}


def box_format(cfg) -> str:
    f = (cfg.get("vlm") or {}).get("box_format", "auto")
    return profile(cfg["vlm"].get("model", ""))["box_format"] if f in ("", "auto") else f


def provider_of(cfg) -> tuple:
    return PROVIDERS.get(cfg["connection"].get("provider") or "custom", PROVIDERS["custom"])
