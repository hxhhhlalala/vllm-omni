# Community Recipes

This directory contains community-maintained recipe templates for answering a
practical user question:

> How do I run model X on hardware Y for task Z?

We suggest organizing recipes by model vendor, aligned with
[`vllm-project/recipes`](https://github.com/vllm-project/recipes), and using
one Markdown file per model family by default.

Example layout:

```text
recipes/
  Qwen/
    Qwen3-Omni.md
    Qwen3-TTS.md
  Tencent-Hunyuan/
    HunyuanVideo.md
```

Within a single recipe file, include different hardware support sections such
as `GPU`, `ROCm`, and `NPU`, and add concrete tested configurations like
`1x A100 80GB` or `2x L40S` inside those sections when applicable.

See [TEMPLATE.md](./TEMPLATE.md) for the recommended format.
