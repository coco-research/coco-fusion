# coco-fusion (public)

BYO-key multi-model **persona fusion council** for CoCo. Bring your own API keys and run a panel of expert personas across distinct model architectures, then synthesise a single advisory verdict.

## Features

- **Fusion policy** — deterministic `should_fuse()` gate by surface, stakes, and load
- **Lane registry** — OpenAI-compatible and Anthropic-direct lanes from env vars or config
- **Council orchestration** — persona × lane round-robin, extract-then-merge synthesis, coverage-repair
- **Providers** — OpenRouter, OpenAI, Anthropic, DeepSeek, Kimi/Moonshot, local Ollama/LM Studio

## Install

```bash
cd public
pip install -e .
```

## Configure lanes

Set API key env vars for the lanes you want. Lanes without a key are skipped.

| Lane preset | Env vars |
|-------------|----------|
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Kimi (Moonshot) | `MOONSHOT_API_KEY` |
| Local (Ollama/LM Studio) | `LOCAL_LLM_BASE_URL` (e.g. `http://127.0.0.1:11434/v1`) |

Optional overrides: `OPENROUTER_BASE_URL`, `DEEPSEEK_BASE_URL`, `MOONSHOT_BASE_URL`, model IDs via `COCO_FUSION_*_MODEL` env vars (see `lane_registry.py`).

### Custom lanes (config dict)

```python
from coco_fusion import Lane, build_lanes_from_config

lanes = build_lanes_from_config([
    {
        "lane_id": "my-gpt",
        "provider": "openai_compat",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model_id": "gpt-4o",
        "architecture_family": "gpt",
        "tier": "frontier",
        "rank": 0,
    },
])
```

## Quick start

```python
from coco_fusion import run_council

result = run_council(
    "Should we adopt event sourcing for the billing service?",
    stakes="high",
)
print(result.verdict)
print(result.consensus, result.dissent)
```

## Architecture families

Fusion picks **one lane per architecture family** for error decorrelation:

`claude`, `gpt`, `gemini`, `llama`, `deepseek`, `local`

High stakes → frontier tier + high reasoning effort. Low stakes → budget tier.

## License

Open-source public profile — no enterprise coupling.
