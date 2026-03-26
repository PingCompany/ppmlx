# ppmlx

**Run LLMs on your Mac.** OpenAI-compatible API powered by Apple Silicon.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

## Install

```bash
pip install ppmlx
```

> Requires macOS on Apple Silicon (M1+) and Python 3.11+

## Get Started

```bash
ppmlx pull qwen3.5:9b      # download a model
ppmlx run qwen3.5:9b       # chat in the terminal
ppmlx serve                 # start API server on :6767
```

That's it. Any OpenAI-compatible tool works out of the box:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:6767/v1", api_key="local")
response = client.chat.completions.create(
    model="qwen3.5:9b",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Commands

```
ppmlx launch           Interactive launcher (pick action + model)
ppmlx pull <model>     Download a model
ppmlx run <model>      Interactive chat
ppmlx serve            Start API server
ppmlx list             Show downloaded models
ppmlx rm <model>       Remove a model
ppmlx ps               Show loaded models & memory
```

## Connect Your Tools

Point any OpenAI-compatible client at `http://localhost:6767/v1` with any API key:

- **Cursor** — Settings > AI > OpenAI-compatible
- **Continue** — config.json: provider `openai`, apiBase above
- **LangChain / LlamaIndex** — set `base_url` and `api_key="local"`

## Config

Optional. `~/.ppmlx/config.toml`:

```toml
[server]
host = "127.0.0.1"
port = 6767

[generation]
temperature = 0.7
max_tokens = 2048
```

## License

MIT
