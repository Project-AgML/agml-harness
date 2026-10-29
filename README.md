# agml-agent

Agentic toolkit for agricultural ML grounding and dataset discovery, exposed via MCP.

- **`retrieve`** — live grounding text/images for a disease, pest, species, or quality class, from external sources (EPPO, GBIF, UC IPM/APS, Bugwood, USDA).
- **`search_agml`** — search or look up datasets in the AgML catalog, including class labels, licenses, and model benchmark results.

## Install

```bash
uvx --from git+https://github.com/Project-AgML/agml-agent agml-agent-mcp
```

**Before adding this to an MCP client, warm the cache once:**

```bash
uvx --from git+https://github.com/Project-AgML/agml-agent agml-agent-search "coffee rust"
```

The `agml` package pulls in a heavy dependency tree (opencv, pandas, matplotlib, scipy, etc.) with no lighter option available — first install can take several minutes. MCP clients enforce a short connection timeout (Claude Code: 30s) and will report a failed/timed-out connection on a cold cache, even though the install would succeed given more time. Running the command above once, directly in a terminal, downloads everything and prints real search results when done — a clear success signal, unlike running `agml-agent-mcp` directly (which, once started, silently waits forever for MCP protocol input that a bare terminal never sends — expected behavior for a server, but useless for confirming the pre-warm actually worked).

## Usage

### MCP clients (Claude Code, Claude Desktop, Cursor, etc.)

Claude Code (`-s user` makes it available in every project, not just the current one):

```bash
claude mcp add agml-agent -s user -- uvx --from git+https://github.com/Project-AgML/agml-agent agml-agent-mcp
```

Claude Desktop / any other MCP-compatible client:

```json
{
  "mcpServers": {
    "agml-agent": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/Project-AgML/agml-agent", "agml-agent-mcp"]
    }
  }
}
```

### Ollama

Ollama has no MCP client support, so `agml-agent-chat` runs as a standalone bridge process instead of a client config: it connects to your Ollama server, translates the MCP tool schemas into Ollama's tool-calling format, and forwards tool calls back through MCP.

```bash
uvx --from git+https://github.com/Project-AgML/agml-agent agml-agent-chat --host http://localhost:11434
```

### CLI

```bash
uvx --from git+https://github.com/Project-AgML/agml-agent agml-agent-retrieve "Coffee leaf rust" --crop coffee --sci "Hemileia vastatrix"
uvx --from git+https://github.com/Project-AgML/agml-agent agml-agent-search "citrus disease" --ml-task image_classification
```

### Python

```python
from agml_agent.retrieve import retrieve
from agml_agent.search_agml import search_agml

retrieve("Coffee leaf rust", crop="coffee", scientific_name="Hemileia vastatrix")
search_agml("citrus disease", ml_task="image_classification")
```

## Tools

### `retrieve(class_name, crop="", task="", scientific_name=None, aliases=None, sources=None)`

Returns `{source_name: [chunk, ...]}`. Each chunk:

| field | type |
|---|---|
| `class_name` | str |
| `tags` | list[str] |
| `excerpt_text` | str |
| `raw_response` | str (JSON) |
| `source_url` | str |
| `license` | str |
| `accessed_date` | str |
| `extra` | dict |

`task`: `disease` \| `pest` \| `species` \| `quality`.
`sources`: restrict to specific source names (default: all active sources).

**Sources**: EPPO, UC IPM/APS, Bugwood, GBIF, USDA NALT, USDA PLANTS, USDA AMS Grade Standards. Configured in `src/agml_agent/config/sources.yaml`.

### `search_agml(query="", exact_name=None, ml_task=None, ag_task=None, limit=10)`

Returns `{"results": [...], "total_matched": N}`. Keyword search returns trimmed summaries; `exact_name` returns full metadata plus benchmark results.

`ml_task`: `image_classification` \| `object_detection` \| `semantic_segmentation`.
`ag_task`: e.g. `disease_classification`, `weed_detection`, `quality_classification`.

## Benchmarking (AGgentic-benchmarks)

The harness is scored with agbench, unchanged: items, prompts, sampling and effort settings, grading and the judge
all come from the pinned agbench commit (`[tool.uv.sources]` in `pyproject.toml`), so a harness run is directly
comparable with the plain-model baselines. agbench is a private repo, so it is an optional extra that needs GitHub
SSH access:

```bash
uv sync --extra bench
uv run agml-agent-bench --model google/gemma-4-12B-it --data-dir /path/with/base-data --judge
uv run agml-agent-bench ... --only '^(bean_disease_uganda|AgriBench)$' --limit-per-dataset 2   # pilot
uv run agml-agent-bench ... --run-dir runs/<run>                                                # resume
```

- **Policy ("exploit"):** one pass with tools available, at most `--max-tool-calls` calls (default 4), agbench's
  prompt unchanged. The model's server is started with tool calling on (`config/tool_parsers.yaml`), or pass
  `--endpoint` for one already running with tools on.
- **Leakage:** only external-database tools are allowed (`retrieve`); `search_agml` is refused because AgML class
  lists and benchmark results would leak labels. `retrieve` results reach the model as excerpt text and source
  URLs only, never images.
- **The `retrieve` tool** takes a `description` of what the model sees and/or up to 4 `candidates` (name,
  scientific name, aliases), plus `crop` and `task`. A description searches UC IPM and Bugwood fact sheets
  (site-restricted, never the open web) and returns pages with matching features, those mentioning the crop first and
  then pages about other hosts (the crop never filters pages out); candidates are looked
  up by name in every database (EPPO, UC IPM/APS, Bugwood, GBIF, USDA), with long excerpts narrowed to the
  sentences matching the description. `--retrieve-mode candidates` offers name checks only (no description
  search), to compare the two. Lookups are cached in `<out>/retrieve_cache.jsonl` across runs
  (`--retrieve-cache`, `--no-retrieve-cache`), so repeated classes are fast and every model sees the same text.
- **Budget:** `max_tokens` (agbench's) caps completion tokens summed over all turns; tool results count only
  toward the context window.
- **Output:** a standard agbench run dir plus `traces.jsonl` (every model turn, tool call, arguments, result and
  latency per item).

### Viewing results

Scores, next to the plain-model baselines, in agbench's viewer:

```bash
uv run agbench eval view --runs <baseline run> runs/<harness run> --data-dir /path
```

Tool use and traces in the harness viewer (`agml-agent-view`, Streamlit, http://localhost:8503):

```bash
uv run agml-agent-view --runs runs/<harness run> --baseline <baseline run> --data-dir /path
```

- **Tool use:** how often each tool was called, overall and per task; per baseline, which items the tools fixed and
  which they broke.
- **Traces:** per item, the image, prompt, ground truth, the harness and baseline answers with scores, and every model
  turn: reasoning, each tool call with its arguments, and the tool result with its latency. Filter by run, dataset,
  format, tool use and fixed/broken.

`--runs` takes run dirs or folders of them (several harness runs can be compared); `--data-dir` is only needed when
images live elsewhere than when the runs were made.

## Updating / removing

`uvx --from git+...` resolves the latest commit on the default branch on every launch — each new session picks up the newest version automatically, no reinstall needed. To pin a stable version instead, append `@<tag>` to the URL, e.g. `git+https://github.com/Project-AgML/agml-agent@v0.1.0`.

To remove:

```bash
claude mcp remove agml-agent
```

For other MCP clients, delete the `agml-agent` entry from the client's MCP config.

## Configuration

```bash
cp .env.example .env
```

| variable | required for | notes |
|---|---|---|
| `EPPO_API_KEY` | `retrieve` (EPPO source only) | free — [register](https://data.eppo.int/ui/#/user/register) |
| `OLLAMA_HOST` | `agml-agent-chat` | defaults to `http://localhost:11434` |

## Development

```bash
git clone https://github.com/Project-AgML/agml-agent
cd agml-agent
uv sync
uv run agml-agent-mcp
uv sync --extra bench && uv run pytest   # tests (benchmark code needs the bench extra)
```

## License

Apache 2.0
