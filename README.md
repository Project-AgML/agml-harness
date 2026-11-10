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

**Sources**: EPPO, UC IPM/APS, Bugwood, GBIF, USDA NALT, USDA PLANTS, USDA AMS Grade Standards, and open sources with
no key: TNAU Agritech, APS Plant Disease Lessons, PNW and Texas plant disease handbooks, NC State Extension, ProMusa,
PlantVillage, Feedipedia, Wikipedia, iNaturalist, Catalogue of Life, EOL. Configured in `src/agml_agent/config/sources.yaml`
(fact-sheet sites share one fetcher, `sources/factsheets.py`, configured per site with `args`).

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
  lists and benchmark results would leak labels. Text about datasets or machine learning is dropped from every source. `retrieve` results reach the model as excerpt text and source
  URLs only, never images.
- **The `retrieve` tool** takes a `description` of what the model sees and/or up to 4 `candidates` (name,
  scientific name, aliases), plus `crop` and `task` (both optional; they order results, they never filter: every
  database is searched). Candidates are looked up by name in every database, with long excerpts narrowed to the
  sentences matching the description. A description alone searches fact-sheet sites (site-restricted, never the open
  web) and Wikipedia. Lookups are cached in `<out>/retrieve_cache.jsonl` across runs (`--retrieve-cache`,
  `--no-retrieve-cache`), so repeated classes are fast and every model sees the same text. A lookup keeps the sources
  that answer within 15 s.
- **Modes** (`--retrieve-mode`):
  - `description+candidates` (default), `candidates` (name checks only);
  - `crosscheck`: every call gives a description of the photo and an initial diagnosis (up to 4 candidates);
  - `crosscheck-healthy`: also reminds the model, in the tool description and result, that the plant may have no problem;
  - `crosscheck-assess`: first says whether anything is wrong; candidates are not needed if nothing is;
  - `crosscheck-look-first`: answers with no tool offered, then sees that answer and may cross-check it
    (classification and image-text items; detection and segmentation stop at the first answer);
  - `crosscheck-placebo`, `crosscheck-look-first-placebo`: the same flows with no reference text and no lookups, to
    separate what the process does from what the databases add.
- **Reference text options:**
  - `--clean-text 1`: appearance text first, then identification, then taxonomy (text on the crop first); drops
    disambiguation pages, site boilerplate, Wikipedia pages naming no living thing, and USDA grades unless the task
    is quality. `--clean-text 2` also shows Wikipedia's appearance sections (Symptoms, Description...) and shrinks
    taxonomy to name lines (scientific name, common names, synonyms).
  - `--check-options`: on multiple choice, every answer option is looked up, then the model's own candidates (up to 6).
  - `--tidy-names`: candidate names are cleaned before lookup (dataset-label form `Crop___Disease`, brackets as
    aliases, underscores and hyphens, a trailing plant part, a repeated first word); the name as given stays an alias.
  - `--crop-from-prompt`: a call without `crop` gets the item's crop when the prompt names it.
  - `--keep-healthy` (`crosscheck-look-first`): a first answer that finds no problem gets a second look without the
    tool, kept unless the image clearly shows a problem (vivid symptom text talked healthy plants into problems).
  - With `--clean-text 2`, EPPO, GBIF and Catalogue of Life names and hosts for a disease's common name (e.g.
    "Fusarium wilt") are shown only if the text names the candidate: a name lookup can land on another host's
    disease.
  - `--kb DIR`: also search the local knowledge base (below); its sections come first.
  - `--kb-only` (with `--kb`): the local knowledge base and its names table are the only sources; no API, web search
    or cache, and connections off the machine raise.
  - `--code-tool`: every task also gets `run_python(code)` next to `retrieve` after the first answer, and the model
    chooses (neither, either, or both in any order, same call budget); detection and segmentation then get the check
    step too (without it they stop at the first answer). `run_python` runs Python on the images the model saw (`img`,
    `imgs`, `W`, `H`, `np`, `Image`, `show()`; numpy, PIL, cv2, skimage, scipy). `print()` output and up to 2 images per
    call from `show()` come back (8 images per prompt in all). The tool description says what is available, not how to
    use it. Sandbox (`sandbox.py`): a fresh `python -I` process per call in a temp folder holding only the item's
    images, 60 s, 3 GB, no network, no other files, imports limited to those libraries (no pretrained models), code
    reaching for `os`, `subprocess`, `__import__`, `eval` etc. refused. Shown
    images are saved in `<run>/tool_images/`; the viewer's Traces page shows the code, output and images.
  - Pages from sites whose robots.txt blocks AI crawlers (PlantVillage, PNW Handbooks, APS, ProMusa) are never shown,
    even from the cache.
- **Budget:** `max_tokens` (agbench's) caps completion tokens summed over all turns; tool results count only
  toward the context window.
- **Output:** a standard agbench run dir plus `traces.jsonl` (every model turn, tool call, arguments, result and
  latency per item).

### Local knowledge base

Extension fact sheets describe what a problem looks like crop by crop, but most sites have no search API and web
search is rate-limited. `agml-agent-kb` downloads them once, splits each page into sections and indexes them with
SQLite full-text search, so lookups take milliseconds and every run sees the same text:

```bash
uv run agml-agent-kb --kb kb crawl                  # download (hours: robots.txt crawl delays); resumable
uv run agml-agent-kb --kb kb crawl --site ucipm --max-pages 50
uv run agml-agent-kb --kb kb build                  # extract sections, write docs.jsonl.gz, kb.sqlite, manifest.json
uv run agml-agent-kb --kb kb search downy mildew --crop cucumber
uv run agml-agent-bench ... --kb kb                 # use it in a benchmark run
```

- **Sites** (`config/kb_sites.yaml`): UC IPM, NC State Extension, Texas Plant Disease Handbook, Feedipedia. Only
  sites whose robots.txt allows us; PlantVillage, PNW Handbooks, APS and ProMusa block AI crawlers and are left out.
  The crawler obeys robots.txt and Crawl-delay (at least 1 s between requests per site) and downloads HTML only (no
  images, no PDFs).
- **Layout:** `raw/<site>/pages.jsonl` (every request) and `raw/<site>/html/<sha1>.html.gz` (pages as downloaded, so
  extraction can be redone without re-crawling); `docs.jsonl.gz` (one section per line: site, url, title, heading,
  kind, text); `kb.sqlite` (FTS5 index, BM25); `manifest.json` (counts, build date).
- **Sections:** split at the page's headings, long ones at paragraph breaks (about 1,500 characters); `kind` from the
  heading (symptom, life_cycle, management, other); repeated blocks (navigation) dropped; pages about datasets or
  machine learning dropped (leakage).
- **Search:** a candidate's name (then scientific name, then aliases) as a phrase; matches in the title or heading
  first, then appearance sections, then text mentioning the crop (ordering only), then BM25; at most 2 sections per
  page.
- **On farm:** build locally and copy the folder (`rsync -a kb/ farm:<repo>/kb/`); the compute nodes don't need
  internet for it. `kb/` is git-ignored: the sites' text is for research use and is not redistributed.

### Viewing results

Scores, next to the plain-model baselines, in agbench's viewer:

```bash
uv run agbench eval view --runs <baseline run> runs/<harness run> --data-dir /path
```

Tool use and traces in the harness viewer (`agml-agent-view`, Streamlit, http://localhost:8503):

```bash
uv run agml-agent-view --runs runs/<harness run> --baseline <baseline run> --data-dir /path
```

- **Knowledge base:** every section the local knowledge base returned, with relevance checks (the page title or
  heading names the candidate; the page is about exactly the candidate; the crop is named; the heading is about
  appearance; the candidate is the correct class) and use (the model's later reasoning or answer repeats two 4-word
  phrases from it that it had not written before); per run, the share of each, and per source, how often its text is
  used. Filter the sections by these checks to read examples.
- **Tool use:** how often each tool was called, overall and per task; per baseline, which items the tools fixed and
  which they broke.
- **Sources:** per source, the items where its text was shown to the model and where the model used it (repeated a
  phrase or page title from it), with fixed / broken counts against the baseline, next to the noise level (items with no
  tool call).
- **Traces:** per item, the image, prompt, ground truth, the harness and baseline answers with scores, and every model
  turn: reasoning, each tool call with its arguments, and the tool result with its latency. Filter by run, dataset,
  format, tool use, fixed/broken and source.

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
