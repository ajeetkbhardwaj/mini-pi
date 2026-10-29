# mini-pi

A modular, lightweight terminal coding agent inspired by Pi, written in Python with Pydantic v2, Rich, and Textual.

## Architecture

```
mini-pi/
├── src/mini_pi/
│   ├── agent/        # Core agent loop, context builder, and message types (with ThinkingBlock)
│   ├── tools/        # Pydantic tools (read, write, edit, bash) & registry
│   ├── session/      # JSONL persistence, branching session tree, compaction
│   ├── providers/    # Local OpenAI-compatible (LM Studio/Ollama/llama.cpp) & Cloud adapters
│   ├── extensions/   # Event hooks, extension runtime & safety guardrails
│   ├── skills/       # SKILL.md progressive disclosure & prompt templates
│   ├── tui/          # Rich streaming console & full Textual interactive TUI
│   └── cli.py        # CLI entry point (headless, REPL, and interactive TUI)
├── prompts/          # Reusable prompt templates (/review, /debug)
├── skills/           # Custom agent skills (e.g. release workflow)
└── tests/            # 29 unit & integration tests
```

## Features

- **Live Streaming Execution**:
  - Full real-time streaming for both reasoning/thinking and text generation.
  - Streaming thinking panels display the model's chain-of-thought token-by-token as it reasons.
  - Streaming assistant panels show the response in real time without blocking.
- **Visible Thinking & Reasoning Process**:
  - Automatically captures and separates chain-of-thought thinking tokens from local models (DeepSeek-R1, QwQ, Qwen2.5-Coder) and cloud models (Claude extended thinking, OpenAI o1/o3-mini).
  - Automatically parses `<think>...</think>` tags and `reasoning_content` API fields.
  - Displays thinking in dedicated stylish `💭 Thinking / Reasoning` panels so the user sees what the agent is reasoning before taking tool actions or delivering answers.
  - Live token metrics: prompt tokens, completion tokens, reasoning/thinking tokens, and total processed tokens.
- **Interactive Textual TUI (`mini-pi --tui`)**:
  - Live connection status to local model backends (LM Studio, Ollama, llama.cpp).
  - Quick model and provider switcher modal (`Ctrl+M`).
  - Command palette / help reference (`Ctrl+K`).
  - Slash command auto-completion (`/` suggestions).
  - Working tree live git diff viewer (`/diff`).
  - Collapsible tool execution blocks and Markdown rendering.
  - Interactive confirmation modal for safety-sensitive operations.
- **First-Class Local Models**:
  - **LM Studio** preset: `http://localhost:1234/v1`
  - **Ollama** preset: `http://localhost:11434/v1`
  - **llama.cpp** preset: `http://localhost:8080/v1`
  - Dynamic model discovery from `/v1/models`.
- **Pydantic Tooling**: Strongly typed input schemas for `read`, `write`, `edit`, and `bash`.
- **Safety Pipeline**: Workspace directory isolation, protected sensitive file rules (`.git`, `.env`), and confirmation gates.
- **Session Durability &**
- **Branching**: JSONL session log, branching tree navigation (`/tree`, `/branch <id>`), and context compaction (`/compact`).
- **Progressive Skills**: Automatic discovery of `SKILL.md` workflows and templates (`/prompts`).

---

## TUI Keyboard Shortcuts

| Shortcut                | Description                                       |
| ----------------------- | ------------------------------------------------- |
| `Ctrl+M`              | Open interactive Model & Provider switcher dialog |
| `Ctrl+K`              | Display Slash Commands & hotkeys reference        |
| `Ctrl+L`              | Clear chat log view                               |
| `Ctrl+R`              | Probe & refresh connection to local server        |
| `Ctrl+C` / `Ctrl+Q` | Exit TUI                                          |

---

## Interactive Slash Commands

Type `/` in the prompt input to activate auto-suggestions:

| Command              | Description                                                                   |
| -------------------- | ----------------------------------------------------------------------------- |
| `/tokens`          | Display a breakdown table of input, output, and thinking tokens               |
| `/models`          | Query and list all models loaded or available on the local server             |
| `/model <name>`    | Switch active model on the fly (e.g.`/model qwen2.5-coder:7b`)              |
| `/provider <name>` | Switch backend provider (`lmstudio`, `ollama`, `llamacpp`, etc.)        |
| `/connect <url>`   | Change base URL & test connection (e.g.`/connect http://localhost:1234/v1`) |
| `/probe`           | Test connectivity to local backend and report status                          |
| `/diff`            | Display color-coded git diff of working tree changes                          |
| `/skills`          | List available skills from`skills/` and `~/.mini-pi/skills/`              |
| `/skill <name>`    | Inject a skill workflow into the next agent prompt                            |
| `/prompts`         | List available prompt templates                                               |
| `/prompt <name>`   | Expand prompt template (e.g.`/prompt review`)                               |
| `/compact`         | Condense earlier conversation history into a structured summary               |
| `/tree`            | View branching conversation tree with active leaf indicators                  |
| `/branch <id>`     | Switch conversation branch to another message leaf                            |
| `/clear-session`   | Clear conversation history and reset token metrics                            |
| `/clear`           | Clear the visible chat screen                                                 |
| `/exit`            | Exit mini-pi                                                                  |

---

## Quickstart

### Run with Conda Environment (`aisystem`)

1. **Interactive Textual TUI**:

   ```bash
   mini-pi --tui
   ```
2. **Run with a specific local backend**:

   ```bash
   # Connect to LM Studio
   mini-pi --tui --provider lmstudio

   mini-pi --tui --provider lmstudio --model spark-x2.6-4b

   # Connect to Ollama
   mini-pi --tui --provider ollama --model qwen2.5-coder
   ```
3. **Command-Line REPL**:

   ```bash
   mini-pi
   ```
4. **Single-prompt execution**:

   ```bash
   mini-pi "Read README.md and describe the project"
   ```
5. **Run test suite**:

   ```bash
   pytest mini-pi/tests -v
   ```
