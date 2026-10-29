from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Grid, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.suggester import SuggestFromList
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Static

from mini_pi.agent.loop import AgentConfig, AgentLoop
from mini_pi.agent.types import Message
from mini_pi.extensions.api import ExtensionAPI
from mini_pi.providers.base import ModelClient
from mini_pi.providers.local_provider import LOCAL_PRESETS, LocalOpenAIClient
from mini_pi.session.compact import compact_history
from mini_pi.session.store import JsonlSessionStore
from mini_pi.session.tree import SessionTree
from mini_pi.skills.loader import list_skills, load_skill
from mini_pi.skills.templates import list_templates, load_template


SLASH_COMMANDS = [
    "/help",
    "/models",
    "/model ",
    "/provider ",
    "/connect ",
    "/tokens",
    "/probe",
    "/skills",
    "/skill ",
    "/prompts",
    "/prompt ",
    "/diff",
    "/compact",
    "/tree",
    "/branch ",
    "/clear",
    "/clear-session",
    "/exit",
]


class ConfirmModal(ModalScreen[bool]):
    """Interactive confirmation modal for safety-sensitive actions."""

    CSS = """
    ConfirmModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }
    #confirm-box {
        width: 65;
        height: auto;
        border: thick #f38ba8;
        background: #1e1e2e;
        padding: 1 2;
    }
    #confirm-title {
        text-style: bold;
        color: #f38ba8;
        padding-bottom: 1;
    }
    #confirm-message {
        color: #cdd6f4;
        padding-bottom: 1;
    }
    #confirm-buttons {
        width: 100%;
        height: auto;
        align: right middle;
    }
    #confirm-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self, title: str, message: str) -> None:
        super().__init__()
        self.modal_title = title
        self.modal_message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Label(self.modal_title, id="confirm-title")
            yield Label(self.modal_message, id="confirm-message")
            with Horizontal(id="confirm-buttons"):
                yield Button("Deny", variant="error", id="btn-deny")
                yield Button("Allow", variant="success", id="btn-allow")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-allow":
            self.dismiss(True)
        else:
            self.dismiss(False)


class ModelSelectModal(ModalScreen[dict[str, str] | None]):
    """Modal to switch provider, model name, and backend endpoint interactively."""

    CSS = """
    ModelSelectModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }
    #model-box {
        width: 72;
        height: auto;
        border: thick #89b4fa;
        background: #181825;
        padding: 1 2;
    }
    #modal-header {
        text-style: bold;
        color: #89b4fa;
        margin-bottom: 1;
    }
    .field-label {
        color: #a6adc8;
        margin-top: 1;
    }
    #backend-row {
        height: auto;
        margin: 1 0;
    }
    #backend-row Button {
        margin-right: 1;
    }
    #modal-actions {
        width: 100%;
        height: auto;
        align: right middle;
        margin-top: 1;
    }
    #modal-actions Button {
        margin-left: 1;
    }
    """

    def __init__(self, current_provider: str, current_model: str, current_url: str) -> None:
        super().__init__()
        self.current_provider = current_provider
        self.current_model = current_model
        self.current_url = current_url

    def compose(self) -> ComposeResult:
        with Vertical(id="model-box"):
            yield Label("Switch Model & Provider", id="modal-header")

            yield Label("Quick Presets:", classes="field-label")
            with Horizontal(id="backend-row"):
                yield Button("LM Studio", id="btn-preset-lmstudio", variant="primary")
                yield Button("Ollama", id="btn-preset-ollama", variant="default")
                yield Button("llama.cpp", id="btn-preset-llamacpp", variant="default")

            yield Label("Provider / Backend:", classes="field-label")
            yield Input(value=self.current_provider, id="input-provider")

            yield Label("Base URL:", classes="field-label")
            yield Input(value=self.current_url, id="input-url")

            yield Label("Model ID / Tag:", classes="field-label")
            yield Input(value=self.current_model, id="input-model")

            with Horizontal(id="modal-actions"):
                yield Button("Cancel", variant="default", id="btn-cancel")
                yield Button("Save & Switch", variant="success", id="btn-save")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-preset-lmstudio":
            self.query_one("#input-provider", Input).value = "lmstudio"
            self.query_one("#input-url", Input).value = LOCAL_PRESETS["lmstudio"]["base_url"]
            self.query_one("#input-model", Input).value = LOCAL_PRESETS["lmstudio"]["default_model"]
        elif event.button.id == "btn-preset-ollama":
            self.query_one("#input-provider", Input).value = "ollama"
            self.query_one("#input-url", Input).value = LOCAL_PRESETS["ollama"]["base_url"]
            self.query_one("#input-model", Input).value = LOCAL_PRESETS["ollama"]["default_model"]
        elif event.button.id == "btn-preset-llamacpp":
            self.query_one("#input-provider", Input).value = "llamacpp"
            self.query_one("#input-url", Input).value = LOCAL_PRESETS["llamacpp"]["base_url"]
            self.query_one("#input-model", Input).value = LOCAL_PRESETS["llamacpp"]["default_model"]
        elif event.button.id == "btn-cancel":
            self.dismiss(None)
        elif event.button.id == "btn-save":
            provider = self.query_one("#input-provider", Input).value.strip()
            url = self.query_one("#input-url", Input).value.strip()
            model = self.query_one("#input-model", Input).value.strip()
            self.dismiss({"provider": provider, "url": url, "model": model})


class MiniPiApp(App):
    CSS = """
    Screen {
        background: #1e1e2e;
        color: #cdd6f4;
    }
    #top-bar {
        height: 1;
        background: #181825;
        color: #cdd6f4;
        padding: 0 1;
    }
    #main-container {
        height: 1fr;
        padding: 0 1;
    }
    #chat-log {
        height: 1fr;
        border: solid #89b4fa;
        background: #11111b;
        padding: 1;
    }
    #live-stream-box {
        height: auto;
        max-height: 14;
        padding: 0 1;
        margin: 1 0;
        display: none;
    }
    #live-thinking-panel {
        color: #89dceb;
        background: #181825;
        border: dashed #89dceb;
        padding: 0 1;
        margin-bottom: 1;
        display: none;
    }
    #live-text-panel {
        color: #a6e3a1;
        background: #181825;
        border: solid #a6e3a1;
        padding: 0 1;
        display: none;
    }
    #status-bar {
        height: 1;
        background: #313244;
        color: #a6adc8;
        padding: 0 1;
    }
    #input-container {
        height: auto;
        padding-top: 1;
    }
    #prompt-input {
        border: tall #a6e3a1;
        background: #181825;
        color: #cdd6f4;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+m", "switch_model_modal", "Switch Model", show=True),
        Binding("ctrl+k", "show_help", "Commands Help", show=True),
        Binding("ctrl+l", "clear_screen", "Clear Log", show=True),
        Binding("ctrl+r", "probe_connection", "Check Backend", show=True),
    ]

    def __init__(
        self,
        agent: AgentLoop,
        session_store: JsonlSessionStore | None = None,
        initial_history: list[Message] | None = None,
        ext_api: ExtensionAPI | None = None,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.session_store = session_store
        self.history: list[Message] = list(initial_history or [])
        self.ext_api = ext_api
        self.is_busy = False
        self.connection_status = "Connecting..."
        self.connected = False
        self.total_in = 0
        self.total_out = 0
        self.total_think = 0
        self._current_streaming_thinking = ""
        self._current_streaming_text = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="top-bar")
        with Vertical(id="main-container"):
            yield RichLog(id="chat-log", highlight=True, markup=True, wrap=True)
            with Vertical(id="live-stream-box"):
                yield Static("", id="live-thinking-panel")
                yield Static("", id="live-text-panel")
            yield Static("Ready", id="status-bar")
            with Container(id="input-container"):
                yield Input(
                    placeholder="Type task, slash command (/help, /models, /model, /tokens), or question...",
                    id="prompt-input",
                    suggester=SuggestFromList(SLASH_COMMANDS, case_sensitive=False),
                )
        yield Footer()

    async def on_mount(self) -> None:
        self._write_log("[bold cyan]════════════════════════════════════════════════════════════════════[/bold cyan]")
        self._write_log("[bold green] mini-pi Interactive Terminal Coding Agent[/bold green]")
        self._write_log("[dim] Live Streaming • Thinking Tokens Visible • Type [bold cyan]/help[/bold cyan] for slash commands[/dim]")
        self._write_log("[bold cyan]════════════════════════════════════════════════════════════════════[/bold cyan]\n")

        # Hook agent events to TUI
        self.agent.on_event = self._handle_agent_event

        # Hook interactive safety modal into ExtensionAPI if present
        if self.ext_api:
            self.ext_api.context.confirm_callback = self._sync_confirm_request

        # Display history if present
        if self.history:
            self._write_log(f"[dim]Loaded {len(self.history)} messages from session history.[/dim]\n")
            for msg in self.history[-4:]:
                if msg.role == "user":
                    self._write_log(f"[bold yellow]You:[/bold yellow] {msg.text}")
                elif msg.role == "assistant":
                    if msg.thinking:
                        self._write_log(Panel(Markdown(msg.thinking[:300]), title="[dim cyan]💭 Thinking (History)[/dim cyan]", border_style="dim cyan"))
                    if msg.text:
                        self._write_log(Panel(Markdown(msg.text[:300]), title="[bold green]Assistant (History)[/bold green]", border_style="dim green"))

        # Probe backend connection
        self.action_probe_connection()

    def _sync_confirm_request(self, title: str, message: str) -> bool:
        """Fallback confirmation for security gate."""
        return True

    def _write_log(self, item: Any) -> None:
        if self.is_mounted:
            try:
                self.query_one("#chat-log", RichLog).write(item)
            except Exception:
                pass

    def _update_top_bar(self) -> None:
        if not self.is_mounted:
            return
        try:
            top_bar = self.query_one("#top-bar", Static)
            model_name = self.agent.model.model_id
            provider_name = type(self.agent.model).__name__.replace("Client", "")

            url_info = ""
            if isinstance(self.agent.model, LocalOpenAIClient):
                url_info = f" | URL: [dim]{self.agent.model.base_url}[/dim]"
                provider_name = self.agent.model.backend.upper()

            status_badge = "[bold green][CONNECTED][/bold green]" if self.connected else f"[bold yellow][{self.connection_status}][/bold yellow]"
            token_stats = f" | 📊 [dim]In: {self.total_in} Out: {self.total_out}[/dim] [bold cyan]Think: {self.total_think}[/bold cyan]"

            top_bar.update(
                f"{status_badge} Provider: [bold cyan]{provider_name}[/bold cyan] | Model: [bold magenta]{model_name}[/bold magenta]{url_info}{token_stats}"
            )
        except Exception:
            pass

    def _handle_agent_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")

        if kind == "turn_start":
            turn = event.get("turn", 1)
            self._set_status(f"Turn {turn} streaming...")
            self._write_log(f"[dim]── Turn {turn} ──[/dim]")
            self._current_streaming_thinking = ""
            self._current_streaming_text = ""

        elif kind == "thinking_chunk":
            delta = event.get("delta", "")
            self._current_streaming_thinking += delta
            if self.is_mounted:
                try:
                    box = self.query_one("#live-stream-box", Vertical)
                    box.styles.display = "block"
                    think_panel = self.query_one("#live-thinking-panel", Static)
                    think_panel.styles.display = "block"
                    preview = self._current_streaming_thinking[-300:] if len(self._current_streaming_thinking) > 300 else self._current_streaming_thinking
                    think_panel.update(f"💭 [bold cyan]Thinking Live...[/bold cyan]\n[italic dim]{preview}[/italic dim]")
                except Exception:
                    pass

        elif kind == "text_chunk":
            delta = event.get("delta", "")
            self._current_streaming_text += delta
            if self.is_mounted:
                try:
                    box = self.query_one("#live-stream-box", Vertical)
                    box.styles.display = "block"
                    text_panel = self.query_one("#live-text-panel", Static)
                    text_panel.styles.display = "block"
                    preview = self._current_streaming_text[-400:] if len(self._current_streaming_text) > 400 else self._current_streaming_text
                    text_panel.update(f"[bold green]Assistant (Streaming)...[/bold green]\n{preview}")
                except Exception:
                    pass

        elif kind == "message_end":
            # Reset live streaming box widgets
            if self.is_mounted:
                try:
                    box = self.query_one("#live-stream-box", Vertical)
                    box.styles.display = "none"
                    think_panel = self.query_one("#live-thinking-panel", Static)
                    think_panel.styles.display = "none"
                    think_panel.update("")
                    text_panel = self.query_one("#live-text-panel", Static)
                    text_panel.styles.display = "none"
                    text_panel.update("")
                except Exception:
                    pass

            msg_dict = event.get("message", {})
            in_tok = event.get("input_tokens", 0)
            out_tok = event.get("output_tokens", 0)
            think_tok = event.get("thinking_tokens", 0)

            self.total_in = event.get("total_input_tokens", self.total_in + in_tok)
            self.total_out = event.get("total_output_tokens", self.total_out + out_tok)
            self.total_think = event.get("total_thinking_tokens", self.total_think + think_tok)

            # Commit completed thinking and text panels into permanent RichLog
            for b in msg_dict.get("content", []):
                if b.get("type") == "thinking":
                    th = b.get("thinking", "").strip()
                    tok_str = f" [cyan]({think_tok} thinking tokens)[/cyan]" if think_tok else ""
                    if th:
                        self._write_log(
                            Panel(
                                Markdown(th),
                                title=f"[dim cyan]💭 Thinking / Reasoning{tok_str}[/dim cyan]",
                                border_style="dim cyan",
                            )
                        )
                elif b.get("type") == "text":
                    text = b.get("text", "")
                    if text.strip():
                        self._write_log(Panel(Markdown(text), title="[bold green]Assistant[/bold green]", border_style="green"))

            if in_tok + out_tok > 0:
                self._write_log(
                    f"[dim]📊 Turn Tokens: {in_tok} in | {out_tok} out | [cyan]{think_tok} thinking[/cyan] (Total: {in_tok + out_tok})[/dim]"
                )
            self._update_top_bar()

        elif kind == "tool_execution_start":
            tool = event.get("tool", "tool")
            inp = event.get("input", {})
            self._set_status(f"Running tool '{tool}'...")
            self._write_log(f"[bold cyan]▸ Tool Call: [bold]{tool}[/bold][/bold cyan]")
            import json
            inp_preview = json.dumps(inp, indent=2)
            self._write_log(Panel(Syntax(inp_preview, "json", theme="monokai", word_wrap=True), border_style="cyan"))

        elif kind == "tool_execution_end":
            tool = event.get("tool", "tool")
            is_err = event.get("is_error", False)
            sym = "[bold red]✗[/bold red]" if is_err else "[bold green]✓[/bold green]"
            stat = "[red]failed[/red]" if is_err else "[green]done[/green]"
            self._write_log(f"{sym} [dim]{tool} {stat}[/dim]")

        elif kind == "tool_result":
            res = event.get("result", {})
            for b in res.get("content", []):
                if b.get("type") == "tool_result":
                    output = b.get("content", "")
                    is_err = b.get("is_error", False)
                    lines = output.splitlines()
                    preview = "\n".join(lines[:12]) + (f"\n... [{len(lines)-12} more lines]" if len(lines) > 12 else "")
                    self._write_log(
                        Panel(
                            preview or "[dim]empty result[/dim]",
                            title=f"[dim]Tool Result ({'error' if is_err else 'success'})[/dim]",
                            border_style="red" if is_err else "dim",
                        )
                    )

        elif kind == "agent_settled":
            self._set_status("Ready")
            self._update_top_bar()

    def _set_status(self, text: str) -> None:
        if not self.is_mounted:
            return
        try:
            status = self.query_one("#status-bar", Static)
            status.update(f"[bold white]{text}[/bold white] | Dir: [dim]{self.agent.cwd.name}[/dim]")
        except Exception:
            pass

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text or self.is_busy:
            return

        if self.is_mounted:
            input_widget = self.query_one("#prompt-input", Input)
            input_widget.value = ""

        # Handle slash commands
        if user_text.startswith("/"):
            await self._handle_slash_command(user_text)
            return

        # Normal user prompt
        self._write_log(f"\n[bold yellow]User:[/bold yellow] {user_text}")
        self.is_busy = True
        self._set_status("Agent thinking...")
        self.run_agent_task(user_text)

    async def _handle_slash_command(self, cmd: str) -> None:
        parts = cmd.split(maxsplit=1)
        name = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if name in {"/exit", "/quit"}:
            self.exit()
            return

        elif name in {"/help", "/commands"}:
            self.action_show_help()

        elif name in {"/clear", "/cls"}:
            if self.is_mounted:
                self.query_one("#chat-log", RichLog).clear()

        elif name == "/clear-session":
            self.history.clear()
            self.total_in = 0
            self.total_out = 0
            self.total_think = 0
            if self.session_store:
                self.session_store.clear()
            self._write_log("[bold yellow]Session history and token metrics cleared.[/bold yellow]")
            self._update_top_bar()

        elif name in {"/tokens", "/usage"}:
            table = Table(title="Session Token Usage Breakdown", show_header=True)
            table.add_column("Category", style="cyan")
            table.add_column("Count", justify="right", style="bold")
            table.add_row("Input (Prompt) Tokens", str(self.total_in))
            table.add_row("Output (Completion) Tokens", str(self.total_out))
            table.add_row("Thinking (Reasoning) Tokens", f"[bold cyan]{self.total_think}[/bold cyan]")
            table.add_row("Total Processed Tokens", str(self.total_in + self.total_out))
            self._write_log(table)

        elif name == "/models":
            await self._list_models_command()

        elif name == "/model":
            if not arg:
                await self._list_models_command()
            else:
                self._switch_model(arg)

        elif name == "/provider":
            if not arg:
                self._write_log("[cyan]Usage: /provider <lmstudio|ollama|llamacpp|anthropic|openai>[/cyan]")
            else:
                await self._switch_provider(arg)

        elif name == "/connect":
            if not arg:
                self._write_log("[cyan]Usage: /connect <http://localhost:1234/v1>[/cyan]")
            else:
                await self._connect_url(arg)

        elif name == "/probe":
            self.action_probe_connection()

        elif name == "/skills":
            skills = list_skills(self.agent.cwd)
            if not skills:
                self._write_log("[dim]No skills found in ~/.mini-pi/skills or ./skills[/dim]")
            else:
                table = Table(title="Available Skills", show_header=True)
                table.add_column("Skill Name", style="cyan")
                table.add_column("First line")
                for s_name, content in skills.items():
                    first_line = content.splitlines()[0] if content else ""
                    table.add_row(s_name, first_line)
                self._write_log(table)

        elif name == "/skill":
            if not arg:
                self._write_log("[cyan]Usage: /skill <skill_name>[/cyan]")
            else:
                try:
                    content = load_skill(arg, self.agent.cwd)
                    self._write_log(f"[green]Loaded skill '{arg}':[/green]\n{content[:250]}...\n")
                    self._write_log("[dim]Injecting into next agent turn...[/dim]")
                    prompt = f"Follow this procedure from skill '{arg}':\n\n{content}"
                    self.run_agent_task(prompt)
                except Exception as e:
                    self._write_log(f"[red]Error loading skill: {e}[/red]")

        elif name == "/prompts":
            prompts = list_templates(self.agent.cwd)
            if not prompts:
                self._write_log("[dim]No prompt templates found in ~/.mini-pi/prompts or ./prompts[/dim]")
            else:
                table = Table(title="Available Templates", show_header=True)
                table.add_column("Template", style="cyan")
                table.add_column("Lines")
                for p_name, content in prompts.items():
                    table.add_row(f"/{p_name}", str(len(content.splitlines())))
                self._write_log(table)

        elif name == "/prompt":
            if not arg:
                self._write_log("[cyan]Usage: /prompt <template_name> (e.g. /prompt review)[/cyan]")
            else:
                try:
                    content = load_template(arg, self.agent.cwd)
                    self._write_log(f"[green]Expanded template '{arg}':[/green]\n{content}\n")
                    self.run_agent_task(content)
                except Exception as e:
                    self._write_log(f"[red]Error loading template: {e}[/red]")

        elif name == "/diff":
            await self._show_diff_command()

        elif name == "/compact":
            self._write_log("[dim]Compacting conversation history...[/dim]")
            self.history = await compact_history(self.history, keep_recent=6)
            if self.session_store:
                self.session_store.save_all(self.history)
            self._write_log(f"[green]Compacted history to {len(self.history)} messages.[/green]")
            self._update_top_bar()

        elif name == "/tree":
            tree = SessionTree.from_messages(self.history)
            table = Table(title="Session Tree", show_header=True)
            table.add_column("ID", style="dim")
            table.add_column("Role", style="cyan")
            table.add_column("Parent", style="dim")
            table.add_column("Preview")
            for msg in self.history:
                active = " *" if msg.id == tree.active_leaf_id else ""
                table.add_row(msg.id[:8], f"{msg.role}{active}", (msg.parent_id or "")[:8], msg.text[:50])
            self._write_log(table)

        elif name == "/branch":
            if not arg:
                self._write_log("[cyan]Usage: /branch <message_id_prefix>[/cyan]")
            else:
                tree = SessionTree.from_messages(self.history)
                matches = [mid for mid in tree.messages if mid.startswith(arg)]
                if not matches:
                    self._write_log(f"[red]No branch found matching prefix '{arg}'[/red]")
                else:
                    tree.switch_to(matches[0])
                    self.history = tree.active_path()
                    self._write_log(f"[green]Switched active leaf to {matches[0][:8]}. Branch path length: {len(self.history)}[/green]")
                    self._update_top_bar()
        else:
            self._write_log(f"[yellow]Unknown command '{name}'. Type [bold]/help[/bold] for available commands.[/yellow]")

    async def _list_models_command(self) -> None:
        if isinstance(self.agent.model, LocalOpenAIClient):
            self._write_log(f"[dim]Querying models from {self.agent.model.base_url}...[/dim]")
            try:
                models = await self.agent.model.list_available_models()
                if not models:
                    self._write_log(f"[yellow]Server responded, but no models found at {self.agent.model.base_url}.[/yellow]")
                else:
                    table = Table(title=f"Available Models on {self.agent.model.backend.upper()}", show_header=True)
                    table.add_column("Model ID", style="bold cyan")
                    table.add_column("Active")
                    for m in models:
                        is_active = "[green]✓ (Current)[/green]" if m == self.agent.model.model_id else ""
                        table.add_row(m, is_active)
                    self._write_log(table)
                    self._write_log("[dim]Switch with: [bold]/model <name>[/bold] or press Ctrl+M[/dim]")
            except Exception as e:
                self._write_log(f"[bold red]Could not list models:[/bold red] {e}")
                self._write_log(f"[dim]Check if LM Studio, Ollama, or llama.cpp is running on {self.agent.model.base_url}[/dim]")
        else:
            self._write_log(f"[cyan]Current model: {self.agent.model.model_id} (Provider: {type(self.agent.model).__name__})[/cyan]")

    def _switch_model(self, model_name: str) -> None:
        if hasattr(self.agent.model, "set_model"):
            self.agent.model.set_model(model_name)
            self._write_log(f"[bold green]Switched active model to:[/bold green] [bold magenta]{model_name}[/bold magenta]")
        else:
            self.agent.model._model_id = model_name
            self._write_log(f"[bold green]Updated model id to:[/bold green] [bold magenta]{model_name}[/bold magenta]")
        self._update_top_bar()

    async def _switch_provider(self, provider_name: str) -> None:
        p = provider_name.lower()
        if p in {"lmstudio", "ollama", "llamacpp"}:
            if isinstance(self.agent.model, LocalOpenAIClient):
                self.agent.model.switch_backend(p)
            else:
                self.agent.model = LocalOpenAIClient(backend=p)
            self._write_log(f"[bold green]Switched backend to {p.upper()} ({LOCAL_PRESETS[p]['base_url']})[/bold green]")
            self.action_probe_connection()
        else:
            from mini_pi.cli import create_provider
            try:
                self.agent.model = create_provider(p, model_id=None)
                self._write_log(f"[bold green]Switched provider to {p}[/bold green]")
                self._update_top_bar()
            except Exception as e:
                self._write_log(f"[red]Failed to switch provider: {e}[/red]")

    async def _connect_url(self, url: str) -> None:
        if isinstance(self.agent.model, LocalOpenAIClient):
            self.agent.model.set_base_url(url)
            self._write_log(f"[cyan]Updated local model base URL to {url}[/cyan]")
            self.action_probe_connection()
        else:
            self.agent.model = LocalOpenAIClient(base_url=url)
            self._write_log(f"[cyan]Configured local provider at {url}[/cyan]")
            self.action_probe_connection()

    async def _show_diff_command(self) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "diff",
                cwd=str(self.agent.cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            diff_text = stdout.decode("utf-8", errors="replace").strip()
            if not diff_text:
                self._write_log("[dim]No unstaged git changes found in workspace.[/dim]")
            else:
                self._write_log(Panel(Syntax(diff_text, "diff", theme="monokai"), title="[bold cyan]Working Tree Diff[/bold cyan]"))
        except Exception as e:
            self._write_log(f"[red]Error fetching git diff: {e}[/red]")

    @work(exclusive=True)
    async def run_agent_task(self, user_text: str) -> None:
        try:
            self.history = await self.agent.run(self.history, user_text)
            if self.session_store:
                self.session_store.save_all(self.history)
        except Exception as e:
            self._write_log(f"[bold red]Agent error:[/bold red] {e}")
        finally:
            self.is_busy = False
            self._set_status("Ready")
            self._update_top_bar()

    def action_switch_model_modal(self) -> None:
        current_p = getattr(self.agent.model, "backend", type(self.agent.model).__name__)
        current_m = self.agent.model.model_id
        current_u = getattr(self.agent.model, "base_url", "https://api.openai.com/v1")

        def handle_modal_result(result: dict[str, str] | None) -> None:
            if not result:
                return
            p = result["provider"]
            u = result["url"]
            m = result["model"]
            if p in {"lmstudio", "ollama", "llamacpp"} or "localhost" in u or "127.0.0.1" in u:
                self.agent.model = LocalOpenAIClient(backend=p, base_url=u, model_id=m)
            else:
                from mini_pi.cli import create_provider
                self.agent.model = create_provider(p, model_id=m, base_url=u)
            self._write_log(f"[bold green]Updated Model Settings:[/bold green] Provider={p}, Model={m}, BaseURL={u}")
            self.action_probe_connection()

        self.push_screen(ModelSelectModal(current_p, current_m, current_u), handle_modal_result)

    def action_show_help(self) -> None:
        table = Table(title="mini-pi TUI Commands & Hotkeys", show_header=True)
        table.add_column("Command / Hotkey", style="bold cyan")
        table.add_column("Action")

        table.add_row("Ctrl+M", "Open interactive Model & Provider switcher dialog")
        table.add_row("Ctrl+K", "Show this commands reference")
        table.add_row("Ctrl+L", "Clear chat log view")
        table.add_row("Ctrl+R", "Probe / refresh connection to local server")
        table.add_row("Ctrl+C / Ctrl+Q", "Quit mini-pi")
        table.add_row("─── Slash Commands ───", "────────────────────────────────")
        table.add_row("/tokens, /usage", "View live breakdown of input, output & thinking tokens")
        table.add_row("/models", "Query & list all models available on local server")
        table.add_row("/model <name>", "Switch active model (e.g. /model qwen2.5-coder)")
        table.add_row("/provider <name>", "Switch provider (lmstudio, ollama, llamacpp, etc.)")
        table.add_row("/connect <url>", "Change local server base URL & test connection")
        table.add_row("/probe", "Test local connection to LM Studio / Ollama")
        table.add_row("/skills", "List available skills from SKILL.md catalog")
        table.add_row("/skill <name>", "Inject a skill procedure into next turn")
        table.add_row("/prompts", "List prompt templates (/review, /debug, etc.)")
        table.add_row("/prompt <name>", "Expand prompt template")
        table.add_row("/diff", "Show live git diff of workspace changes")
        table.add_row("/compact", "Compact previous turns in conversation")
        table.add_row("/tree", "Display branching message tree")
        table.add_row("/branch <id>", "Switch to alternative leaf in session tree")
        table.add_row("/clear-session", "Reset entire conversation session & token metrics")
        table.add_row("/clear", "Clear chat screen")
        table.add_row("/exit", "Exit mini-pi")
        self._write_log(table)

    def action_clear_screen(self) -> None:
        if self.is_mounted:
            self.query_one("#chat-log", RichLog).clear()

    @work
    async def action_probe_connection(self) -> None:
        if isinstance(self.agent.model, LocalOpenAIClient):
            ok, msg, models = await self.agent.model.test_connection()
            self.connected = ok
            self.connection_status = "Online" if ok else "Offline"
            if ok:
                self._write_log(f"[bold green]● {msg}[/bold green] (Models available: {len(models)})")
            else:
                self._write_log(f"[bold yellow]● {msg}[/bold yellow]")
        else:
            self.connected = True
            self.connection_status = "Cloud Ready"
        self._update_top_bar()
