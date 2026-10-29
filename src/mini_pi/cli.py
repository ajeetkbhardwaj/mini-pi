from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from mini_pi.agent.context import build_system_prompt
from mini_pi.agent.loop import AgentConfig, AgentLoop
from mini_pi.agent.types import Message
from mini_pi.extensions.api import ExtensionAPI
from mini_pi.extensions.runtime import ExtensionRuntime
from mini_pi.providers.anthropic_provider import AnthropicClient
from mini_pi.providers.base import ModelClient
from mini_pi.providers.local_provider import LOCAL_PRESETS, LocalOpenAIClient
from mini_pi.providers.mock_provider import MockModelClient
from mini_pi.providers.openai_provider import OpenAIClient
from mini_pi.session.compact import compact_history
from mini_pi.session.store import JsonlSessionStore
from mini_pi.session.tree import SessionTree
from mini_pi.skills.loader import get_skill_summary, list_skills, load_skill
from mini_pi.skills.templates import list_templates, load_template
from mini_pi.tools.bash import BashTool
from mini_pi.tools.edit import EditTool
from mini_pi.tools.read import ReadTool
from mini_pi.tools.registry import ToolRegistry
from mini_pi.tools.write import WriteTool
from mini_pi.tui.app import MiniPiApp
from mini_pi.tui.console import console, render_event


def create_provider(
    provider_name: str | None,
    model_id: str | None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> ModelClient:
    name = (provider_name or "").lower()

    if not name:
        if os.environ.get("LOCAL_OPENAI_BASE_URL"):
            name = "local"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            name = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            name = "openai"
        else:
            name = "local"  # Default to local model (LM Studio / Ollama / llama.cpp)

    if name in {"local", "lmstudio", "ollama", "llamacpp"}:
        backend = "lmstudio" if name == "local" else name
        return LocalOpenAIClient(
            backend=backend,
            base_url=base_url,
            model_id=model_id,
            api_key=api_key or "local-key",
        )
    elif name == "anthropic":
        return AnthropicClient(
            model_id=model_id or "claude-3-7-sonnet-20250219",
            api_key=api_key,
        )
    elif name == "openai":
        return OpenAIClient(
            model_id=model_id or "gpt-4o",
            api_key=api_key,
            base_url=base_url,
        )
    elif name == "mock":
        return MockModelClient(model_id=model_id or "mock-model")
    else:
        raise ValueError(
            f"Unsupported provider: '{name}'. Choose 'local', 'lmstudio', 'ollama', 'llamacpp', 'anthropic', or 'openai'."
        )


def build_tool_registry(ext_api: ExtensionAPI | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ReadTool())
    registry.register(WriteTool())
    registry.register(EditTool())
    registry.register(BashTool())

    if ext_api:
        for tool in ext_api.tools:
            if not registry.has(tool.name):
                registry.register(tool)

    return registry


async def run_repl(
    agent: AgentLoop,
    store: JsonlSessionStore,
    history: list[Message],
    cwd: Path,
) -> None:
    console.print(f"[bold cyan]mini-pi Interactive REPL[/bold cyan] [dim](Model: {agent.model.model_id})[/dim]")
    console.print("[dim]Type your message or a slash command (/help, /skills, /prompts, /compact, /tree, /exit)[/dim]\n")

    tree = SessionTree.from_messages(history)

    while True:
        try:
            user_input = input("mini-pi > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if not user_input:
            continue

        if user_input in {"/exit", "/quit"}:
            console.print("[yellow]Exiting mini-pi.[/yellow]")
            break

        if user_input == "/help":
            table = Table(title="Slash Commands", show_header=True)
            table.add_column("Command", style="cyan")
            table.add_column("Description")
            table.add_row("/help", "Show this help table")
            table.add_row("/tokens", "View input, output & thinking tokens")
            table.add_row("/skills", "List available skills")
            table.add_row("/skill <name>", "Inject a skill workflow into next prompt")
            table.add_row("/prompts", "List prompt templates")
            table.add_row("/prompt <name>", "Expand a prompt template")
            table.add_row("/compact", "Compact previous turns in session")
            table.add_row("/tree", "Display branching session tree")
            table.add_row("/branch <id>", "Switch active branch leaf")
            table.add_row("/clear", "Clear current session")
            table.add_row("/exit, /quit", "Quit session")
            console.print(table)
            continue

        if user_input in {"/tokens", "/usage"}:
            console.print(
                f"[cyan]📊 Session Tokens:[/cyan] Prompt={agent.total_input_tokens} | Completion={agent.total_output_tokens} | [bold cyan]Thinking={agent.total_thinking_tokens}[/bold cyan] (Total={agent.total_input_tokens + agent.total_output_tokens})"
            )
            continue

        if user_input == "/skills":
            skills = list_skills(cwd)
            if not skills:
                console.print("[dim]No skills found in ~/.mini-pi/skills or ./skills[/dim]")
            else:
                table = Table(title="Available Skills", show_header=True)
                table.add_column("Skill Name", style="cyan")
                table.add_column("Preview")
                for s_name, content in skills.items():
                    first_line = content.splitlines()[0] if content else ""
                    table.add_row(s_name, first_line)
                console.print(table)
            continue

        if user_input.startswith("/skill "):
            s_name = user_input.split(maxsplit=1)[1].strip()
            try:
                skill_content = load_skill(s_name, cwd)
                user_input = f"Follow this procedure from skill '{s_name}':\n\n{skill_content}"
                console.print(f"[green]Loaded skill '{s_name}' into context.[/green]")
            except Exception as e:
                console.print(f"[red]Error loading skill: {e}[/red]")
                continue

        if user_input == "/prompts":
            prompts = list_templates(cwd)
            if not prompts:
                console.print("[dim]No templates found in ~/.mini-pi/prompts or ./prompts[/dim]")
            else:
                table = Table(title="Prompt Templates", show_header=True)
                table.add_column("Template", style="cyan")
                table.add_column("Lines")
                for p_name, content in prompts.items():
                    table.add_row(f"/{p_name}", str(len(content.splitlines())))
                console.print(table)
            continue

        if user_input.startswith("/prompt "):
            p_name = user_input.split(maxsplit=1)[1].strip()
            try:
                user_input = load_template(p_name, cwd)
                console.print(f"[green]Loaded template '{p_name}':[/green]\n{user_input}\n")
            except Exception as e:
                console.print(f"[red]Error loading template: {e}[/red]")
                continue

        if user_input == "/compact":
            history = await compact_history(history, keep_recent=6)
            store.save_all(history)
            console.print(f"[green]Session compacted to {len(history)} messages.[/green]")
            continue

        if user_input == "/tree":
            table = Table(title="Session Message Tree", show_header=True)
            table.add_column("Message ID", style="dim")
            table.add_column("Role", style="cyan")
            table.add_column("Parent ID", style="dim")
            table.add_column("Preview")
            for msg in history:
                active_marker = " *active*" if msg.id == tree.active_leaf_id else ""
                table.add_row(msg.id[:8], f"{msg.role}{active_marker}", (msg.parent_id or "")[:8], msg.text[:60])
            console.print(table)
            continue

        if user_input.startswith("/branch "):
            target_id = user_input.split(maxsplit=1)[1].strip()
            matches = [m_id for m_id in tree.messages if m_id.startswith(target_id)]
            if not matches:
                console.print(f"[red]No message matching id prefix '{target_id}'[/red]")
                continue
            tree.switch_to(matches[0])
            history = tree.active_path()
            console.print(f"[green]Switched active branch to leaf {matches[0][:8]}. Path length: {len(history)}[/green]")
            continue

        if user_input == "/clear":
            history.clear()
            store.clear()
            tree = SessionTree()
            console.print("[yellow]Session cleared.[/yellow]")
            continue

        # Execute normal agent turn
        try:
            history = await agent.run(history, user_input)
            store.save_all(history)
            tree = SessionTree.from_messages(history)
        except Exception as e:
            console.print(f"[bold red]Turn failed:[/bold red] {e}")


async def async_main() -> int:
    parser = argparse.ArgumentParser(description="mini-pi: Modular terminal coding agent in Python")
    parser.add_argument("prompt", nargs="*", help="Optional initial user prompt to execute")
    parser.add_argument(
        "--provider",
        choices=["local", "lmstudio", "ollama", "llamacpp", "anthropic", "openai", "mock"],
        help="LLM provider (defaults to local LMStudio/Ollama/llama.cpp)",
    )
    parser.add_argument("--model", help="Model identifier (e.g. qwen2.5-coder, claude-3-7-sonnet, gpt-4o)")
    parser.add_argument("--api-key", help="API key for selected provider")
    parser.add_argument("--base-url", help="Base URL for OpenAI-compatible endpoint (e.g. http://localhost:1234/v1)")
    parser.add_argument("--cwd", default=".", help="Project workspace directory")
    parser.add_argument("--tui", action="store_true", help="Launch interactive Textual TUI interface")
    parser.add_argument("--compact", action="store_true", help="Compact prior history before running")
    parser.add_argument("--clear-session", action="store_true", help="Clear previous session history")
    parser.add_argument("--max-turns", type=int, default=40, help="Maximum agent turns")

    args = parser.parse_args()

    cwd = Path(args.cwd).resolve()
    session_file = cwd / ".mini-pi" / "session.jsonl"
    store = JsonlSessionStore(session_file)

    if args.clear_session:
        store.clear()

    history = store.load()
    if args.compact and history:
        history = await compact_history(history)
        store.save_all(history)

    # Initialize extensions and safety runtime
    ext_api = ExtensionAPI(cwd=cwd)
    ext_runtime = ExtensionRuntime(ext_api)
    await ext_runtime.initialize(enable_safety=True)

    tools = build_tool_registry(ext_api)

    async def safety_gate(tool_name: str, tool_args: dict[str, Any]) -> tuple[bool, str]:
        res = await ext_api.emit("tool_call", {"tool": tool_name, "input": tool_args})
        if isinstance(res, dict) and res.get("block"):
            return False, res.get("reason", "Blocked by safety extension")
        return True, ""

    try:
        model = create_provider(
            provider_name=args.provider,
            model_id=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
        )
    except Exception as e:
        console.print(f"[bold red]Provider setup error:[/bold red] {e}")
        return 1

    system_prompt = build_system_prompt(cwd)
    config = AgentConfig(system_prompt=system_prompt, max_turns=args.max_turns)

    agent = AgentLoop(
        model=model,
        tools=tools,
        config=config,
        cwd=cwd,
        on_event=render_event,
        tool_interceptor=safety_gate,
    )

    prompt_text = " ".join(args.prompt).strip()

    # Launch Textual TUI if requested
    if args.tui:
        app = MiniPiApp(
            agent=agent,
            session_store=store,
            initial_history=history,
            ext_api=ext_api,
        )
        await app.run_async()
        return 0

    # If prompt passed, run headless/single command mode
    if prompt_text:
        try:
            history = await agent.run(history, prompt_text)
            store.save_all(history)
            final_msg = history[-1]
            if final_msg.thinking:
                console.print(Panel(final_msg.thinking, title="[dim cyan]💭 Thinking / Reasoning[/dim cyan]", border_style="dim cyan"))
            if final_msg.text:
                console.print(f"\n[bold green]Final Answer:[/bold green]\n{final_msg.text}")
            if agent.total_input_tokens + agent.total_output_tokens > 0:
                console.print(
                    f"\n[dim]📊 Tokens: {agent.total_input_tokens} prompt | {agent.total_output_tokens} completion | [bold cyan]{agent.total_thinking_tokens} thinking[/bold cyan] (Total: {agent.total_input_tokens + agent.total_output_tokens})[/dim]"
                )
            return 0
        except Exception as e:
            console.print(f"[bold red]Execution error:[/bold red] {e}")
            return 1

    # Otherwise enter interactive REPL
    await run_repl(agent, store, history, cwd)
    return 0


def cli_entry() -> None:
    code = asyncio.run(async_main())
    sys.exit(code)


if __name__ == "__main__":
    cli_entry()
