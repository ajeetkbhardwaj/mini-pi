from __future__ import annotations

import json
import sys
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

console = Console()

_streaming_state = {
    "is_thinking": False,
    "is_text": False,
}


def render_event(event: dict[str, Any]) -> None:
    kind = event.get("type")

    if kind == "agent_start":
        prompt = event.get("prompt", "")
        if prompt:
            console.print(f"\n[bold yellow]User:[/bold yellow] {prompt}")

    elif kind == "turn_start":
        _streaming_state["is_thinking"] = False
        _streaming_state["is_text"] = False
        turn = event.get("turn", 1)
        console.rule(f"[dim]Turn {turn}[/dim]", style="dim cyan")

    elif kind == "thinking_chunk":
        delta = event.get("delta", "")
        if not _streaming_state["is_thinking"]:
            _streaming_state["is_thinking"] = True
            console.print("[dim cyan]💭 Thinking:[/dim cyan] [italic dim]", end="")
        console.print(delta, end="")
        sys.stdout.flush()

    elif kind == "text_chunk":
        delta = event.get("delta", "")
        if _streaming_state["is_thinking"]:
            _streaming_state["is_thinking"] = False
            console.print("[/italic dim]\n")
        if not _streaming_state["is_text"]:
            _streaming_state["is_text"] = True
            console.print("[bold green]Assistant:[/bold green] ", end="")
        console.print(delta, end="")
        sys.stdout.flush()

    elif kind == "message_end":
        if _streaming_state["is_thinking"]:
            _streaming_state["is_thinking"] = False
            console.print("[/italic dim]\n")
        if _streaming_state["is_text"]:
            _streaming_state["is_text"] = False
            console.print("\n")

        in_tok = event.get("input_tokens", 0)
        out_tok = event.get("output_tokens", 0)
        think_tok = event.get("thinking_tokens", 0)
        tot_tok = in_tok + out_tok
        if tot_tok > 0:
            console.print(
                f"[dim]📊 Tokens: {in_tok} in | {out_tok} out | {think_tok} thinking (Turn Total: {tot_tok})[/dim]"
            )

    elif kind == "tool_execution_start":
        if _streaming_state["is_text"]:
            _streaming_state["is_text"] = False
            console.print("\n")
        tool = event.get("tool", "unknown")
        inp = event.get("input", {})
        inp_str = json.dumps(inp, indent=2)
        console.print(
            Panel(
                Syntax(inp_str, "json", theme="monokai", word_wrap=True),
                title=f"[bold cyan]▸ Tool Call: {tool}[/bold cyan]",
                border_style="cyan",
            )
        )

    elif kind == "tool_execution_end":
        tool = event.get("tool", "unknown")
        is_error = event.get("is_error", False)
        style = "bold red" if is_error else "bold green"
        symbol = "✗" if is_error else "✓"
        status_text = "failed" if is_error else "succeeded"
        console.print(f"[{style}]{symbol} {tool} {status_text}[/{style}]")

    elif kind == "tool_result":
        res = event.get("result", {})
        content_list = res.get("content", [])
        for item in content_list:
            if item.get("type") == "tool_result":
                output = item.get("content", "")
                is_err = item.get("is_error", False)
                border = "red" if is_err else "dim"
                lines = output.splitlines()
                if len(lines) > 20:
                    preview = "\n".join(lines[:15]) + f"\n... [{len(lines)-15} more lines]"
                else:
                    preview = output
                console.print(
                    Panel(
                        preview or "[dim]empty output[/dim]",
                        title=f"[dim]Tool Output ({'error' if is_err else 'ok'})[/dim]",
                        border_style=border,
                    )
                )

    elif kind == "agent_settled":
        console.print("[bold green]● Ready[/bold green]\n")
