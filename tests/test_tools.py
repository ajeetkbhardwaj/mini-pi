from __future__ import annotations

from pathlib import Path
import pytest

from mini_pi.tools.bash import BashTool
from mini_pi.tools.edit import EditTool
from mini_pi.tools.path_guard import validate_workspace_path
from mini_pi.tools.read import ReadTool
from mini_pi.tools.registry import ToolContext, ToolRegistry
from mini_pi.tools.write import WriteTool


@pytest.fixture
def temp_workspace(tmp_path: Path) -> ToolContext:
    (tmp_path / "sample.txt").write_text("line one\nline two\nline three\n", encoding="utf-8")
    return ToolContext(cwd=tmp_path)


@pytest.mark.asyncio
async def test_read_tool(temp_workspace: ToolContext):
    tool = ReadTool()
    output = await tool.run({"path": "sample.txt"}, temp_workspace)
    assert "1 | line one" in output
    assert "2 | line two" in output
    assert "3 | line three" in output


@pytest.mark.asyncio
async def test_read_tool_with_offset_and_limit(temp_workspace: ToolContext):
    tool = ReadTool()
    output = await tool.run({"path": "sample.txt", "offset": 2, "limit": 1}, temp_workspace)
    assert "2 | line two" in output
    assert "1 | line one" not in output
    assert "3 | line three" not in output


@pytest.mark.asyncio
async def test_write_and_edit_tools(temp_workspace: ToolContext):
    write_tool = WriteTool()
    edit_tool = EditTool()

    # Write a new file
    write_res = await write_tool.run(
        {"path": "subdir/test.py", "content": "def foo():\n    return 41\n"},
        temp_workspace,
    )
    assert "Wrote subdir/test.py" in write_res
    created_file = temp_workspace.cwd / "subdir" / "test.py"
    assert created_file.exists()
    assert "41" in created_file.read_text(encoding="utf-8")

    # Edit file
    edit_res = await edit_tool.run(
        {"path": "subdir/test.py", "old_text": "return 41", "new_text": "return 42"},
        temp_workspace,
    )
    assert "Edited subdir/test.py" in edit_res
    assert "return 42" in created_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_edit_tool_validation(temp_workspace: ToolContext):
    edit_tool = EditTool()

    # Old text not found
    with pytest.raises(ValueError, match="old_text not found"):
        await edit_tool.run(
            {"path": "sample.txt", "old_text": "nonexistent", "new_text": "foo"},
            temp_workspace,
        )

    # Ambiguous match
    (temp_workspace.cwd / "dups.txt").write_text("hello\nhello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="matched 2 times"):
        await edit_tool.run(
            {"path": "dups.txt", "old_text": "hello", "new_text": "world"},
            temp_workspace,
        )


@pytest.mark.asyncio
async def test_bash_tool(temp_workspace: ToolContext):
    tool = BashTool()
    res = await tool.run({"command": "echo 'mini-pi rocks'"}, temp_workspace)
    assert "Exit code: 0" in res
    assert "mini-pi rocks" in res


def test_path_guard(tmp_path: Path):
    # Escape test
    with pytest.raises(PermissionError, match="escapes workspace"):
        validate_workspace_path(tmp_path, "../../etc/passwd")

    # Protected file test
    with pytest.raises(PermissionError, match="Access to protected path"):
        validate_workspace_path(tmp_path, ".env")

    with pytest.raises(PermissionError, match="Access to protected path"):
        validate_workspace_path(tmp_path, ".git/config")

