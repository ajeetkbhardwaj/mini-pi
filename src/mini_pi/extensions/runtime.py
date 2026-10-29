from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

from mini_pi.extensions.api import ExtensionAPI
from mini_pi.extensions.safety import register as register_safety


class ExtensionRuntime:
    def __init__(self, api: ExtensionAPI) -> None:
        self.api = api

    async def initialize(self, enable_safety: bool = True) -> None:
        if enable_safety:
            await register_safety(self.api)

        # Look for local and global extensions
        search_dirs = [
            Path.home() / ".mini-pi" / "extensions",
            self.api.cwd / ".mini-pi" / "extensions",
        ]

        for directory in search_dirs:
            if directory.is_dir():
                for file_path in directory.glob("*.py"):
                    if not file_path.name.startswith("_"):
                        await self.load_extension_from_file(file_path)

    async def load_extension_from_file(self, file_path: Path) -> None:
        try:
            module_name = f"mini_pi_ext_{file_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, "register"):
                    register_fn = getattr(module, "register")
                    if inspect.iscoroutinefunction(register_fn):
                        await register_fn(self.api)
                    else:
                        register_fn(self.api)
        except Exception:
            pass

