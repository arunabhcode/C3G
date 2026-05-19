"""Helpers for Modal local entrypoints (blocking .remote vs detached .spawn)."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


def dispatch_remote(
    fn: Callable[..., T],
    /,
    *args: Any,
    detach: bool = False,
    job_name: str = "job",
    app_name: str | None = None,
    **kwargs: Any,
) -> T | Any:
    """
    Run a Modal function or cls method, blocking unless ``detach`` is True.

    When detached, returns a :class:`modal.FunctionCall` handle immediately.
    """
    if detach:
        handle = fn.spawn(*args, **kwargs)
        print(f"Detached {job_name}.")
        print(f"  call_id: {handle.object_id}")
        if app_name:
            print(f"  logs:    modal app logs {app_name}")
        print("  (Or use: modal run --detach …  from the CLI.)")
        return handle
    return fn.remote(*args, **kwargs)
