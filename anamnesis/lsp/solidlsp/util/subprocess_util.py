import platform
import shlex
import subprocess


def subprocess_kwargs() -> dict:
    """
    Returns a dictionary of keyword arguments for subprocess calls, adding platform-specific
    flags that we want to use consistently.
    """
    kwargs = {}
    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore
    return kwargs


def quote_arg(arg: str) -> str:
    """Safely quote a single argument for shell command strings."""
    return shlex.quote(arg)
