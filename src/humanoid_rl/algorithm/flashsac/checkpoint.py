"""Publish replay pairs atomically, independently of a usable model checkpoint."""

import json
from pathlib import Path
import shutil
import tempfile


def write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def replay_directory(checkpoint):
    """New complete replay directories and legacy checkpoints are both readable."""
    checkpoint = Path(checkpoint)
    for directory in (checkpoint / "replay", checkpoint):
        if (directory / "replay_buffer.pt").is_file():
            return directory
    return None


def save_replay_atomic(agent, checkpoint):
    """Publish the entire directory only after all replay files were written."""
    checkpoint = Path(checkpoint)
    temporary = Path(tempfile.mkdtemp(prefix=".replay-writing-", dir=checkpoint))
    destination = checkpoint / "replay"
    try:
        agent.save_replay_buffer(str(temporary))
        if not (temporary / "replay_buffer.pt").is_file():
            raise RuntimeError("Replay writer returned without replay_buffer.pt")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def is_replay_io_error(error):
    # PyTorch wraps low-level stream/write errors in RuntimeError. Programming
    # errors must still propagate; only serialization I/O failures are optional.
    return isinstance(error, OSError) or any(message in str(error) for message in (
        "PytorchStreamWriter", "file write failed", "unexpected pos", "basic_ios", "iostream error",
        "No space left", "failed writing file",
    ))
