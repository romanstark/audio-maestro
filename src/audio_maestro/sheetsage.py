"""Optional audio.cpp SheetSage2 adapter; bounded memory and explicit provenance."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from audio_maestro.reference import fingerprint
from audio_maestro.runtime import native_environment, sheetsage_config
from audio_maestro.transcription import Event


def transcribe(source: Path, output: Path) -> tuple[list[Event], dict[str, Any]]:
    config = sheetsage_config()
    if not config:
        raise RuntimeError(
            "SheetSage is optional and unconfigured; run audio-maestro-setup "
            "--backend sheetsage --executable PATH --model-dir PATH --confirm."
        )
    binary = Path(config["executable"])
    weights = Path(config["model_dir"], "sheetsage2-orig.gguf")
    if (
        fingerprint(binary) != config["executable_sha256"]
        or fingerprint(weights) != config["model_sha256"]
    ):
        raise RuntimeError("SheetSage runtime/model changed; explicitly register it again.")
    command = [
        str(binary),
        "--task",
        "midi",
        "--family",
        "sheetsage2",
        "--model",
        config["model_dir"],
        "--backend",
        config["device"],
        "--threads",
        str(min(8, os.cpu_count() or 1)),
        "--session-option",
        "sheetsage2.weight_context_mb=128",
        "--session-option",
        "sheetsage2.decoder_graph_arena_mb=128",
        "--audio",
        str(source),
        "--out-dir",
        str(output),
    ]
    with (output / "sheetsage.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            check=True,
            stdout=log,
            stderr=log,
            timeout=600,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env=native_environment(config.get("library_dir", "")),
        )
    raw = json.loads((output / "events.json").read_text(encoding="utf-8"))
    payload = json.loads(bytes.fromhex(raw["payload_hex"]))
    notes = [
        Event(float(event["time"]), float(note["end_time"]), int(note["pitch"]))
        for event in payload["events"]
        for note in event.get("notes", [])
    ]
    return notes, {
        "backend": "sheetsage",
        "device": config["device"],
        "executable_sha256": config["executable_sha256"],
        "model_sha256": config["model_sha256"],
        "license_url": config["license_url"],
        "native_events": payload["events"],
        "basis": "SheetSage2 lead-sheet estimates, not all polyphonic instrument voices.",
        "review_required": True,
    }
