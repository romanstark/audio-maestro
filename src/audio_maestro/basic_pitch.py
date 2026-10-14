"""Run Basic Pitch in its independently configured ONNX/CPU interpreter.

Configuration is read at call time. Inference never installs packages or downloads
weights. Model version and fingerprint come from the actual sidecar output.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from audio_maestro.transcription import Event

__all__ = [
    "ENV_VAR",
    "TranscriberError",
    "default_python",
    "is_available",
    "transcribe",
]

#: Names the Python of the environment ``basic-pitch`` is installed into.
ENV_VAR = "AUDIO_MAESTRO_BASIC_PITCH_PYTHON"

#: Looked at when the variable is unset, in order.
_FALLBACKS = (
    Path.home() / ".local" / "share" / "audio-maestro" / "basic-pitch" / "bin" / "python",
    Path.home() / ".local" / "share" / "audio-maestro" / "basic-pitch" / "Scripts" / "python.exe",
    Path(".venv-basic-pitch") / "bin" / "python",
    Path(".venv-basic-pitch") / "Scripts" / "python.exe",
)

#: How long one file may take. A four minute stem measured about two seconds, so this
#: is a stuck process rather than a slow one.
TIMEOUT_S = 600

#: Run inside the sidecar. Written here rather than shipped as a file because the
#: sidecar has no copy of this project and nothing to import from it.
_RUNNER = """
import json, sys, hashlib
from pathlib import Path
from importlib.metadata import version
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import basic_pitch
if not getattr(basic_pitch, "ONNX_PRESENT", False):
    raise RuntimeError("ONNX Runtime is required; run audio-maestro-setup --confirm")
model_path = basic_pitch.build_icassp_2022_model_path(basic_pitch.FilenameSuffix.onnx)
source, destination = sys.argv[1], sys.argv[2]
_, _, events = predict(source, model_or_model_path=str(model_path))
rows = [
    {"start": float(e[0]), "end": float(e[1]), "pitch": int(e[2]), "amp": float(e[3])}
    for e in events
]
with open(destination, "w", encoding="utf-8") as handle:
    json.dump({"events": rows, "provenance": {
        "version": version("basic-pitch"),
        "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        "inference_backend": "onnx" if getattr(basic_pitch, "ONNX_PRESENT", False) else "default"
    }}, handle)
"""


class TranscriberError(RuntimeError):
    """Raised when the transcriber could not be reached or could not run.

    ``code`` is the machine-readable reason, ``error`` is the sentence for a person.
    """

    def __init__(self, code: str, error: str) -> None:
        super().__init__(error)
        self.code = code
        self.error = error


def default_python() -> Path | None:
    """The interpreter to run the transcriber with, or None where none was found."""
    named = os.environ.get(ENV_VAR)
    if named:
        candidate = Path(named).expanduser()
        return candidate if candidate.is_file() else None
    for candidate in _FALLBACKS:
        if candidate.expanduser().is_file():
            return candidate.expanduser()
    from audio_maestro.runtime import transcription_interpreter

    provisioned = transcription_interpreter()
    if provisioned.is_file():
        return provisioned
    found = shutil.which("basic-pitch")
    if found:
        bp = Path(found)
        for name in (
            "python",
            "python.exe",
            "python3",
            "python3.exe",
            "python.cmd",
            "python.bat",
        ):
            py_cand = bp.parent / name
            if py_cand.is_file():
                return py_cand
    return None


def is_available() -> bool:
    """Return whether an environment to run the transcriber in was found."""
    return default_python() is not None


def transcribe(
    path: str | Path, *, python: Path | None = None, provenance: dict[str, Any] | None = None
) -> list[Event]:
    """Transcribe one audio file into note events, uncleaned.

    Args:
        path: Any audio file the transcriber reads.
        python: Interpreter of the environment holding ``basic-pitch``. Found from
            :data:`ENV_VAR` or the usual places when not given.

    Returns:
        Events in seconds, in the order the transcriber reported them. Cleaning is a
        separate step, in :mod:`audio_maestro.transcription`, because what to
        remove depends on whether the part is played one note at a time.

    Raises:
        TranscriberError: No environment was found, the file is missing, the
            transcriber failed, timed out, or wrote something unreadable. Every
            failure leaves this as a TranscriberError, so a caller has one thing to
            catch.
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise TranscriberError("no_such_file", f"{source} is not a file.")

    interpreter = python or default_python()
    if interpreter is None:
        raise TranscriberError(
            "no_transcriber",
            "No environment with basic-pitch was found. Create one with a Python "
            "between 3.7 and 3.11, install basic-pitch into it, and point "
            f"{ENV_VAR} at its interpreter.",
        )

    with tempfile.TemporaryDirectory(prefix="audio-maestro-transcribe-") as folder:
        runner = Path(folder) / "run.py"
        runner.write_text(_RUNNER)
        target = Path(folder) / "notes.json"
        cmd = [str(interpreter), str(runner), str(source), str(target)]
        if sys.platform == "win32" and str(interpreter).lower().endswith((".bat", ".cmd")):
            cmd = ["cmd.exe", "/c", *cmd]
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                timeout=TIMEOUT_S,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            raise TranscriberError(
                "transcriber_timed_out",
                f"basic-pitch did not finish with {source.name} inside {TIMEOUT_S} seconds.",
            ) from exc
        if completed.returncode != 0 or not target.is_file():
            detail = (completed.stderr or b"").decode("utf-8", "replace").strip()
            last = detail.splitlines()[-1] if detail else ""
            raise TranscriberError(
                "transcriber_failed",
                f"basic-pitch could not transcribe {source.name}" + (f": {last}" if last else "."),
            )
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            rows = payload["events"] if isinstance(payload, dict) else payload
            if provenance is not None:
                provenance.update(
                    payload.get("provenance", {})
                    if isinstance(payload, dict)
                    else {"version": None, "basis": "Sidecar did not report model metadata."}
                )
        except (OSError, ValueError) as exc:
            raise TranscriberError(
                "unreadable_result",
                f"basic-pitch wrote something this cannot read: {exc}",
            ) from exc

    return [
        Event(
            start_s=float(row["start"]),
            end_s=float(row["end"]),
            pitch=int(row["pitch"]),
            amplitude=float(row.get("amp", 1.0)),
        )
        for row in rows
    ]
