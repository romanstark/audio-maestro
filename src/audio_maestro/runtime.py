"""Explicit model provisioning; tool discovery and inference never download weights."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from audio_maestro.jobs import atomic, data_root
from audio_maestro.reference import fingerprint

DEPENDENCIES = [
    "basic-pitch==0.4.0",
    "numpy>=1.26,<2",
    "onnxruntime==1.20.1",
    "setuptools==80.9.0",
    # SciPy 1.15 wheels for Python 3.10 hold a Mach-O section that macOS 27 refuses to load.
    "scipy<1.15; sys_platform == 'darwin'",
]
SHEETSAGE_LICENSE = "https://huggingface.co/audio-cpp/SheetSage2-GGUF#license"


def transcription_interpreter() -> Path:
    return (
        data_root()
        / "basic-pitch/venv"
        / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    )


def setup_basic_pitch(confirm: bool) -> dict[str, Any]:
    python = transcription_interpreter()
    plan = {
        "backend": "basic_pitch",
        "python": str(python),
        "dependencies": DEPENDENCIES,
        "device": "cpu",
        "audio_uploads": False,
    }
    if not confirm:
        return {"ready": False, "executed": False, "plan": plan}
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("Install uv before provisioning Basic Pitch.")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if not python.is_file():
        subprocess.run(
            [uv, "venv", "--managed-python", "--python", "3.10", str(python.parent.parent)],
            check=True,
            creationflags=flags,
        )
    subprocess.run(
        [uv, "pip", "install", "--python", str(python), *DEPENDENCIES],
        check=True,
        creationflags=flags,
    )
    probe = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            (
                "from basic_pitch import FilenameSuffix,build_icassp_2022_model_path; "
                "from basic_pitch.inference import Model; "
                "Model(build_icassp_2022_model_path(FilenameSuffix.onnx))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        creationflags=flags,
    )
    return {"ready": probe.returncode == 0, "executed": True, **plan}


def sheetsage_config() -> dict[str, Any] | None:
    path = data_root() / "sheetsage.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def native_environment(library_dir: str = "") -> dict[str, str]:
    env = dict(os.environ)
    if library_dir:
        env["PATH"] = library_dir + os.pathsep + env.get("PATH", "")
    return env


def setup_sheetsage(
    executable: str,
    model_dir: str,
    backend: str = "cpu",
    confirm: bool = False,
    library_dir: str = "",
) -> dict[str, Any]:
    if backend not in {"cpu", "cuda", "metal"}:
        raise ValueError("SheetSage backend must be cpu, cuda or metal.")
    if not executable or not model_dir:
        raise ValueError("Supply the audio.cpp executable and SheetSage2 model directory.")
    binary = Path(executable).expanduser().resolve()
    model = Path(model_dir).expanduser().resolve()
    weights = model / "sheetsage2-orig.gguf"
    plan = {
        "backend": "sheetsage",
        "executable": str(binary),
        "model_dir": str(model),
        "device": backend,
        "license_url": SHEETSAGE_LICENSE,
        "weight_context_mb": 128,
        "decoder_graph_arena_mb": 128,
        "audio_uploads": False,
        "downloads": False,
        "library_dir": str(Path(library_dir).expanduser().resolve()) if library_dir else "",
    }
    if not confirm:
        return {"ready": False, "executed": False, "plan": plan}
    if not binary.is_file() or not weights.is_file():
        raise ValueError("Executable and sheetsage2-orig.gguf must already exist locally.")
    if library_dir and not Path(plan["library_dir"]).is_dir():
        raise ValueError("Native library directory does not exist.")
    with weights.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise ValueError("SheetSage weights are not a GGUF file.")
    probe = subprocess.run(
        [str(binary), "--help"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        env=native_environment(plan["library_dir"]),
    )
    help_text = probe.stdout + probe.stderr
    if not all(flag in help_text for flag in ("--task", "--family", "--session-option")):
        raise ValueError("This executable does not advertise the required audio.cpp CLI interface.")
    config = {
        **plan,
        "executable_sha256": fingerprint(binary),
        "model_sha256": fingerprint(weights),
    }
    atomic(data_root() / "sheetsage.json", config)
    return {
        "ready": True,
        "executed": True,
        **config,
        "basis": "Local files, hashes and CLI interface; actual inference checked separately.",
    }


def status() -> dict[str, Any]:
    from audio_maestro.basic_pitch import default_python

    python = default_python()
    config = sheetsage_config()
    ready = bool(
        config
        and Path(config["executable"]).is_file()
        and Path(config["model_dir"], "sheetsage2-orig.gguf").is_file()
    )
    return {
        "ok": True,
        "basic_pitch": {
            "configured": python is not None,
            "python": str(python) if python else None,
            "device": "cpu",
        },
        "sheetsage": {"configured": ready, "config": config},
        "basis": "Configuration and file presence; not an inference health check.",
        "audio_uploads": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("basic_pitch", "sheetsage"), default="basic_pitch")
    parser.add_argument("--executable", default="")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--library-dir", default="")
    parser.add_argument("--device", choices=("cpu", "cuda", "metal"), default="cpu")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    try:
        result = (
            setup_basic_pitch(args.confirm)
            if args.backend == "basic_pitch"
            else setup_sheetsage(
                args.executable, args.model_dir, args.device, args.confirm, args.library_dir
            )
        )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(json.dumps({"ready": False, "error": str(error)}))
        raise SystemExit(1) from None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
