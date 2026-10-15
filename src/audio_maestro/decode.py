"""Open sources libsndfile cannot read through the platform's own decoder.

libsndfile reads WAV, FLAC, AIFF, OGG and MP3 everywhere. On macOS, AAC/ALAC in
M4A/CAF (Voice Memos, GarageBand, Music) is decoded by Core Audio's ``afconvert``,
which ships with the system and honours encoder delay. Other platforms keep the
libsndfile error. The decoded copy is temporary; callers keep fingerprinting the
original file.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

#: A decode of a long song takes seconds; this is a stuck process.
TIMEOUT_S = 300


@contextmanager
def readable(path: str | Path) -> Iterator[tuple[Path, str | None]]:
    """Yield a path libsndfile opens, and the decoder used, or None for the original.

    Raises:
        ValueError: The format is unsupported on this platform or decoding failed.
    """
    import soundfile as sf

    source = Path(path)
    try:
        sf.info(str(source))
    except sf.LibsndfileError as error:
        converter = shutil.which("afconvert") if sys.platform == "darwin" else None
        if converter is None:
            raise ValueError(
                f"Cannot read {source.name}: {error}. Use WAV, FLAC, AIFF, OGG or MP3"
                + ("; afconvert is missing." if sys.platform == "darwin" else ".")
            ) from error
        with tempfile.TemporaryDirectory(prefix="audio-maestro-decode-") as folder:
            target = Path(folder) / "decoded.wav"
            try:
                completed = subprocess.run(
                    [converter, "-f", "WAVE", "-d", "LEF32", str(source), str(target)],
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT_S,
                    check=False,
                )
            except subprocess.TimeoutExpired as expired:
                raise ValueError(f"afconvert did not decode {source.name} in time.") from expired
            if completed.returncode or not target.is_file():
                detail = (completed.stderr or completed.stdout).strip().splitlines()
                raise ValueError(
                    f"Cannot read {source.name}: afconvert failed"
                    + (f": {detail[-1]}" if detail else ".")
                ) from error
            yield target, "afconvert"
        return
    yield source, None
