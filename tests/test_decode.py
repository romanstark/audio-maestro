"""Platform decoding: libsndfile formats untouched, macOS Core Audio for AAC/ALAC."""

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from audio_maestro import decode
from audio_maestro.reference import fingerprint, read

RATE = 44100
macos = pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("afconvert") is None,
    reason="Core Audio afconvert is macOS-only",
)


def click_track(path: Path, at: float = 1.0) -> None:
    x = np.zeros(RATE * 3)
    t = np.arange(RATE // 10) / RATE
    start = round(at * RATE)
    x[start : start + len(t)] = 0.8 * np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 40)
    sf.write(path, np.stack([x, x], axis=1), RATE, subtype="PCM_16")


def test_libsndfile_formats_are_read_in_place(tmp_path: Path):
    path = tmp_path / "click.wav"
    click_track(path)
    with decode.readable(path) as (decoded, decoder):
        assert decoded == path and decoder is None


def test_unsupported_format_names_the_alternatives(tmp_path: Path, monkeypatch):
    path = tmp_path / "voice.m4a"
    path.write_bytes(b"\0\0\0\x20ftypM4A " + bytes(64))
    monkeypatch.setattr(decode, "sys", SimpleNamespace(platform="win32"))
    with pytest.raises(ValueError, match="WAV, FLAC, AIFF, OGG or MP3"), decode.readable(path):
        pass


@macos
def test_aac_is_decoded_with_encoder_delay_removed(tmp_path: Path):
    wav, m4a = tmp_path / "click.wav", tmp_path / "Sprachmemo ä.m4a"
    click_track(wav)
    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", str(wav), str(m4a)], check=True)
    with decode.readable(m4a) as (decoded, decoder):
        assert decoder == "afconvert" and decoded != m4a
        samples, rate = sf.read(decoded, always_2d=True)
        folder = decoded.parent
    assert not folder.exists()
    assert rate == RATE and samples.shape[1] == 2
    assert abs(len(samples) - RATE * 3) < 2048
    attack = int(np.flatnonzero(np.abs(samples[:, 0]) > 0.1)[0])
    assert abs(attack - RATE) < RATE * 0.002


@macos
def test_decoded_source_keeps_original_identity(tmp_path: Path):
    wav, m4a = tmp_path / "click.wav", tmp_path / "click.m4a"
    click_track(wav)
    subprocess.run(["afconvert", "-f", "m4af", "-d", "alac", str(wav), str(m4a)], check=True)
    samples, rate, info = read(str(m4a), 0.5, 2.0)
    assert info["path"] == str(m4a) and info["sha256"] == fingerprint(m4a)
    assert info["decoded_by"] == "afconvert" and rate == RATE
    expected, _ = sf.read(wav, start=RATE // 2, stop=RATE * 2, always_2d=True)
    assert np.allclose(samples, expected, atol=1e-4)


@macos
def test_undecodable_file_reports_afconvert_failure(tmp_path: Path):
    path = tmp_path / "broken.m4a"
    path.write_bytes(b"not audio at all" * 64)
    with pytest.raises(ValueError, match="afconvert failed"), decode.readable(path):
        pass
