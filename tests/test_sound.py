"""Measured filter rate, envelope and tail on synthesized tones with known settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_maestro.sound import describe

RATE = 16000
ANCHORS = [{"seconds": 0.0, "beat": 0.0}, {"seconds": 10.0, "beat": 20.0}]


def request(path: Path, to_beat: float) -> dict:
    return {
        "kind": "sound",
        "audio_path": str(path),
        "beat_anchors": ANCHORS,
        "source_offset_seconds": 0.0,
        "from_beat": 0.0,
        "to_beat": to_beat,
        "steps_per_beat": 24,
    }


def test_filter_movement_is_named_as_a_tempo_division(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("scipy")
    t = np.arange(RATE * 4) / RATE
    # Cutoff sweeps 300 Hz to 2.4 kHz once per eighth-note triplet (1/3 beat at 120 BPM).
    cutoff = 300 * 2 ** (3 * (0.5 + 0.5 * np.sin(2 * np.pi * t / (0.5 / 3))))
    harmonics = np.arange(1, 70)
    gains = 1 / harmonics[:, None] / np.sqrt(1 + (110 * harmonics[:, None] / cutoff) ** 8)
    saw = (gains * np.sin(2 * np.pi * 110 * harmonics[:, None] * t)).sum(axis=0)
    sf.write(tmp_path / "wah.wav", 0.2 * saw, RATE, subtype="FLOAT")
    report = describe(request(tmp_path / "wah.wav", 8.0))
    modulation = report["estimates"]["filter"]["modulation"]
    assert modulation["division"] == "1/8T"
    low, high = report["estimates"]["filter"]["rolloff_hz_p10_p90"]
    assert high > 1.5 * low


def test_attack_and_decay_come_out_in_seconds(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("scipy")
    t = np.arange(RATE * 4) / RATE
    attack = np.minimum(1, t / 0.05)
    # A 60 dB fall takes 0.8 s after the note ends at 2 s.
    release = np.where(t < 2, 1.0, 10 ** (-3 * (t - 2) / 0.8))
    tone = np.sin(2 * np.pi * 220 * t) * attack * release
    noise = np.random.default_rng(2).normal(0, 1e-5, len(t))
    sf.write(tmp_path / "tone.wav", 0.3 * tone + noise, RATE, subtype="FLOAT")
    report = describe(request(tmp_path / "tone.wav", 8.0))
    envelope = report["estimates"]["envelope"]
    assert envelope["attack_seconds"] == pytest.approx(0.05, abs=0.02)
    assert envelope["decay_seconds_60db"] == pytest.approx(0.8, abs=0.1)
