"""Mixed-kit classification on a synthetic kit with a known halftime and triplet pattern."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audio_maestro.kit import analyze_kit
from audio_maestro.timeline import notes_from_analysis

RATE = 44100
#: 120 BPM for four beats, then 100 BPM, so the grid must follow the anchors. Beat 8
#: carries kick and snare together.
ANCHORS = [
    {"seconds": 0.5, "beat": 0.0},
    {"seconds": 2.5, "beat": 4.0},
    {"seconds": 7.3, "beat": 12.0},
]
KICKS = [0.0, 2.5, 4.0, 6.5, 8.0, 10.5]
SNARES = [2.0, 6.0, 8.0, 10.0]
HATS = [1.0, 1.0 + 1 / 3, 1.0 + 2 / 3, 5.0, 9.0, 11.0]


def seconds(beat: float) -> float:
    return 0.5 + (beat * 0.5 if beat <= 4 else 2.0 + (beat - 4) * 0.6)


def kit(path: Path, gain: float = 1.0):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    signal = pytest.importorskip("scipy.signal")
    rng = np.random.default_rng(7)
    out = rng.normal(0, 0.002, round(RATE * 8))
    t = np.arange(round(RATE * 0.25)) / RATE
    kick = np.sin(2 * np.pi * (50 + 50 * np.exp(-t * 40)) * t) * np.exp(-t * 18)
    noise = rng.normal(0, 1, len(t))
    snare = 0.6 * np.sin(2 * np.pi * 200 * t) * np.exp(-t * 25) + 0.5 * signal.sosfilt(
        signal.butter(4, [1500, 5000], "band", fs=RATE, output="sos"), noise
    ) * np.exp(-t * 30)
    hat = (
        0.4
        * signal.sosfilt(signal.butter(4, 7000, "high", fs=RATE, output="sos"), noise)
        * np.exp(-t * 80)
    )
    for beats, sound in ((KICKS, kick), (SNARES, snare), (HATS, hat)):
        for beat in beats:
            start = round(seconds(beat) * RATE)
            out[start : start + len(sound)] += sound[: len(out) - start]
    sf.write(path, gain * 0.5 * out, RATE, subtype="FLOAT")


def request(tmp_path: Path, audio: Path) -> dict:
    return {
        "kind": "kit",
        "audio_path": str(audio),
        "output_dir": str(tmp_path),
        "beat_anchors": ANCHORS,
        "source_offset_seconds": 0.0,
        "from_beat": None,
        "to_beat": None,
        "subdivisions": [4, 3],
        "beats_per_bar": 4,
        "pitches": {"kick": 36, "snare": 38, "hat": 42},
        "rules": {},
    }


def found(report: dict, instrument: str) -> list[float]:
    return [event["beat"] for event in report["events"] if event["instrument"] == instrument]


def test_kit_pattern_follows_the_tempo_map(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    kit(tmp_path / "kit.wav")
    report = analyze_kit(request(tmp_path, tmp_path / "kit.wav"))
    assert found(report, "kick") == pytest.approx(KICKS, abs=1e-6)
    assert found(report, "snare") == pytest.approx(SNARES, abs=1e-6)
    assert found(report, "hat") == pytest.approx(HATS, abs=1e-6)
    triplets = [event for event in report["events"] if event["division"] == 3]
    assert {round(event["beat"] % 1, 3) for event in triplets} == {0.333, 0.667}
    assert "bar 1 1/4 K x---------x-----" in [line.replace("X", "x") for line in report["pattern"]]
    data = json.loads(Path(report["drum_events_path"]).read_text(encoding="utf-8"))
    assert len(notes_from_analysis(data)) == len(report["events"])
    assert Path(report["midi_path"]).read_bytes()[:4] == b"MThd"


def test_quieter_kit_gives_the_same_pattern(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    kit(tmp_path / "loud.wav")
    kit(tmp_path / "quiet.wav", gain=10 ** (-12 / 20))
    loud = analyze_kit(request(tmp_path, tmp_path / "loud.wav"))
    quiet = analyze_kit(request(tmp_path, tmp_path / "quiet.wav"))
    assert [(e["beat"], e["instrument"]) for e in loud["events"]] == [
        (e["beat"], e["instrument"]) for e in quiet["events"]
    ]
