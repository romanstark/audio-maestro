"""Voice tracking and section-wise onset correction on synthetic melodies."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from audio_maestro.lines import align, track

RATE = 22050
#: 120 BPM: one beat is half a second.
ANCHORS = [{"seconds": 0.0, "beat": 0.0}, {"seconds": 40.0, "beat": 80.0}]


def melody(np, pitches: Sequence[int | None], beats_each: float = 1.0, duck: bool = False):
    """Return harmonic tones, one per entry; None is a rest. duck fades each attack in."""
    rng = np.random.default_rng(11)
    length = round(RATE * 0.5 * beats_each)
    t = np.arange(length) / RATE
    parts = []
    for pitch in pitches:
        if pitch is None:
            parts.append(np.zeros(length))
            continue
        frequency = 440 * 2 ** ((pitch - 69) / 12)
        tone = sum(0.3 / k * np.sin(2 * np.pi * k * frequency * t) for k in (1, 2, 3, 4))
        if duck:
            tone *= np.minimum(1, 10 ** (-1.5 + 1.5 * t / 0.25))
        parts.append(tone)
    return np.concatenate(parts) + rng.normal(0, 0.001, len(parts) * length)


def test_track_follows_a_melody_and_its_rests() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    line = [60, 62, 64, None, 65, 67, 65, 64]
    notes = track(melody(np, line), RATE, 0.0, ANCHORS, 0.0, 8.0, 55, 72)
    assert [n["pitch"] for n in notes] == [p for p in line if p is not None]
    assert [n["start_beat"] for n in notes] == pytest.approx([0, 1, 2, 4, 5, 6, 7], abs=0.25)


def test_ducked_attack_still_starts_on_its_beat() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    notes = track(
        melody(np, [40, 43, 40, 45], duck=True),
        RATE,
        0.0,
        ANCHORS,
        0.0,
        4.0,
        33,
        52,
        voice="bottom",
    )
    assert [n["start_beat"] for n in notes] == pytest.approx([0, 1, 2, 3])


def test_align_shifts_each_section_by_its_own_drift() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    pitches = [60, 62, 64, 65, 67, 65, 64, 62] * 2
    audio = melody(np, pitches)
    line = track(audio, RATE, 0.0, ANCHORS, 0.0, 16.0, 55, 72)
    # Section 0 is on time, section 1 (beats 8..16) comes 0.6 beat late from the model.
    model = [
        {
            "pitch": p,
            "start_seconds": 0.5 * (b + (0.6 if b >= 8 else 0)),
            "end_seconds": 0.5 * (b + 0.9 + (0.6 if b >= 8 else 0)),
        }
        for b, p in enumerate(pitches)
    ]
    moved, sections = align(model, line, ANCHORS, section_beats=8)
    assert [s["shift_beats"] for s in sections] == pytest.approx([0, -0.6], abs=0.05)
    starts = [n["start_seconds"] / 0.5 for n in moved]
    assert starts == pytest.approx(list(range(16)), abs=0.05)
    assert moved[9]["original_start_seconds"] == pytest.approx(0.5 * 9.6)


def test_section_without_enough_matches_stays_put() -> None:
    model = [{"pitch": 60, "start_seconds": 0.5 * 2.3, "end_seconds": 0.5 * 3}]
    line = [{"pitch": 61, "start_beat": 2.0}]
    moved, sections = align(model, line, ANCHORS)
    assert sections[0]["shift_beats"] is None
    assert moved[0]["start_seconds"] == pytest.approx(0.5 * 2.3)


def test_transcription_aligns_late_model_notes(tmp_path, monkeypatch) -> None:
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("scipy")
    from audio_maestro import basic_pitch
    from audio_maestro.pipeline import transcribe
    from audio_maestro.transcription import Event

    pitches = [60, 62, 64, 65, 67, 65, 64, 62] * 2
    sf.write(tmp_path / "line.wav", melody(np, pitches), RATE, subtype="FLOAT")
    late = [Event(0.5 * b + 0.3, 0.5 * b + 0.75, p, 0.5) for b, p in enumerate(pitches)]
    monkeypatch.setattr(basic_pitch, "transcribe", lambda path, **kwargs: late)
    report = transcribe(
        {
            "kind": "transcription",
            "audio_path": str(tmp_path / "line.wav"),
            "output_dir": str(tmp_path),
            "backend": "basic_pitch",
            "start_seconds": 0,
            "end_seconds": 8,
            "source_offset_seconds": 0,
            "beat_anchors": ANCHORS,
            "monophonic": False,
            "merge": False,
            "cleanup": False,
            "velocity": 100,
            "align_to_audio": True,
            "align_voice": "top",
            "align_low_note": None,
            "align_high_note": None,
            "align_scale": None,
            "align_section_beats": 8,
            "align_snap_beats": None,
        }
    )
    assert [s["shift_beats"] for s in report["alignment"]["sections"]] == pytest.approx(
        [-0.6, -0.6], abs=0.05
    )
    assert report["notes"][3]["start_seconds"] == pytest.approx(1.5, abs=0.03)
    assert report["unaligned_notes"][3]["start_seconds"] == pytest.approx(1.8)
