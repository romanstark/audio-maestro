"""Original-source offsets, explicit velocities, silence gates and portable MIDI."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_maestro import basic_pitch, jobs, server
from audio_maestro.jobs import AudioJobs
from audio_maestro.pipeline import beat_notes, transcribe
from audio_maestro.transcription import Event


def request(tmp_path: Path, *, silence: bool = False, backend: str = "basic_pitch") -> dict:
    rate = 16000
    x = (
        np.zeros(rate * 4)
        if silence
        else 0.1 * np.sin(2 * np.pi * 220 * np.arange(rate * 4) / rate)
    )
    path = tmp_path / "input.wav"
    sf.write(path, x, rate)
    return {
        "audio_path": str(path),
        "output_dir": str(tmp_path),
        "kind": "transcription",
        "start_seconds": 1,
        "end_seconds": 3,
        "source_offset_seconds": 10,
        "backend": backend,
        "beat_anchors": [{"seconds": 10, "beat": 0}, {"seconds": 14, "beat": 8}],
        "monophonic": False,
        "merge": False,
        "cleanup": False,
        "velocity": 83,
    }


def test_excerpt_offset_maps_notes_without_treating_activation_as_velocity(tmp_path, monkeypatch):
    monkeypatch.setattr(basic_pitch, "transcribe", lambda path, **kwargs: [Event(0.5, 1, 57, 0.05)])
    report = transcribe(request(tmp_path))
    assert report["notes"] == [
        {"pitch": 57, "start_seconds": 11.5, "end_seconds": 12, "activation": 0.05}
    ]
    assert report["beat_notes"] == [{"pitch": 57, "start_time": 3, "duration": 1, "velocity": 83}]
    assert report["review_required"] and not report["timing"]["review_required"]
    midi = Path(report["midi_path"]).read_bytes()
    assert midi.startswith(b"MThd") and b"\x90\x39\x53" in midi
    assert midi.endswith(b"\x00\xff\x2f\x00")
    assert sf.info(report["audio_paths"][0]).frames == 32000


@pytest.mark.parametrize("backend", ["basic_pitch", "sheetsage"])
def test_zero_valued_excerpt_never_invokes_either_model(tmp_path, monkeypatch, backend):
    from audio_maestro import sheetsage

    monkeypatch.setattr(basic_pitch, "transcribe", lambda path, **kwargs: pytest.fail("invoked BP"))
    monkeypatch.setattr(sheetsage, "transcribe", lambda *args: pytest.fail("invoked SheetSage"))
    report = transcribe(request(tmp_path, silence=True, backend=backend))
    assert not report["provenance"]["model_invoked"]
    assert report["notes"] == []
    assert Path(report["midi_path"]).is_file()


def test_optional_backend_is_explicit_and_retains_provenance(tmp_path, monkeypatch):
    from audio_maestro import sheetsage

    monkeypatch.setattr(
        basic_pitch, "transcribe", lambda path, **kwargs: pytest.fail("wrong backend")
    )
    monkeypatch.setattr(
        sheetsage,
        "transcribe",
        lambda *args: (
            [Event(0, 0.5, 69)],
            {"backend": "sheetsage", "model_sha256": "verified-test"},
        ),
    )
    report = transcribe(request(tmp_path, backend="sheetsage"))
    assert report["provenance"]["model_sha256"] == "verified-test"
    assert report["notes"][0]["start_seconds"] == 11


def test_neutral_midi_carrier_is_not_claimed_as_song_tempo(tmp_path, monkeypatch):
    monkeypatch.setattr(basic_pitch, "transcribe", lambda path, **kwargs: [Event(0, 0.5, 57)])
    data = request(tmp_path)
    data["beat_anchors"] = None
    report = transcribe(data)
    assert report["beat_notes"] is None
    assert report["timing"]["review_required"]
    assert "not detected tempo" in report["timing"]["midi_timing_basis"]


@pytest.mark.parametrize("event", [Event(0, 1, 128), Event(0, 1, 57, float("nan"))])
def test_invalid_model_values_fail_instead_of_publishing_midi(tmp_path, monkeypatch, event):
    monkeypatch.setattr(basic_pitch, "transcribe", lambda path, **kwargs: [event])
    with pytest.raises(ValueError, match="invalid note"):
        transcribe(request(tmp_path))


def test_mapping_requires_coverage_and_preserves_variable_tempo():
    anchors = [{"seconds": 0, "beat": 0}, {"seconds": 1, "beat": 2}, {"seconds": 3, "beat": 4}]
    note = {"pitch": 60, "start_seconds": 0.5, "end_seconds": 2}
    assert beat_notes([note], anchors) == [
        {"pitch": 60, "start_time": 1, "duration": 2, "velocity": 100}
    ]
    with pytest.raises(ValueError, match="cover"):
        beat_notes([{**note, "end_seconds": 4}], anchors)
    with pytest.raises(ValueError, match="before"):
        beat_notes([note], anchors, origin=2)


def test_preview_and_status_create_no_jobs_or_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_MAESTRO_DATA_DIR", str(tmp_path / "state"))
    monkeypatch.setenv(basic_pitch.ENV_VAR, str(tmp_path / "absent"))
    monkeypatch.setattr(jobs, "manager", lambda: pytest.fail("created jobs"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("ran setup"))
    assert server.audio_status()["ok"]
    assert server.setup_backend()["executed"] is False
    assert server.transcribe_audio("missing.wav", "out")["executed"] is False
    assert server.analyze_reference_audio("missing.wav", "out")["executed"] is False
    assert not (tmp_path / "state").exists()


def test_completed_report_pages_notes_without_dumping_native_events(tmp_path, monkeypatch):
    manager = AudioJobs(tmp_path / "state")
    identifier = "a" * 32
    run = tmp_path / "run"
    run.mkdir()
    report = {
        "schema": "maestro.transcription",
        "schema_version": 1,
        "notes": [{"pitch": 60}] * 3,
        "provenance": {"backend": "sheetsage", "native_events": list(range(1000))},
    }
    path = run / "report.json"
    path.write_text(json.dumps(report))
    manager.jobs[identifier] = {
        "id": identifier,
        "state": "succeeded",
        "result": {"report_path": str(path)},
    }
    monkeypatch.setattr(jobs, "manager", lambda: manager)
    try:
        summary = server.get_audio_report(identifier)["report"]
        assert "native_events" not in summary["provenance"] and "notes" not in summary
        page = server.get_audio_report(identifier, section="notes", limit=2)
        assert len(page["rows"]) == 2 and page["next_offset"] == 2
    finally:
        manager.close()
