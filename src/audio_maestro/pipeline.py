"""Pitched notes in source seconds, with reviewed beat mapping and portable MIDI."""

from __future__ import annotations

import json
import math
import struct
from itertools import pairwise
from pathlib import Path
from typing import Any

from audio_maestro import basic_pitch
from audio_maestro.reference import features, read
from audio_maestro.timeline import to_beat, validate_map, vlq
from audio_maestro.transcription import Event, clean


def beat_notes(
    events: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    origin: float = 0,
    velocity: int = 100,
) -> list[dict[str, Any]]:
    mapping = validate_map(anchors)
    if not math.isfinite(origin) or origin < 0 or not 1 <= velocity <= 127:
        raise ValueError("Origin must be finite/nonnegative and velocity 1..127.")
    notes = []
    for event in events:
        start, end = float(event["start_seconds"]), float(event["end_seconds"])
        pitch = event["pitch"]
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
            or isinstance(pitch, bool)
            or not isinstance(pitch, int)
            or not 0 <= pitch <= 127
        ):
            raise ValueError("Notes require finite increasing seconds and integer MIDI pitch.")
        if start < mapping[0]["seconds"] or end > mapping[-1]["seconds"]:
            raise ValueError("Reviewed anchors must cover every note; extrapolation is refused.")
        first, last = to_beat(start, mapping) - origin, to_beat(end, mapping) - origin
        if first < -1e-9:
            raise ValueError("Note falls before the clip origin.")
        notes.append(
            {
                "pitch": pitch,
                "start_time": max(0, first),
                "duration": last - first,
                "velocity": velocity,
            }
        )
    return sorted(notes, key=lambda n: (n["start_time"], n["pitch"]))


def write_midi(path: Path, notes: list[dict[str, Any]], anchors: list[dict[str, Any]]) -> None:
    """SMF type 0/channel 1 at 960 PPQ; map is piecewise constant tempo."""
    mapping = validate_map(anchors)
    if mapping[0]["beat"] != 0:
        raise ValueError("MIDI mapping must begin at beat zero.")
    events: list[tuple[int, int, bytes]] = []
    for left, right in pairwise(mapping):
        micros = round(
            1_000_000 * (right["seconds"] - left["seconds"]) / (right["beat"] - left["beat"])
        )
        if not 1 <= micros <= 0xFFFFFF:
            raise ValueError("Tempo cannot be represented in MIDI.")
        events.append((round(left["beat"] * 960), 0, b"\xff\x51\x03" + micros.to_bytes(3, "big")))
    for note in notes:
        first = round(note["start_time"] * 960)
        last = max(first + 1, round((note["start_time"] + note["duration"]) * 960))
        events.extend(
            [
                (first, 2, bytes([0x90, note["pitch"], note["velocity"]])),
                (last, 1, bytes([0x80, note["pitch"], 0])),
            ]
        )
    stream, previous = bytearray(), 0
    for tick, _, message in sorted(events):
        stream.extend(vlq(tick - previous) + message)
        previous = tick
    stream.extend(b"\0\xff\x2f\0")
    path.write_bytes(
        b"MThd"
        + struct.pack(">IHHH", 6, 0, 1, 960)
        + b"MTrk"
        + struct.pack(">I", len(stream))
        + stream
    )


def transcribe(request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    output = Path(request["output_dir"])
    samples, rate, source = read(
        request["audio_path"], request["start_seconds"], request["end_seconds"]
    )
    clip = output / "excerpt.wav"
    sf.write(clip, samples, rate, subtype="FLOAT")
    print(
        json.dumps({"event": "progress", "progress": 0.1, "stage": "excerpt prepared"}), flush=True
    )
    backend = request["backend"]
    print(
        json.dumps(
            {"event": "progress", "progress": 0.25, "stage": "transcribing with " + backend}
        ),
        flush=True,
    )
    provenance: dict[str, Any] = {"backend": backend, "model_invoked": bool(np.any(samples))}
    if not np.any(samples):
        heard = []
        provenance["basis"] = "Measured zero-valued samples; model not invoked."
    elif backend == "basic_pitch":
        heard = basic_pitch.transcribe(clip, provenance=provenance)
        provenance.update(
            {
                "device": "cpu",
                "basis": "Basic Pitch pitched-note estimates; activation is not MIDI velocity.",
            }
        )
    elif backend == "sheetsage":
        from audio_maestro.sheetsage import transcribe as score

        heard, native = score(clip, output)
        provenance.update(native)
        provenance["native_events_time_basis"] = "excerpt-local seconds"
    else:
        raise ValueError("Unknown transcription backend.")
    duration = source["frames"] / rate
    bounded = []
    for note in heard:
        if (
            not all(math.isfinite(v) for v in (note.start_s, note.end_s, note.amplitude))
            or not 0 <= note.pitch <= 127
            or not 0 <= note.amplitude <= 1
            or note.end_s <= note.start_s
        ):
            raise ValueError("Model emitted invalid note values.")
        start, end = max(0, note.start_s), min(duration, note.end_s)
        if end > start:
            bounded.append(Event(start, end, note.pitch, note.amplitude))
    kept = (
        clean(bounded, monophonic=request["monophonic"], merge=request["merge"])
        if request["cleanup"]
        else bounded
    )
    offset = request["source_offset_seconds"] + source["start_seconds"]

    def rows(events: list[Event]) -> list[dict[str, Any]]:
        return [
            {
                "pitch": n.pitch,
                "start_seconds": offset + n.start_s,
                "end_seconds": offset + n.end_s,
                "activation": n.amplitude,
            }
            for n in events
        ]

    raw, notes = rows(bounded), rows(kept)
    anchors = request.get("beat_anchors")
    coverage = request["source_offset_seconds"] + source["end_seconds"]
    # A neutral 120-BPM carrier preserves seconds. It is not an inferred song tempo.
    midi_map = anchors or [
        {"seconds": 0, "beat": 0},
        {"seconds": max(coverage, 0.01), "beat": max(coverage, 0.01) * 2},
    ]
    midi_notes = beat_notes(notes, midi_map, velocity=request["velocity"])
    midi = output / "notes.mid"
    if midi_map[0]["beat"] != 0:
        # Export relative to the reviewed first anchor without inventing a prior tempo.
        first_beat = midi_map[0]["beat"]
        export_map = [{**a, "beat": a["beat"] - first_beat} for a in midi_map]
        export_notes = [{**n, "start_time": n["start_time"] - first_beat} for n in midi_notes]
    else:
        export_map, export_notes = midi_map, midi_notes
    write_midi(midi, export_notes, export_map)
    profile = features(samples, rate)
    note_attacks = []
    for note in sorted(kept, key=lambda n: n.start_s):
        if not note_attacks or note.start_s - note_attacks[-1] > 0.03:
            note_attacks.append(note.start_s)
    return {
        "schema": "maestro.transcription",
        "schema_version": 1,
        "source": {**source, "source_offset_seconds": request["source_offset_seconds"]},
        "provenance": provenance,
        "raw_notes": raw,
        "notes": notes,
        "cleanup": {
            "enabled": request["cleanup"],
            "monophonic": request["monophonic"],
            "merge": request["merge"],
        },
        "beat_notes": midi_notes if anchors else None,
        "timing": {
            "anchors": anchors,
            "review_required": not bool(anchors),
            "midi_timing_basis": "reviewed source anchors"
            if anchors
            else "120 BPM carrier preserves seconds; not detected tempo",
            "midi_origin_source_seconds": midi_map[0]["seconds"],
            "midi_origin_source_beat": midi_map[0]["beat"],
            "midi_ppq": 960,
        },
        "velocity": request["velocity"],
        "velocity_basis": "caller-supplied constant; not recovered dynamics",
        "onset_check": {
            "audio_candidates": len(profile["onsets_seconds"]),
            "transcribed_attacks": len(note_attacks),
            "basis": "Energy-rise versus note attacks; mismatch is not an accuracy score.",
        },
        "review_required": True,
        "midi_path": str(midi),
        "audio_paths": [str(clip)],
        "limitations": [
            "Pitch and note ends are model estimates.",
            "No mixed-kit drum classification, sound patch or effects-chain recovery.",
            "Chords, key and beats in SheetSage native events are hypotheses.",
        ],
    }
