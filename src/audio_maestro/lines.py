"""One voice per register followed on a beat grid, and model note times moved onto it."""

from __future__ import annotations

import math
from typing import Any

from audio_maestro.timeline import to_beat, validate_map

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
# (weight, semitones above the fundamental)
HARMONICS = ((1.0, 0), (0.5, 12), (0.33, 19), (0.25, 24), (0.2, 28))
STEP_COST, LEAP_BASE, LEAP_PER_SEMITONE = 0.6, 0.9, 0.15
REST_COST, OUT_OF_SCALE_COST = 1.5, 1.2
SILENCE_DB = 30.0
REATTACK_RATIO = 1.8


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def beat_to_seconds(beat: float, anchors: list[dict[str, float]]) -> float:
    index = max(0, min(len(anchors) - 2, sum(1 for item in anchors if item["beat"] <= beat) - 1))
    left, right = anchors[index], anchors[index + 1]
    return left["seconds"] + (beat - left["beat"]) * (
        (right["seconds"] - left["seconds"]) / (right["beat"] - left["beat"])
    )


def step_salience(
    samples: Any,
    rate: int,
    origin: float,
    anchors: list[dict[str, float]],
    grid: Any,
    step: float,
    low: int,
    high: int,
) -> tuple[Any, Any]:
    import numpy as np
    from scipy.signal import stft

    size = 2 ** round(math.log2(rate * 0.17))
    hop = max(1, round(rate * 0.0053))
    half = size / rate / 2
    frequencies, times, spectrum = stft(samples, rate, nperseg=size, noverlap=size - hop)
    times = times + origin
    power = np.abs(spectrum) ** 2
    lowest, highest = low - 12, high + HARMONICS[-1][1]
    energy = {}
    for pitch in range(lowest, highest + 1):
        centre = 440 * 2 ** ((pitch - 69) / 12)
        selected = (frequencies > centre * 2 ** (-1 / 24)) & (frequencies < centre * 2 ** (1 / 24))
        energy[pitch] = (
            power[selected].sum(axis=0)
            if selected.any()
            else np.zeros(len(times))
            if centre >= frequencies[-1]
            else power[int(np.argmin(np.abs(frequencies - centre)))]
        )
    pitches = range(low, high + 1)
    salience = np.array(
        [sum(weight * energy[p + shift] for weight, shift in HARMONICS) for p in pitches]
    )
    below = np.array([energy[p - 12] for p in pitches])
    salience = salience - 0.5 * np.minimum(salience, 3 * below)
    starts = np.array([beat_to_seconds(float(beat), anchors) for beat in grid])
    stops = np.array([beat_to_seconds(float(beat) + step, anchors) for beat in grid])
    steps = np.zeros((len(grid), len(pitches)))
    level = np.zeros(len(grid))
    for index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        # Start inside the step so the previous note's tail stays out.
        selected = (times >= start + half) & (times <= max(stop - half, start + half + hop / rate))
        if not selected.any():
            selected = np.zeros(len(times), dtype=bool)
            selected[int(np.argmin(np.abs(times - (start + half))))] = True
        steps[index] = salience[:, selected].mean(axis=1)
        first = max(0, round((start - origin) * rate))
        last = max(first + 1, round((stop - origin) * rate))
        level[index] = float(np.mean(samples[first:last] ** 2)) if first < len(samples) else 0.0
    return steps, level


def viterbi(
    steps: Any, level: Any, pitches: Any, voice: str, scale: list[int] | None
) -> list[int | None]:
    import numpy as np

    relative = steps / (steps.max(axis=1, keepdims=True) + 1e-12)
    height = (pitches - pitches[0]) / max(1, pitches[-1] - pitches[0])
    emission = np.log(relative + 0.02) + 0.25 * (height if voice == "top" else 1 - height)
    if scale is not None:
        emission[:, [index for index, p in enumerate(pitches) if p % 12 not in scale]] -= (
            OUT_OF_SCALE_COST
        )
    loud = float(np.percentile(level, 90)) if len(level) else 0.0
    rest = np.where(level < loud * 10 ** (-SILENCE_DB / 10), 1.0, -3.0)
    count = len(pitches)
    distance = np.abs(np.subtract.outer(np.arange(count), np.arange(count)))
    moves = np.where(
        distance == 0,
        0.0,
        np.where(distance <= 2, -STEP_COST, -LEAP_BASE - LEAP_PER_SEMITONE * distance),
    )
    transition = np.full((count + 1, count + 1), -REST_COST)
    transition[:count, :count] = moves
    transition[count, count] = 0.0
    score = np.append(emission[0], rest[0])
    back = []
    for index in range(1, len(steps)):
        candidates = score[:, None] + transition
        back.append(candidates.argmax(axis=0))
        score = candidates.max(axis=0) + np.append(emission[index], rest[index])
    path = [int(score.argmax())]
    for pointer in reversed(back):
        path.append(int(pointer[path[-1]]))
    path.reverse()
    return [None if state == count else int(pitches[state]) for state in path]


def track(
    samples: Any,
    rate: int,
    origin: float,
    anchors: list[dict[str, Any]],
    from_beat: float,
    to_beat: float,
    low: int,
    high: int,
    voice: str = "top",
    scale: list[int] | None = None,
    steps_per_beat: int = 4,
) -> list[dict[str, Any]]:
    import numpy as np

    mapping = validate_map(anchors)
    if not low < high:
        raise ValueError("register needs low below high")
    step = 1 / steps_per_beat
    grid = np.arange(from_beat, to_beat - 1e-9, step)
    if len(grid) < 2:
        raise ValueError("beat range must hold at least two grid steps")
    pitches = np.arange(low, high + 1)
    steps, level = step_salience(samples, rate, origin, mapping, grid, step, low, high)
    path = viterbi(steps, level, pitches, voice, scale)
    relative = steps / (steps.max(axis=1, keepdims=True) + 1e-12)
    notes: list[dict[str, Any]] = []
    for index, pitch in enumerate(path):
        if pitch is None:
            continue
        column = pitch - low
        confidence = float(relative[index, column])
        struck = (
            index >= 2
            and path[index - 2] == pitch
            and steps[index - 1, column] < steps[index - 2, column]
            and steps[index, column] > REATTACK_RATIO * steps[index - 1, column]
        )
        held = notes and index > 0 and path[index - 1] == pitch and not struck
        if held:
            notes[-1]["duration_beats"] += step
            notes[-1]["confidences"].append(confidence)
        else:
            notes.append(
                {
                    "pitch": pitch,
                    "start_beat": float(grid[index]),
                    "duration_beats": step,
                    "confidences": [confidence],
                }
            )
    for note in notes:
        note["confidence"] = round(float(np.mean(note.pop("confidences"))), 3)
        note["note"] = note_name(note["pitch"])
        note["start_seconds"] = beat_to_seconds(note["start_beat"], mapping)
        note["end_seconds"] = beat_to_seconds(note["start_beat"] + note["duration_beats"], mapping)
    return notes


def align(
    notes: list[dict[str, Any]],
    line: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    section_beats: float = 16,
    window_beats: float = 0.75,
    snap_beats: float | None = None,
    minimum_matches: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import statistics

    mapping = validate_map(anchors)
    onsets = [(item["start_beat"], item["pitch"] % 12) for item in line]
    sections: dict[int, list[float]] = {}
    matched = []
    for note in notes:
        beat = to_beat(note["start_seconds"], mapping)
        near = [
            (abs(onset - beat), onset - beat)
            for onset, pitch_class in onsets
            if pitch_class == note["pitch"] % 12 and abs(onset - beat) <= window_beats
        ]
        delta = min(near)[1] if near else None
        matched.append((beat, delta))
        if delta is not None:
            sections.setdefault(math.floor(beat / section_beats), []).append(delta)
    shifts = {
        index: statistics.median(values)
        for index, values in sections.items()
        if len(values) >= minimum_matches
    }
    corrected = []
    for note, (beat, delta) in zip(notes, matched, strict=True):
        section = math.floor(beat / section_beats)
        shift = shifts.get(section, 0.0)
        start = beat + shift
        if snap_beats:
            start = round(start / snap_beats) * snap_beats
        length = to_beat(note["end_seconds"], mapping) - beat
        corrected.append(
            {
                **note,
                "start_seconds": beat_to_seconds(start, mapping),
                "end_seconds": beat_to_seconds(start + length, mapping),
                "original_start_seconds": note["start_seconds"],
                "shift_beats": round(start - beat, 4),
                "line_delta_beats": None if delta is None else round(delta, 4),
            }
        )
    table = [
        {
            "section": index,
            "from_beat": index * section_beats,
            "to_beat": (index + 1) * section_beats,
            "matches": len(sections.get(index, [])),
            "shift_beats": round(shifts[index], 4) if index in shifts else None,
        }
        for index in sorted({math.floor(beat / section_beats) for beat, _ in matched})
    ]
    return corrected, table


def track_voice(request: dict[str, Any]) -> dict[str, Any]:
    from pathlib import Path

    from audio_maestro.pipeline import beat_notes, write_midi
    from audio_maestro.reference import read

    anchors = validate_map(request["beat_anchors"])
    start = beat_to_seconds(request["from_beat"], anchors) - request["source_offset_seconds"]
    stop = beat_to_seconds(request["to_beat"], anchors) - request["source_offset_seconds"]
    samples, rate, source = read(request["audio_path"], max(0.0, start - 0.3), stop + 0.3)
    origin = request["source_offset_seconds"] + source["start_seconds"]
    notes = track(
        samples.mean(axis=1),
        rate,
        origin,
        anchors,
        request["from_beat"],
        request["to_beat"],
        request["low_note"],
        request["high_note"],
        request["voice"],
        request.get("scale"),
        request["steps_per_beat"],
    )
    output = Path(request["output_dir"])
    midi = output / "voice.mid"
    rows = [
        {"pitch": n["pitch"], "start_seconds": n["start_seconds"], "end_seconds": n["end_seconds"]}
        for n in notes
    ]
    placed = beat_notes(rows, anchors, velocity=request["velocity"])
    first = anchors[0]["beat"]
    write_midi(
        midi,
        [{**n, "start_time": n["start_time"] - first} for n in placed],
        [{**a, "beat": a["beat"] - first} for a in anchors],
    )
    return {
        "schema": "maestro.voice",
        "schema_version": 1,
        "source": source,
        "register": {
            "low": note_name(request["low_note"]),
            "high": note_name(request["high_note"]),
        },
        "voice": request["voice"],
        "scale": request.get("scale"),
        "steps_per_beat": request["steps_per_beat"],
        "notes": notes,
        "midi_path": str(midi),
        "audio_paths": [],
        "basis": "Viterbi over harmonic salience per grid step; one pitch or rest per "
        "step; confidence is salience relative to the step's strongest pitch",
        "review_required": True,
    }
