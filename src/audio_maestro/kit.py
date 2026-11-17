"""Mixed drum-kit hits classified from band rises on a reviewed beat grid."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from audio_maestro.timeline import validate_map, write_midi

BANDS = {
    "low": (35.0, 120.0),
    "body": (150.0, 400.0),
    "crack": (1500.0, 5000.0),
    "air": (7000.0, 16000.0),
}
POST_MS = (-5.0, 35.0)
PRE_MS = (-60.0, -15.0)
MERGE_MS = 50.0
DEFAULT_RULES = {
    "kick_rise_db": 15.0,
    "snare_rise_db": 12.0,
    "crack_rise_db": 8.0,
    "hat_rise_db": 9.0,
    "min_level_db": -30.0,
}


def band_levels(samples: Any, rate: int) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    from scipy.signal import stft

    size = 2 ** round(math.log2(rate * 0.021))
    frequencies, times, spectrum = stft(samples, rate, nperseg=size, noverlap=size - size // 8)
    power = np.abs(spectrum) ** 2
    levels = {}
    for name, (low, high) in BANDS.items():
        selected = (frequencies >= low) & (frequencies < min(high, rate / 2))
        energy = power[selected].sum(axis=0) if selected.any() else np.zeros(len(times))
        levels[name] = 10 * np.log10(energy + 1e-12)
    return times, levels


def rise(times: Any, level: Any, at: float) -> tuple[float, float]:
    after = (times >= at + POST_MS[0] / 1000) & (times < at + POST_MS[1] / 1000)
    before = (times >= at + PRE_MS[0] / 1000) & (times < at + PRE_MS[1] / 1000)
    if not after.any() or not before.any():
        return 0.0, -120.0
    peak = float(level[after].max())
    return peak - float(level[before].mean()), peak


def classify(
    rises: dict[str, float], levels: dict[str, float], rules: dict[str, float]
) -> list[tuple[str, float]]:
    hits = []
    floor = rules["min_level_db"]
    if (
        rises["low"] >= rules["kick_rise_db"]
        and levels["low"] >= floor
        and levels["low_abs"] >= levels["body_abs"] - 6
    ):
        hits.append(("kick", levels["low"]))
    masked = bool(hits) and levels["crack_abs"] < levels["body_abs"] - 12
    snare = (
        rises["body"] >= rules["snare_rise_db"]
        and rises["crack"] >= rules["crack_rise_db"]
        and not masked
    )
    strong = rises["body"] >= 15 and rises["crack"] >= 15
    if (snare or strong) and max(levels["body"], levels["crack"]) >= floor:
        hits.append(("snare", max(levels["body"], levels["crack"])))
    if (
        rises["air"] >= rules["hat_rise_db"]
        and rises["body"] < rules["snare_rise_db"]
        and rises["low"] < rules["snare_rise_db"]
        and levels["air"] >= floor
    ):
        hits.append(("hat", levels["air"]))
    return hits


def _strongest(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for event in events:
        rival = next(
            (
                item
                for item in reversed(kept)
                if item["instrument"] == event["instrument"]
                and event["source_seconds"] - item["source_seconds"] < MERGE_MS / 1000
            ),
            None,
        )
        if rival is None:
            kept.append(event)
        elif event["relative_level_db"] > rival["relative_level_db"]:
            kept[kept.index(rival)] = event
    return sorted(kept, key=lambda item: (item["beat"], item["midi_pitch"]))


def pattern(
    events: list[dict[str, Any]], beats_per_bar: float, subdivisions: list[int]
) -> list[str]:
    lines = []
    bars = sorted({int(event["beat"] // beats_per_bar) for event in events})
    for bar in bars:
        for division in subdivisions:
            cells = round(beats_per_bar * division)
            for instrument, letter in (("kick", "K"), ("snare", "S"), ("hat", "H")):
                row = ["-"] * cells
                for event in events:
                    if event["instrument"] != instrument or event["division"] != division:
                        continue
                    index = round((event["beat"] - bar * beats_per_bar) * division)
                    if 0 <= index < cells and int(event["beat"] // beats_per_bar) == bar:
                        row[index] = "X" if event["velocity"] >= 100 else "x"
                if "x" in row or "X" in row:
                    lines.append(f"bar {bar + 1} 1/{division} {letter} {''.join(row)}")
    return lines


def analyze_kit(request: dict[str, Any]) -> dict[str, Any]:
    import json

    import numpy as np
    import soundfile as sf

    from audio_maestro.decode import readable

    output = Path(request["output_dir"])
    anchors = validate_map(request["beat_anchors"])
    path = Path(request["audio_path"])
    with readable(path) as (decoded, _):
        samples, rate = sf.read(decoded, always_2d=True)
    if not samples.size or samples.shape[1] > 2 or not np.isfinite(samples).all():
        raise ValueError("drum stem must contain finite, nonempty mono/stereo samples")
    offset = float(request["source_offset_seconds"])
    times, levels = band_levels(samples.mean(axis=1), rate)
    times = times + offset
    tops = {name: float(level.max()) for name, level in levels.items()}
    known = np.array([item["beat"] for item in anchors])
    seconds = np.array([item["seconds"] for item in anchors])
    first = max(float(request.get("from_beat") or known[0]), float(known[0]))
    last = min(float(request.get("to_beat") or known[-1]), float(known[-1]))
    rules = {
        **DEFAULT_RULES,
        **{key: value for key, value in request.get("rules", {}).items() if value is not None},
    }
    pitches = request["pitches"]
    found: dict[tuple[float, str], dict[str, Any]] = {}
    for division in request["subdivisions"]:
        grid = np.arange(math.ceil(first * division), math.floor(last * division) + 1) / division
        for beat in grid:
            at = float(np.interp(beat, known, seconds))
            measured = {name: rise(times, levels[name], at) for name in BANDS}
            rises = {name: value[0] for name, value in measured.items()}
            relative = {name: value[1] - tops[name] for name, value in measured.items()}
            relative.update(
                low_abs=measured["low"][1],
                body_abs=measured["body"][1],
                crack_abs=measured["crack"][1],
            )
            for instrument, level in classify(rises, relative, rules):
                key = (round(float(beat), 6), instrument)
                if key in found:
                    continue
                found[key] = {
                    "instrument": instrument,
                    "beat": round(float(beat), 6),
                    "division": int(division),
                    "source_seconds": round(at, 6),
                    "midi_pitch": int(pitches[instrument]),
                    "velocity": max(1, min(127, round(127 * 10 ** (level / 40)))),
                    "relative_level_db": round(level, 2),
                    "rises_db": {name: round(value, 2) for name, value in rises.items()},
                }
    events = _strongest(sorted(found.values(), key=lambda item: item["source_seconds"]))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    data = {
        "schema": "maestro.drum-events",
        "schema_version": 1,
        "events": events,
        "sources": [
            {
                "path": str(path),
                "sample_rate": rate,
                "frames": len(samples),
                "source_offset_seconds": offset,
                "instrument": "kit",
                "sha256": digest,
            }
        ],
        "tempo_map": {
            "anchors": anchors,
            "review_required": False,
            "grid_origin": "caller-supplied anchors",
            "meter_inferred": False,
        },
        "method": "band-rise rules on a reviewed beat grid; an estimate to review",
    }
    events_path = output / "drum-events.json"
    events_path.write_text(json.dumps(data, allow_nan=False, indent=2), encoding="utf-8")
    midi_path = output / "kit.mid"
    write_midi(midi_path, data)
    counts = {
        name: sum(1 for event in events if event["instrument"] == name)
        for name in ("kick", "snare", "hat")
    }
    return {
        "schema": "maestro.kit",
        "schema_version": 1,
        "source": data["sources"][0],
        "from_beat": first,
        "to_beat": last,
        "subdivisions": request["subdivisions"],
        "beats_per_bar": request["beats_per_bar"],
        "rules": rules,
        "bands_hz": {name: list(edges) for name, edges in BANDS.items()},
        "counts": counts,
        "events": events,
        "pattern": pattern(events, float(request["beats_per_bar"]), request["subdivisions"]),
        "drum_events_path": str(events_path),
        "midi_path": str(midi_path),
        "data_paths": [str(events_path)],
        "audio_paths": [],
        "basis": "rise = loudest level 5 ms before to 35 ms after a grid point, less the "
        "mean 60 to 15 ms before it; levels relative to each band's loudest moment",
        "review_required": True,
    }
