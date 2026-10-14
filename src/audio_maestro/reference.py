"""Measured source excerpts and specific reconstruction differences; no quality score."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: str, start: float, end: float | None) -> tuple[Any, int, dict[str, Any]]:
    import numpy as np
    import soundfile as sf

    source = Path(path)
    with sf.SoundFile(source) as audio:
        rate = audio.samplerate
        first = math.floor(start * rate)
        last = min(audio.frames, math.floor(end * rate)) if end is not None else audio.frames
        if first < 0 or last <= first or audio.channels not in (1, 2):
            raise ValueError("excerpt must intersect finite mono/stereo audio")
        audio.seek(first)
        samples = audio.read(last - first, always_2d=True, dtype="float64")
        info = {
            "path": str(source),
            "sha256": fingerprint(source),
            "sample_rate": rate,
            "source_frames": audio.frames,
            "channels": audio.channels,
            "first_frame": first,
            "frames": len(samples),
            "start_seconds": first / rate,
            "end_seconds": (first + len(samples)) / rate,
        }
    if not np.isfinite(samples).all():
        raise ValueError("audio contains nonfinite samples")
    return samples, rate, info


def _db(value: float) -> float:
    return round(20 * math.log10(max(1e-12, value)), 3)


def features(samples: Any, rate: int) -> dict[str, Any]:
    import numpy as np
    from scipy.signal import find_peaks, stft

    from audio_maestro.features import BANDS

    block = max(1, round(rate * 0.01))
    usable = len(samples) // block * block
    if usable < block:
        raise ValueError("excerpt must be at least 10 ms long")
    envelope = np.sqrt(
        np.mean(samples[:usable].reshape(-1, block, samples.shape[1]) ** 2, axis=(1, 2))
    )
    rise = np.maximum(0, np.diff(envelope, prepend=0))
    threshold = max(
        float(np.median(rise) + 4 * np.median(np.abs(rise - np.median(rise)))),
        float(rise.max()) * 0.06,
        1e-6,
    )
    peaks, _ = find_peaks(rise, height=threshold, distance=5)
    frame = min(4096, len(samples))
    boundary: Any = None  # SciPy's runtime accepts None; its annotation lists only strings.
    frequencies, _, spectrum = stft(
        samples.T, fs=rate, nperseg=frame, noverlap=frame * 3 // 4, boundary=boundary
    )
    power = np.mean(np.abs(spectrum) ** 2, axis=(0, 2))
    total = float(power.sum())
    bands: dict[str, float | None] = {}
    for name, low, high in BANDS:
        selected = (frequencies >= low) & (frequencies < (high or rate / 2 + 1))
        energy = float(power[selected].sum())
        bands[name] = round(energy / total, 8) if total > 1e-24 else None
    chroma = np.zeros(12)
    voiced = (frequencies >= 40) & (frequencies <= min(6000, rate / 2))
    note_bins = np.rint(69 + 12 * np.log2(frequencies[voiced] / 440)).astype(int) % 12
    np.add.at(chroma, note_bins, power[voiced])
    chroma_total = float(chroma.sum())
    chroma = chroma / chroma_total if chroma_total else chroma
    rms = float(np.sqrt(np.mean(samples**2)))
    peak = float(np.max(np.abs(samples)))
    correlation = None
    if samples.shape[1] == 2 and np.min(np.std(samples, axis=0)) > 1e-8:
        correlation = float(np.corrcoef(samples.T)[0, 1])
    return {
        "rms_dbfs": _db(rms),
        "sample_peak_dbfs": _db(peak),
        "crest_factor_db": round(_db(peak) - _db(rms), 3) if rms > 1e-8 else None,
        "silent": rms < 1e-8,
        "stereo_correlation": correlation,
        "band_power_shares": bands,
        "chroma": chroma.tolist(),
        "chroma_basis": "FFT harmonic energy; not identified notes or chords",
        "onsets_seconds": (peaks * block / rate).tolist(),
        "onset_resolution_seconds": block / rate,
        "envelope_hop_seconds": block / rate,
        "envelope": envelope.tolist(),
        "centroid_hz": float(np.dot(frequencies, power) / total) if total > 1e-24 else None,
    }


def analyze(request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    from audio_maestro.drum_worker import tempo_candidates
    from audio_maestro.timeline import validate_map

    start, end = request["start_seconds"], request["end_seconds"]
    output = Path(request["output_dir"])
    inputs: list[dict[str, Any]] = [
        {"path": request["reference_path"], "role": "reference_mix", "source_offset_seconds": 0},
        *request.get("stems", []),
    ]
    sources, audio_paths = [], []
    for index, item in enumerate(inputs):
        offset = item["source_offset_seconds"]
        if end <= offset:
            raise ValueError(f"requested section ends before stem {item['path']} starts")
        local_start, local_end = max(0, start - offset), end - offset
        samples, rate, info = read(item["path"], local_start, local_end)
        profile = features(samples, rate)
        profile["onsets_seconds"] = [
            offset + info["start_seconds"] + value for value in profile["onsets_seconds"]
        ]
        info.update(
            role=item["role"],
            role_basis="caller-supplied" if index else "reference",
            source_offset_seconds=offset,
            features=profile,
        )
        clip = output / f"source-{index:02d}.wav"
        sf.write(clip, samples, rate, subtype="FLOAT")
        info["excerpt_path"] = str(clip)
        audio_paths.append(str(clip))
        # One-second energy changes propose boundaries, with no invented verse/chorus labels.
        envelope = np.asarray(profile["envelope"])
        width = max(1, round(1 / profile["envelope_hop_seconds"]))
        summaries = [
            float(np.mean(envelope[i : i + width])) for i in range(0, len(envelope), width)
        ]
        info["section_change_candidates"] = [
            offset + info["start_seconds"] + i
            for i in range(1, len(summaries))
            if abs(_db(summaries[i]) - _db(summaries[i - 1])) >= 8
        ]
        sources.append(info)
        print(
            __import__("json").dumps(
                {"event": "progress", "progress": (index + 1) / (len(inputs) + 1)}
            ),
            flush=True,
        )
    rhythmic = next(
        (source for source in sources if source["role"] in {"drums", "kick"}), sources[0]
    )
    events = [
        {"source_seconds": seconds, "peak": 1.0}
        for seconds in rhythmic["features"]["onsets_seconds"]
    ]
    candidates = tempo_candidates(events, start, end)
    timing: dict[str, Any] = {
        "review_required": True,
        "tempo_candidates": candidates,
        "basis": "energy-rise periodicity; meter and downbeats unknown",
    }
    if request.get("beat_anchors"):
        timing = {
            "review_required": False,
            "anchors": validate_map(request["beat_anchors"]),
            "basis": "caller-supplied reviewed source-time/quarter-beat map",
        }
    return {
        "schema": "maestro.reference",
        "schema_version": 1,
        "sources": sources,
        "review_required": bool(timing["review_required"]),
        "source_time_range": [start, end],
        "timing": timing,
        "audio_paths": audio_paths,
        "mix_sum_residual": {
            "available": False,
            "reason": "supplied stems are not asserted to be a complete complementary set",
        },
        "next_actions": [
            "Review timing before writing MIDI.",
            "Use transcribe_audio for model estimates and review notes before export.",
            "Use a DAW and audio comparison tools to audition reconstruction candidates.",
        ],
    }
