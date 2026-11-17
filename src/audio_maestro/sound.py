"""Instrument parameters measured from audio on a beat grid, in synthesizer units."""

from __future__ import annotations

import math
from typing import Any

from audio_maestro.timeline import validate_map

DIVISIONS = {
    "1/1": 4.0,
    "1/2": 2.0,
    "1/2T": 4 / 3,
    "1/4": 1.0,
    "1/4T": 2 / 3,
    "1/8": 0.5,
    "1/8T": 1 / 3,
    "1/16": 0.25,
    "1/16T": 1 / 6,
    "1/32": 0.125,
}
ACTIVE_DB = 40.0
TAIL_BANDS = {"low": (25.0, 300.0), "mid": (300.0, 2000.0), "high": (2000.0, 8000.0)}


def _division(period_beats: float) -> tuple[str | None, float | None]:
    name, value = min(DIVISIONS.items(), key=lambda item: abs(math.log(item[1] / period_beats)))
    error = abs(value / period_beats - 1)
    return (name, round(error, 3)) if error <= 0.08 else (None, None)


def _modulation(curve: Any, step: float) -> dict[str, Any]:
    import numpy as np
    from scipy.signal import find_peaks

    centred = curve - curve.mean()
    if float(np.std(centred)) < 1e-9 or len(centred) < 8:
        return {"period_beats": None, "division": None, "strength": 0.0}
    full = np.correlate(centred, centred, mode="full")[len(centred) - 1 :]
    full = full / full[0]
    limit = min(len(full) - 1, round(4 / step))
    peaks, _ = find_peaks(full[: limit + 1])
    if not len(peaks):
        return {"period_beats": None, "division": None, "strength": 0.0}
    best = int(peaks[np.argmax(full[peaks])])
    period = best * step
    name, error = _division(period)
    return {
        "period_beats": round(period, 4),
        "division": name,
        "division_error": error,
        "strength": round(float(full[best]), 3),
    }


def _segments(active: Any) -> list[tuple[int, int]]:
    runs, start = [], None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        if not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(active) - 1))
    return runs


def _decay_seconds(level: Any, seconds: Any, peak_index: int) -> float | None:
    import numpy as np

    tail = level[peak_index:]
    times = seconds[peak_index:]
    fit = (tail <= tail[0] - 3) & (tail >= tail[0] - 35)
    if fit.sum() < 4:
        return None
    slope = float(np.polyfit(times[fit], tail[fit], 1)[0])
    return round(-60 / slope, 3) if slope < -1e-6 else None


def describe(request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from scipy.signal import stft

    from audio_maestro.lines import beat_to_seconds
    from audio_maestro.reference import read

    anchors = validate_map(request["beat_anchors"])
    offset = float(request["source_offset_seconds"])
    first, last = float(request["from_beat"]), float(request["to_beat"])
    start = beat_to_seconds(first, anchors) - offset
    stop = beat_to_seconds(last, anchors) - offset
    samples, rate, source = read(request["audio_path"], max(0.0, start), stop)
    mono = samples.mean(axis=1)
    size = 2 ** round(math.log2(rate * 0.046))
    hop = max(1, size // 8)
    frequencies, times, spectrum = stft(mono, rate, nperseg=size, noverlap=size - hop)
    times = times + offset + source["start_seconds"]
    power = np.abs(spectrum) ** 2
    step = 1 / int(request["steps_per_beat"])
    grid = np.arange(first, last - 1e-9, step)
    centres = np.array([beat_to_seconds(float(beat) + step / 2, anchors) for beat in grid])
    frame = np.clip(np.searchsorted(times, centres), 0, len(times) - 1)
    spectra = power[:, frame]
    audible = (frequencies >= 25) & (frequencies < 8000)
    level = 10 * np.log10(spectra[audible].sum(axis=0) + 1e-12)
    upper = (frequencies >= 100) & (frequencies < 8000)
    cumulative = np.cumsum(spectra[upper], axis=0)
    rolloff = frequencies[upper][np.argmax(cumulative >= 0.85 * cumulative[-1:], axis=0)]
    resonant = (frequencies >= 150) & (frequencies < 3000)
    peak = frequencies[resonant][np.argmax(spectra[resonant], axis=0)]
    sub = 10 * np.log10(spectra[(frequencies >= 25) & (frequencies < 100)].sum(axis=0) + 1e-12)
    active = level > level.max() - ACTIVE_DB

    seconds = np.array([beat_to_seconds(float(beat), anchors) for beat in grid])
    fine = times[(times >= seconds[0]) & (times <= seconds[-1] + step)]
    fine_level = 10 * np.log10(power[audible][:, np.isin(times, fine)].sum(axis=0) + 1e-12)
    attacks, decays = [], []
    for low, high in _segments(fine_level > fine_level.max() - ACTIVE_DB):
        segment = fine_level[low : high + 1]
        top = int(np.argmax(segment))
        reach = int(np.argmax(segment >= segment[top] - 3))
        attacks.append(float(fine[low + reach] - fine[low]))
        decay = _decay_seconds(segment, fine[low : high + 1], top)
        if decay is not None:
            decays.append(decay)
    tails = {}
    end = int(np.argmax(fine_level)) if len(fine_level) else 0
    for name, (low, high) in TAIL_BANDS.items():
        band = (frequencies >= low) & (frequencies < min(high, rate / 2))
        curve = 10 * np.log10(power[band][:, np.isin(times, fine)].sum(axis=0) + 1e-12)
        tails[name] = _decay_seconds(curve, fine, end) if band.any() else None
    modulation = (
        _modulation(np.log2(rolloff[active]), step)
        if active.sum() >= 8
        else {"period_beats": None, "division": None, "strength": 0.0}
    )
    sounding_rolloff = rolloff[active] if active.any() else rolloff
    sounding_peak = peak[active] if active.any() else peak

    def median(values: list[float]) -> float | None:
        return round(float(np.median(values)), 4) if values else None

    curves = [
        {
            "beat": round(float(beat), 4),
            "level_db": round(float(lv), 2),
            "rolloff_hz": round(float(ro), 1),
            "peak_hz": round(float(pk), 1),
            "sub_db": round(float(sb), 2),
        }
        for beat, lv, ro, pk, sb in zip(grid, level, rolloff, peak, sub, strict=True)
    ]
    return {
        "schema": "maestro.sound",
        "schema_version": 1,
        "source": source,
        "from_beat": first,
        "to_beat": last,
        "steps_per_beat": request["steps_per_beat"],
        "estimates": {
            "envelope": {
                "attack_seconds": median(attacks),
                "decay_seconds_60db": median(decays),
                "segments": len(attacks),
            },
            "filter": {
                "rolloff_hz_median": round(float(np.median(sounding_rolloff)), 1),
                "rolloff_hz_p10_p90": [
                    round(float(v), 1) for v in np.percentile(sounding_rolloff, [10, 90])
                ],
                "modulation": modulation,
            },
            "resonance": {
                "peak_hz_median": round(float(np.median(sounding_peak)), 1),
                "peak_hz_p10_p90": [
                    round(float(v), 1) for v in np.percentile(sounding_peak, [10, 90])
                ],
            },
            "tail_decay_seconds_60db": tails,
        },
        "curves": curves,
        "audio_paths": [],
        "basis": "rolloff = frequency below which 85 % of 100 Hz-8 kHz power lies; "
        "modulation from the autocorrelation of log rolloff over sounding steps; "
        "decay from a line fitted 3 to 35 dB below the peak",
        "review_required": True,
    }
