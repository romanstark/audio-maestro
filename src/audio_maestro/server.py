"""Standalone audio analysis MCP with optional, explicitly provisioned models."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from audio_maestro import jobs, runtime
from audio_maestro.pipeline import beat_notes
from audio_maestro.timeline import validate_map
from audio_maestro.toolargs import BeatAnchorArg, DrumStemArg, ReferenceStemArg

READS = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
)
ADDS = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
)
DESTROYS = ToolAnnotations(
    readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False
)
mcp = FastMCP(
    "audio-maestro",
    instructions=(
        "Analyze local audio independently of a DAW or separator. "
        "Basic Pitch estimates pitched "
        "notes on CPU; optional SheetSage2 provides lead-sheet estimates. "
        "Review notes, chords "
        "or beat hypotheses as verified music. Use reviewed source-second/quarter-beat anchors for "
        "placement. Known kick/snare sources use analyze_drums, not pitched transcription. "
        "Poll get_audio_job and read get_audio_report in bounded pages. Model setup is explicit, "
        "inference downloads nothing and audio is never uploaded. Results include paths and "
        "source fingerprints. Use any DAW to audition exported MIDI/audio."
    ),
)


def _range(start: float, end: float) -> None:
    if (
        not math.isfinite(start)
        or not math.isfinite(end)
        or start < 0
        or not 0 < end - start <= 120
    ):
        raise ValueError("Select a finite source excerpt of 0..120 seconds.")


@mcp.tool(annotations=READS)
def audio_status() -> dict[str, Any]:
    """Read model configuration without creating directories or running inference."""
    try:
        return runtime.status()
    except (OSError, ValueError, KeyError) as error:
        return {"ok": False, "error": str(error)}


@mcp.tool(annotations=ADDS)
def setup_backend(
    backend: Literal["basic_pitch", "sheetsage"] = "basic_pitch",
    executable: str = "",
    model_dir: str = "",
    device: Literal["cpu", "cuda", "metal"] = "cpu",
    confirm: bool = False,
    library_dir: str = "",
) -> dict[str, Any]:
    """Preview/install Basic Pitch or register existing audio.cpp SheetSage files.

    Basic Pitch setup needs uv and downloads its pinned dependencies. SheetSage setup
    only registers local files and hashes; it does not fetch a runtime or weights.
    """
    try:
        return (
            runtime.setup_basic_pitch(confirm)
            if backend == "basic_pitch"
            else runtime.setup_sheetsage(executable, model_dir, device, confirm, library_dir)
        )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        return {"ok": False, "error": str(error)}


@mcp.tool(annotations=ADDS)
def analyze_reference_audio(
    reference_path: str,
    output_dir: str,
    stems: list[ReferenceStemArg] | None = None,
    start_seconds: float = 0,
    end_seconds: float = 30,
    beat_anchors: list[BeatAnchorArg] | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """Measure levels, bands, chroma, envelopes and onset/tempo hypotheses.

    Roles are caller labels, not detected instrument identities. Periodicity does not
    establish meter or downbeats. Model transcription is a separate transcribe_audio job.
    """
    try:
        _range(start_seconds, end_seconds)
        anchors = validate_map([a.model_dump() for a in beat_anchors]) if beat_anchors else None
        request = {
            "kind": "reference",
            "reference_path": reference_path,
            "stems": [s.model_dump() for s in stems or []],
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "beat_anchors": anchors,
            "transcribe": False,
        }
        if not confirm:
            return {"ok": True, "executed": False, "plan": request}
        return jobs.manager().submit(request, output_dir)
    except (OSError, ValueError) as error:
        return {"ok": False, "error": str(error)}


@mcp.tool(annotations=ADDS)
def transcribe_audio(
    audio_path: str,
    output_dir: str,
    backend: Literal["basic_pitch", "sheetsage"] = "basic_pitch",
    start_seconds: float = 0,
    end_seconds: float = 30,
    source_offset_seconds: Annotated[float, Field(ge=0)] = 0,
    beat_anchors: list[BeatAnchorArg] | None = None,
    monophonic: bool = False,
    merge: bool = False,
    cleanup: bool = True,
    velocity: Annotated[int, Field(ge=1, le=127)] = 100,
    align_to_audio: Annotated[
        bool,
        Field(description="Move note onsets to the tracked line's pitch changes; needs anchors."),
    ] = False,
    align_voice: Literal["top", "bottom"] = "top",
    align_low_note: Annotated[int | None, Field(ge=0, le=127)] = None,
    align_high_note: Annotated[int | None, Field(ge=0, le=127)] = None,
    align_scale: Annotated[
        list[int] | None,
        Field(description="Pitch classes of the key (0 = C), or None for no scale."),
    ] = None,
    align_section_beats: Annotated[float, Field(gt=0, le=256)] = 16,
    align_snap_beats: Annotated[float | None, Field(gt=0, le=4)] = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """Queue pitched-note estimates and export MIDI plus source-timed JSON.

    Basic Pitch is the CPU baseline; SheetSage2 is optional, strongest on lead-sheet
    reduction rather than all voices. Harmonic cleanup and lowest-voice monophonic
    reduction can remove real notes; raw output is retained. Velocity is caller-supplied.
    Without reviewed anchors the MIDI uses a 120-BPM carrier preserving seconds,
    which is not a detected tempo or placement map for an existing DAW project.
    align_to_audio tracks one voice in the notes' register (or the given one) and
    shifts each section by the median offset to that line's pitch changes; model
    timing drifts, so one constant shift is not enough. The originals are kept.
    """
    try:
        _range(start_seconds, end_seconds)
        if align_to_audio and not beat_anchors:
            raise ValueError("align_to_audio needs reviewed beat_anchors.")
        if align_scale is not None and not all(0 <= item <= 11 for item in align_scale):
            raise ValueError("Scale entries are pitch classes 0..11.")
        if not math.isfinite(source_offset_seconds) or source_offset_seconds < 0:
            raise ValueError("Source offset must be finite and nonnegative.")
        if not 1 <= velocity <= 127:
            raise ValueError("Velocity must be 1..127.")
        anchors = validate_map([a.model_dump() for a in beat_anchors]) if beat_anchors else None
        if anchors and (
            anchors[0]["seconds"] > source_offset_seconds + start_seconds
            or anchors[-1]["seconds"] < source_offset_seconds + end_seconds
        ):
            raise ValueError(
                "Reviewed anchors must cover the requested excerpt on the original source clock."
            )
        request = {
            "kind": "transcription",
            "audio_path": audio_path,
            "backend": backend,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "source_offset_seconds": source_offset_seconds,
            "beat_anchors": anchors,
            "monophonic": monophonic,
            "merge": merge,
            "cleanup": cleanup,
            "velocity": velocity,
            "align_to_audio": align_to_audio,
            "align_voice": align_voice,
            "align_low_note": align_low_note,
            "align_high_note": align_high_note,
            "align_scale": align_scale,
            "align_section_beats": align_section_beats,
            "align_snap_beats": align_snap_beats,
        }
        if not confirm:
            return {"ok": True, "executed": False, "plan": request}
        return jobs.manager().submit(request, output_dir)
    except (OSError, ValueError) as error:
        return {"ok": False, "error": str(error)}


@mcp.tool(annotations=READS)
def get_audio_job(job_id: str) -> dict[str, Any]:
    """Read durable state, progress, output paths and failure evidence."""
    try:
        return {"ok": True, **jobs.manager().get(job_id)}
    except (OSError, ValueError) as error:
        return {"ok": False, "error": str(error)}


@mcp.tool(annotations=DESTROYS)
def cancel_audio_job(job_id: str) -> dict[str, Any]:
    """Cancel this server's owned worker tree; retain partial files and evidence."""
    try:
        return jobs.manager().cancel(job_id)
    except (OSError, ValueError) as error:
        return {"ok": False, "error": str(error)}


@mcp.tool(annotations=READS)
def get_audio_report(
    job_id: str,
    section: Literal[
        "overview",
        "sources",
        "notes",
        "raw_notes",
        "beat_notes",
        "onsets",
        "envelope",
        "model_events",
        "unaligned_notes",
        "pattern",
        "curves",
    ] = "overview",
    source_index: Annotated[int, Field(ge=0)] = 0,
    offset: Annotated[int, Field(ge=0)] = 0,
    limit: Annotated[int, Field(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    """Read bounded evidence; overview omits all large note, event and envelope arrays."""
    try:
        if offset < 0 or not 1 <= limit <= 200:
            raise ValueError("Offset must be nonnegative and limit 1..200.")
        job = jobs.manager().get(job_id)
        if job["state"] != "succeeded":
            return {
                "ok": False,
                "state": job["state"],
                "error": "Job has no verified complete report.",
            }
        result = job["result"]
        path = result.get("report_path") or result["analysis_path"]
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        if section == "overview":
            payload = {
                k: v
                for k, v in report.items()
                if k
                not in {
                    "sources",
                    "notes",
                    "raw_notes",
                    "beat_notes",
                    "events",
                    "one_shots",
                    "unaligned_notes",
                    "pattern",
                    "curves",
                }
            }
            payload["sources_count"] = len(report.get("sources", []))
            payload["notes_count"] = len(report.get("notes", report.get("events", [])))
            if "provenance" in payload:
                payload["provenance"] = {
                    k: v for k, v in payload["provenance"].items() if k != "native_events"
                }
            return {"ok": True, "report": payload}
        if section == "model_events":
            items = report.get("provenance", {}).get("native_events", [])
        elif section == "sources":
            items = [
                {k: v for k, v in source.items() if k not in {"features", "notes"}}
                for source in report.get("sources", [])
            ]
        elif section in {"unaligned_notes", "pattern", "curves"}:
            items = report.get(section) or []
        elif section in {"notes", "raw_notes", "beat_notes"}:
            items = (
                report.get(section, report.get("events", []) if section == "notes" else []) or []
            )
        else:
            source = report["sources"][source_index]
            items = source["features"]["envelope" if section == "envelope" else "onsets_seconds"]
        return {
            "ok": True,
            "section": section,
            "rows": items[offset : offset + limit],
            "total": len(items),
            "next_offset": offset + limit if offset + limit < len(items) else None,
        }
    except (OSError, ValueError, KeyError, IndexError) as error:
        return {"ok": False, "error": str(error)}


@mcp.tool(annotations=READS)
def notes_to_beats(
    report_path: str,
    beat_anchors: list[BeatAnchorArg],
    clip_origin_beat: float = 0,
    velocity: Annotated[int, Field(ge=1, le=127)] = 100,
    offset: Annotated[int, Field(ge=0)] = 0,
    limit: Annotated[int, Field(ge=1, le=200)] = 100,
) -> dict[str, Any]:
    """Convert reviewed source seconds to portable clip-local notes, without a DAW write."""
    try:
        if offset < 0 or not 1 <= limit <= 200:
            raise ValueError("Invalid page bounds.")
        report = json.loads(Path(report_path).expanduser().read_text(encoding="utf-8"))
        if report.get("schema") != "maestro.transcription" or report.get("schema_version") != 1:
            raise ValueError("Expected a version-1 transcription report.")
        notes = beat_notes(
            report["notes"], [a.model_dump() for a in beat_anchors], clip_origin_beat, velocity
        )
        return {
            "ok": True,
            "notes": notes[offset : offset + limit],
            "total": len(notes),
            "next_offset": offset + limit if offset + limit < len(notes) else None,
            "review_required": True,
            "written": False,
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"ok": False, "error": str(error)}


@mcp.resource("audio://workflow")
def workflow() -> str:
    return (Path(__file__).parent / "skills/audio-analyze/SKILL.md").read_text(encoding="utf-8")


def main() -> None:
    try:
        mcp.run(transport="stdio")
    finally:
        jobs.close()


@mcp.tool(annotations=ADDS)
def analyze_drums(
    stems: Annotated[
        list[DrumStemArg], Field(description="Identified kick/snare stems and source offsets.")
    ],
    output_dir: Annotated[
        str, Field(description="Local output root; a unique job directory prevents overwrite.")
    ],
    tempo_bpm: Annotated[
        float | None,
        Field(gt=0, le=999, description="Known constant BPM; None proposes tempo from hits."),
    ] = None,
    beat_anchors: Annotated[
        list[BeatAnchorArg] | None,
        Field(description="Reviewed source-second/quarter-beat map, at least two anchors."),
    ] = None,
    sensitivity: Annotated[
        float, Field(ge=1, le=50, description="Energy-rise noise threshold multiplier.")
    ] = 6,
    cut_one_shots: Annotated[
        bool, Field(description="Also cut individual PCM_24 WAV hits.")
    ] = False,
    pre_roll_ms: Annotated[
        float, Field(ge=0, le=1000, description="Audio before each attack, in ms.")
    ] = 10,
    tail_ms: Annotated[
        float, Field(gt=0, le=5000, description="Audio after each attack, in ms.")
    ] = 250,
) -> dict[str, Any]:
    """Queue isolated drum analysis, MIDI/tempo export and optional one-shots.

    The caller identifies the instruments. This does not classify a mixed drum kit.
    Detection refines a leading threshold crossing and subtracts supplied calibration.
    Inferred tempo remains a proposal with half/double-time ambiguity. Use beat anchors
    for known tempo changes. Optional dependencies run in a separate interpreter;
    Audio Maestro includes its numerical dependencies; no DAW is required.
    """
    try:
        anchors = [item.model_dump() for item in beat_anchors] if beat_anchors else None
        if anchors is not None:
            anchors = validate_map(anchors)
            if tempo_bpm is not None:
                raise ValueError("supply tempo_bpm or beat_anchors, not both")
        if tempo_bpm is not None and not math.isfinite(tempo_bpm):
            raise ValueError("tempo must be finite")
        return jobs.manager().submit(
            {
                "stems": [item.model_dump() for item in stems],
                "tempo_bpm": tempo_bpm,
                "beat_anchors": anchors,
                "sensitivity": sensitivity,
                "cut_one_shots": cut_one_shots,
                "pre_roll_ms": pre_roll_ms,
                "tail_ms": tail_ms,
            },
            output_dir,
        )
    except (ValueError, OSError) as error:
        return {"ok": False, "error": str(error)}


@mcp.tool(annotations=ADDS)
def analyze_drum_kit(
    audio_path: Annotated[str, Field(description="Local drum stem of a whole kit, any source.")],
    output_dir: Annotated[
        str, Field(description="Local output root; a unique job directory prevents overwrite.")
    ],
    beat_anchors: Annotated[
        list[BeatAnchorArg],
        Field(min_length=2, description="Reviewed source-second/quarter-beat map."),
    ],
    source_offset_seconds: Annotated[
        float, Field(ge=0, description="Position of the file's first sample in the source.")
    ] = 0,
    from_beat: Annotated[
        float | None, Field(ge=0, description="First beat; None = map start.")
    ] = None,
    to_beat: Annotated[float | None, Field(ge=0, description="Last beat; None = map end.")] = None,
    subdivisions: Annotated[
        list[Literal[2, 3, 4, 6, 8]],
        Field(min_length=1, description="Grid points per beat: 4 sixteenths, 3 triplets."),
    ] = [4, 3],  # noqa: B006 -- pydantic copies defaults per call
    beats_per_bar: Annotated[float, Field(gt=0, le=32, description="For pattern lines.")] = 4,
    kick_pitch: Annotated[int, Field(ge=0, le=127)] = 36,
    snare_pitch: Annotated[int, Field(ge=0, le=127)] = 38,
    hat_pitch: Annotated[int, Field(ge=0, le=127)] = 42,
    kick_rise_db: Annotated[float | None, Field(ge=1, le=60)] = None,
    snare_rise_db: Annotated[float | None, Field(ge=1, le=60)] = None,
    hat_rise_db: Annotated[float | None, Field(ge=1, le=60)] = None,
    min_level_db: Annotated[
        float | None, Field(ge=-80, le=0, description="Quietest hit, relative to its band peak.")
    ] = None,
) -> dict[str, Any]:
    """Queue kick, snare and hat classification of a mixed kit on a reviewed beat grid.

    Use this when no separator splits the kit. Band rises at each grid point
    (35-120 Hz, 150-400 Hz, 1.5-5 kHz, 7-16 kHz) decide the instrument, so tempo
    anchors are required. Output: events in the maestro.drum-events schema that a
    DAW writer accepts, a channel-10 MIDI file, and one pattern line per bar. The
    pattern is the result; hits cut from a mixed kit do not make a clean sound.
    """
    try:
        anchors = validate_map([item.model_dump() for item in beat_anchors])
        if from_beat is not None and to_beat is not None and to_beat <= from_beat:
            raise ValueError("to_beat must be greater than from_beat")
        return jobs.manager().submit(
            {
                "kind": "kit",
                "audio_path": audio_path,
                "beat_anchors": anchors,
                "source_offset_seconds": source_offset_seconds,
                "from_beat": from_beat,
                "to_beat": to_beat,
                "subdivisions": sorted(set(subdivisions), reverse=True),
                "beats_per_bar": beats_per_bar,
                "pitches": {"kick": kick_pitch, "snare": snare_pitch, "hat": hat_pitch},
                "rules": {
                    "kick_rise_db": kick_rise_db,
                    "snare_rise_db": snare_rise_db,
                    "hat_rise_db": hat_rise_db,
                    "min_level_db": min_level_db,
                },
            },
            output_dir,
        )
    except (ValueError, OSError) as error:
        return {"ok": False, "error": str(error)}


@mcp.tool(annotations=ADDS)
def track_voice(
    audio_path: Annotated[str, Field(description="Local audio with the voice, any source.")],
    output_dir: Annotated[
        str, Field(description="Local output root; a unique job directory prevents overwrite.")
    ],
    beat_anchors: Annotated[
        list[BeatAnchorArg],
        Field(min_length=2, description="Reviewed source-second/quarter-beat map."),
    ],
    from_beat: Annotated[float, Field(ge=0)],
    to_beat: Annotated[float, Field(gt=0)],
    low_note: Annotated[int, Field(ge=12, le=120, description="Lowest MIDI pitch of the voice.")],
    high_note: Annotated[int, Field(ge=12, le=120, description="Highest MIDI pitch.")],
    voice: Annotated[
        Literal["top", "bottom"],
        Field(description="Prefer the highest (melody) or lowest (bass) pitch present."),
    ] = "top",
    scale: Annotated[
        list[int] | None,
        Field(description="Pitch classes of the key (0 = C); others cost. None = chromatic."),
    ] = None,
    steps_per_beat: Annotated[Literal[2, 3, 4, 6, 8], Field(description="Grid steps.")] = 4,
    source_offset_seconds: Annotated[float, Field(ge=0)] = 0,
    velocity: Annotated[int, Field(ge=1, le=127)] = 100,
) -> dict[str, Any]:
    """Queue one voice followed through a register on the beat grid, as notes and MIDI.

    A Viterbi path over harmonic salience picks one pitch or a rest per grid step,
    preferring held pitches and stepwise motion. Use it for a lead-sheet line, a bass
    line, or to check where a transcription's notes really change. Confidence is
    relative salience, not a probability. Notes are estimates to review.
    """
    try:
        anchors = validate_map([item.model_dump() for item in beat_anchors])
        if to_beat <= from_beat or high_note <= low_note:
            raise ValueError("to_beat and high_note must exceed from_beat and low_note")
        if scale is not None and not all(0 <= item <= 11 for item in scale):
            raise ValueError("Scale entries are pitch classes 0..11.")
        if from_beat < anchors[0]["beat"] or to_beat > anchors[-1]["beat"]:
            raise ValueError("Anchors must cover from_beat..to_beat.")
        return jobs.manager().submit(
            {
                "kind": "voice",
                "audio_path": audio_path,
                "beat_anchors": anchors,
                "from_beat": from_beat,
                "to_beat": to_beat,
                "low_note": low_note,
                "high_note": high_note,
                "voice": voice,
                "scale": sorted(set(scale)) if scale is not None else None,
                "steps_per_beat": steps_per_beat,
                "source_offset_seconds": source_offset_seconds,
                "velocity": velocity,
            },
            output_dir,
        )
    except (ValueError, OSError) as error:
        return {"ok": False, "error": str(error)}


@mcp.tool(annotations=ADDS)
def describe_sound(
    audio_path: Annotated[str, Field(description="Local audio of one instrument, any source.")],
    output_dir: Annotated[
        str, Field(description="Local output root; a unique job directory prevents overwrite.")
    ],
    beat_anchors: Annotated[
        list[BeatAnchorArg],
        Field(min_length=2, description="Reviewed source-second/quarter-beat map."),
    ],
    from_beat: Annotated[float, Field(ge=0)],
    to_beat: Annotated[float, Field(gt=0)],
    steps_per_beat: Annotated[int, Field(ge=4, le=96, description="Curve resolution.")] = 24,
    source_offset_seconds: Annotated[float, Field(ge=0)] = 0,
) -> dict[str, Any]:
    """Queue measurement of an instrument's parameters in synthesizer units.

    Reports attack and 60 dB decay in seconds, spectral rolloff (the audible
    filter edge) and its movement named as a tempo division such as 1/8T, the
    resonance or formant range in Hz, and tail decay per band, which shows a dark or
    bright reverb. Use the values to set an instrument preset, render, compare with
    compare_spectrum in Live Maestro, and measure again. Estimates, not a patch.
    """
    try:
        anchors = validate_map([item.model_dump() for item in beat_anchors])
        if to_beat <= from_beat:
            raise ValueError("to_beat must exceed from_beat")
        if from_beat < anchors[0]["beat"] or to_beat > anchors[-1]["beat"]:
            raise ValueError("Anchors must cover from_beat..to_beat.")
        return jobs.manager().submit(
            {
                "kind": "sound",
                "audio_path": audio_path,
                "beat_anchors": anchors,
                "from_beat": from_beat,
                "to_beat": to_beat,
                "steps_per_beat": steps_per_beat,
                "source_offset_seconds": source_offset_seconds,
            },
            output_dir,
        )
    except (ValueError, OSError) as error:
        return {"ok": False, "error": str(error)}


if __name__ == "__main__":
    main()
