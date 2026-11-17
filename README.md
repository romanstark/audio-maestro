# Audio Maestro

**Turn recordings into musical evidence and editable notes. Keep the uncertainty visible.**

![Local audio](https://img.shields.io/badge/audio-stays_local-brightgreen)
![Telemetry](https://img.shields.io/badge/telemetry-none-brightgreen)
![CPU baseline](https://img.shields.io/badge/Basic_Pitch-CPU-blue)
![Optional models](https://img.shields.io/badge/SheetSage-optional-blue)
![License](https://img.shields.io/badge/license-AGPL--3.0-blue)

Audio Maestro measures local recordings and transcribes pitched parts. It works
with supplied stems, DAW exports or outputs from any separator. No DAW, UVR
installation, GPU or cloud account is required for the Basic Pitch baseline.

| Task | Result |
| --- | --- |
| Reference analysis | Levels, bands, chroma energy, envelopes, onset and tempo hypotheses |
| Basic Pitch | Pitched-note estimates on CPU, JSON and MIDI |
| Optional SheetSage2 | Lead-sheet note/chord/beat hypotheses via a local audio.cpp runtime |
| Known kick/snare stems | Calibrated onset candidates, drum MIDI, tempo carrier and one-shots |
| Reviewed timing | Source seconds mapped to clip-local quarter beats, including changing tempo |
| Long work | Background jobs, cancellation, manifests, fingerprints and paginated reports |

## Install

From this checkout, on Windows or macOS:

```text
uv tool install --python 3.12 .
audio-maestro-setup --backend basic_pitch --confirm
```

The server uses Python 3.11 or later. Setup creates a separate uv-managed Python
3.10 environment for Basic Pitch 0.4.0 and ONNX Runtime. Models are never downloaded
during inference or MCP discovery. Audio is never uploaded.

```json
{"mcpServers":{"audio-maestro":{"command":"audio-maestro","args":[]}}}
```

Examples for common clients are included in `.agents`, `.claude`, `.codex` and
`.cursor`. See [setup and migration](docs/setup.md).

## Work from a short section

1. Use `audio_status` to inspect configuration. `analyze_reference_audio` measures
   a representative section and preserves supplied stem offsets.
2. Use `transcribe_audio` on an isolated pitched part. Choose `basic_pitch` first;
   optionally compare `sheetsage` for melody/lead-sheet material. Model estimates
   require review. Raw notes are retained alongside optional cleanup.
3. Poll `get_audio_job`, read `get_audio_report` in pages, and cancel owned jobs
   with `cancel_audio_job`. Each run has a new output directory and manifest.
4. Review pitch, attacks and note lengths. Use `notes_to_beats` with reviewed
   source-second/quarter-beat anchors before placing notes in an existing song.
5. Import `notes.mid` into any DAW, or pass reviewed beat notes to its MIDI writer.
   Render and compare the result in the DAW.

MIDI without reviewed anchors uses a 120-BPM carrier preserving absolute source
seconds. That carrier does **not** identify the song's tempo. Velocity is an
explicit constant, not inferred dynamics. Model activation is retained separately.

With reviewed anchors, `transcribe_audio(align_to_audio=True)` moves note onsets to
the pitch changes of a tracked line, section by section, because model timing
drifts. `track_voice` returns that line on its own: one voice per register on the
beat grid, as notes and MIDI.

For identified kick/snare files, use `analyze_drums`. For a whole kit in one file,
`analyze_drum_kit` classifies kick, snare and hat from band rises on reviewed beats
and writes events Live Maestro's drum writer accepts. One-shot activity in another
stem is not a measured leakage score.

`describe_sound` measures an instrument in synthesizer units: attack and decay in
seconds, filter edge and its movement as a tempo division, resonance range and tail
decay per band. Together with Live Maestro's `compare_spectrum` it closes the loop
reference, MIDI, render, difference, refine.

## Optional SheetSage

SheetSage2 uses an existing [audio.cpp](https://github.com/0xShug0/audio.cpp)
CLI and the unquantized [SheetSage2 GGUF model](https://huggingface.co/audio-cpp/SheetSage2-GGUF).
Register them explicitly:

```text
audio-maestro-setup --backend sheetsage --executable "/path/to/audiocpp_cli" --model-dir "/path/to/SheetSage2-GGUF" --device cpu --confirm
```

CPU is the default. `cuda` and `metal` are explicit alternatives when the supplied
runtime supports them. CLI/model hashes are recorded and changes require
registration again. The adapter caps its native memory arenas and never downloads
weights. It returns lead-sheet estimates, not every instrument voice. The weights
retain their upstream [CC BY-NC 4.0 license](https://huggingface.co/audio-cpp/SheetSage2-GGUF#license).
Basic Pitch and this server retain their separate licenses.

## Independent tools

[Live Maestro](https://github.com/romanstark/live-maestro) controls Live, records
its actual sound and performs A/B comparisons.
[UVR Maestro](https://github.com/romanstark/uvr-maestro) separates audio.
Audio Maestro analyzes files and exports musical evidence. They exchange paths,
MIDI and documented JSON; none imports another Maestro package.

These tools do not recover exact synth presets, orchestration or effects chains.
See [limits](docs/limits.md), [architecture](docs/architecture.md) and the
[interchange contract](docs/interchange.md).

## Develop

```text
uv sync --extra dev
uv run ruff check src tests
uv run pyright
uv run pyrefly check
uv run pytest -q
```

Status: local implementation; no published release yet. Unit and protocol checks
require no DAW or downloaded model. Actual inference is validated separately.
