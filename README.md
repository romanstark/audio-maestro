# Audio Maestro

> An AI listening partner for **rebuilding music from recordings**. Hand it a reference track or a stem and get back what a producer needs to recreate it: notes and MIDI, a drum pattern, timing on your song's grid, and the shape of a sound, with every estimate marked as one.

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE) [![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/) [![Basic Pitch on CPU](https://img.shields.io/badge/Basic%20Pitch-CPU-blue.svg)](https://github.com/spotify/basic-pitch) [![Audio stays local](https://img.shields.io/badge/audio-stays%20local-brightgreen.svg)](#your-music-stays-yours) [![No telemetry](https://img.shields.io/badge/telemetry-none-brightgreen.svg)](#your-music-stays-yours)

---

## Stay in the creative flow

Rebuilding a part by ear is rewarding, but transcribing a bassline note by note, counting a hi-hat pattern or guessing why a synth sounds different eats the session.

**Audio Maestro** does the listening groundwork and hands you editable material:
- **From audio to MIDI you can play:** Melodies, basslines and lead sheets come out as notes on your song's tempo map, ready for any DAW.
- **From a drum stem to a groove:** Kick, snare and hats are read from a whole kit, even when no separator can split it.
- **From a sound to its settings:** Attack, release, filter movement and reverb tail are measured in the units you dial into a synth.
- **Honest about uncertainty:** Every result says it is an estimate, keeps the raw data next to the cleaned version, and points to the exact source seconds it came from.

No DAW is required. It works on stems from any separator, DAW exports or your own recordings.

---

## How it feels in the studio

### 1. A melody from the reference
> **You:** *"Transcribe the lead line in bars 33 to 48 of this stem and put it on my song's tempo map."*
>
> **Assistant:** Transcribed the lead with SheetSage and checked it against a tracked melody line. The model's notes drifted about 0.6 beats late from bar 41, so that section was moved back onto the line. MIDI is written; the original times stay in the report for review.

### 2. A drum pattern from a full kit
> **You:** *"There's no separate kick or snare stem. Give me the pattern of the drum bus for the chorus."*
>
> **Assistant:** Read kick, snare and hats on sixteenths and triplets over 16 bars. Halftime feel with the snare on 3, triplet hats in the last bar of each phrase, kick and snare together on the downbeat of bar 9. Pattern lines, MIDI and drum events for Live Maestro are ready.

### 3. Why does my bass sound different?
> **You:** *"Measure the reference bass in bars 65 to 68."*
>
> **Assistant:** Fast attack, about 0.3 s decay, filter edge moving between 600 Hz and 1 kHz in eighth-note triplets, a resonance between 700 Hz and 1 kHz, and a low tail that rings longer than the highs. Set your synth to that, render it, and compare with Live Maestro's `compare_spectrum`.

---

## What you can do

| Workflow | Example prompt | What Audio Maestro does |
|---|---|---|
| **🎼 Melody & lead sheet** | *"Transcribe the vocal melody of the verse"* | Transcribes pitched parts with Basic Pitch or optional SheetSage, cleans ghost notes, keeps the raw notes |
| **⏱️ Timing on your grid** | *"Put these notes on my tempo map"* | Maps source seconds to your song's beats, including tempo changes, and writes MIDI |
| **🎯 Fix drifting notes** | *"The transcription runs late in the second half, fix it"* | Tracks the actual line and moves notes back section by section |
| **🎸 One voice at a time** | *"Follow the bassline from bar 17 to 32 in E minor"* | Follows a single voice through its register on the beat grid |
| **🥁 Drums from a full kit** | *"Give me the groove of this drum stem"* | Reads kick, snare and hats from band energy on sixteenths and triplets |
| **🥁 Isolated kick & snare** | *"Turn these kick and snare stems into MIDI and one-shots"* | Detects hits precisely, proposes a tempo, cuts one-shots |
| **🎛️ Sound description** | *"What do this pad's envelope and filter do?"* | Measures attack, decay, filter movement as a tempo division, resonance and tail |
| **📊 Reference analysis** | *"Analyse the mix in this section"* | Levels, bands, chroma, envelopes, onsets and tempo hypotheses |

---

## Why the results are trustworthy

Audio analysis is guesswork made careful. Audio Maestro keeps the guesswork visible:

- **Estimates stay estimates:** Every report carries `review_required`. Raw notes sit beside cleaned notes, and correction steps report what they moved.
- **One timeline:** Note times are original source seconds. Excerpts keep their offset, and beats come only from tempo anchors you reviewed, so nothing shifts silently.
- **No invented tempo:** MIDI without your anchors uses a neutral 120 BPM carrier that preserves seconds and says so. It does not pretend to know your song's tempo.
- **Traceable runs:** Each job writes into its own folder with a manifest and file fingerprints. Long work runs in the background and can be cancelled.
- **Explicit setup:** Models are installed only when you ask. Nothing downloads during analysis or when your assistant lists the tools.

---

## MCP Tools & Resources

### Analysis & transcription
- **Status & setup:** `audio_status`, `setup_backend`
- **Reference analysis:** `analyze_reference_audio`
- **Pitched parts:** `transcribe_audio` (Basic Pitch or SheetSage, with optional `align_to_audio`), `track_voice`
- **Drums:** `analyze_drums` for isolated kick and snare, `analyze_drum_kit` for a whole kit
- **Sound:** `describe_sound`
- **Timing:** `notes_to_beats`

### Jobs & reports
- `get_audio_job`, `cancel_audio_job`, `get_audio_report` (large results page by page)

### Resources
- `audio://workflow`: the working procedure for analysis and transcription

---

## Quick start

You need [uv](https://docs.astral.sh/uv/). No DAW, no GPU and no cloud account.

### 1. Install it

```bash
uv tool install git+https://github.com/romanstark/audio-maestro.git
```

### 2. Set up Basic Pitch

```bash
audio-maestro-setup --backend basic_pitch --confirm
```

This creates a separate Python 3.10 environment for Basic Pitch 0.4.0 and ONNX Runtime. It runs once; nothing is downloaded later.

### 3. Optional: SheetSage for lead sheets

SheetSage2 gives the strongest melody and harmony estimates for lead-sheet material. It uses an existing [audio.cpp](https://github.com/0xShug0/audio.cpp) CLI and the [SheetSage2 GGUF model](https://huggingface.co/audio-cpp/SheetSage2-GGUF), which you register explicitly:

```bash
audio-maestro-setup --backend sheetsage --executable "/path/to/audiocpp_cli" --model-dir "/path/to/SheetSage2-GGUF" --device cpu --confirm
```

CPU is the default; `cuda` and `metal` work when your runtime supports them. The CLI and model hashes are recorded, and a changed file needs registering again. The weights keep their upstream [CC BY-NC 4.0 license](https://huggingface.co/audio-cpp/SheetSage2-GGUF#license).

### 4. Connect your AI assistant

Add the server to your MCP client configuration (Claude Desktop, Claude Code, Cursor, Codex):

```json
{
  "mcpServers": {
    "audio-maestro": {
      "command": "audio-maestro"
    }
  }
}
```

To run it without installing anything permanently, let `uvx` fetch it per launch:

```json
{
  "mcpServers": {
    "audio-maestro": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/romanstark/audio-maestro.git", "audio-maestro"]
    }
  }
}
```

Ready-made examples for common clients are in `.agents`, `.claude`, `.codex` and `.cursor`. See [setup](docs/setup.md) for details.

Start by asking: *"Analyse bars 1 to 8 of this stem and tell me what you hear."*

---

## Your music stays yours

- **100% local:** Your audio is analysed on your computer and never uploaded.
- **No telemetry:** No analytics, no accounts, no cloud database.
- **No surprise downloads:** Models arrive only through an explicit setup command.

---

## What stays in your hands

| Task | Why | How to do it |
|---|---|---|
| **Critical listening** | Measurements are not taste; a note can be right and still feel wrong | Audition the MIDI and the rebuilt sound in your DAW |
| **Reviewing tempo anchors** | Downbeats and meter cannot be proven from onsets | Confirm a few anchor points before placing notes in a song |
| **The exact patch or mix** | No tool recovers a synth preset, an orchestration or an effects chain from audio | Use the measurements as a starting point and refine by ear |
| **The sound of a mixed kit** | Hits cut from a full kit carry the rest of the kit with them | Use the pattern, then choose your own drum sounds |

---

## Related Maestro projects

- [Live Maestro](https://github.com/romanstark/live-maestro) works inside Ableton Live: writes the MIDI, records the rebuilt part and compares it with the reference beat by beat.
- [UVR Maestro](https://github.com/romanstark/uvr-maestro) separates and restores local audio with UVR5 models.
- [Dorico Maestro](https://github.com/romanstark/dorico-maestro) controls notation workflows in Steinberg Dorico.

Each server works independently. They exchange ordinary audio, MIDI and documented JSON; none imports another Maestro package.

---

## Development

```bash
git clone https://github.com/romanstark/audio-maestro.git
cd audio-maestro
uv sync --extra dev
```

Then run the four checks CI runs:

```bash
uv run ruff check src tests
uv run pyright
uv run pyrefly check
uv run pytest -q
```

The tests need no DAW, GPU or downloaded model. Real inference is validated separately.

---

## Documentation & Developer Resources

- [docs/setup.md](docs/setup.md): Installation, backends and client configuration
- [docs/architecture.md](docs/architecture.md): How jobs, workers and reports fit together
- [docs/interchange.md](docs/interchange.md): The JSON and MIDI contract shared with other tools
- [docs/limits.md](docs/limits.md): What the analysis can and cannot tell you
- [CONTRIBUTING.md](CONTRIBUTING.md): Contribution guidelines
- [THIRD-PARTY.md](THIRD-PARTY.md): Third-party notices and model licenses
- [LICENSE](LICENSE): AGPL-3.0 License

---

## Legal & Trademarks

Ableton and Live are trademarks of Ableton AG. Steinberg and Dorico are trademarks or registered trademarks of Steinberg Media Technologies GmbH. Basic Pitch is a project of Spotify AB.

Audio Maestro is an independent open-source project and is not affiliated with, endorsed, sponsored, or supported by Ableton AG, Steinberg Media Technologies GmbH or Spotify AB. For full third-party notices, licenses, and attributions, see [THIRD-PARTY.md](THIRD-PARTY.md).
