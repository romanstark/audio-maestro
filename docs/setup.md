# Setup and migration

Install the checkout with `uv tool install --python 3.12 .`, then run
`audio-maestro-setup --backend basic_pitch --confirm`. The same commands work on
Windows and macOS. Python 3.10 is managed by uv for the separate ONNX runtime;
the MCP itself can use Python 3.11 or later. Model setup is explicit.

An existing Basic Pitch environment can be reused by setting
`AUDIO_MAESTRO_BASIC_PITCH_PYTHON` to its interpreter. This is an explicit path,
not automatic discovery of another Maestro installation. Otherwise setup uses
Audio Maestro's own per-user runtime. `AUDIO_MAESTRO_DATA_DIR` overrides its state
root. Configure this environment independently on each computer.

For optional SheetSage2, obtain the matching CLI from
[audio.cpp](https://github.com/0xShug0/audio.cpp/releases) and the model directory
from [audio-cpp/SheetSage2-GGUF](https://huggingface.co/audio-cpp/SheetSage2-GGUF).
Use the unquantized `sheetsage2-orig.gguf` with its accompanying package metadata.
CLI installation and model downloads are explicit external steps. Register local
files with `audio-maestro-setup --backend sheetsage --executable PATH --model-dir
PATH --device cpu --confirm`. Registration records hashes and checks CLI support;
it does not prove inference accuracy or device availability. CUDA/Metal require
a compatible supplied runtime and drivers. No silent CPU fallback is performed.
On Windows, use `--library-dir PATH` when the runtime needs the separate CUDA
runtime DLLs supplied by the audio.cpp release. This changes only the owned
subprocess environment; no Torch package or global PATH edit is required. CPU
binaries must match the computer's instruction set. The portable CPU build can
be substantially slower. Configuration does not imply a passed inference test.

Migrating from Live Maestro's experimental analysis tools:

| Earlier Live tool | Independent replacement |
| --- | --- |
| `analyze_reference_audio` | Audio Maestro `analyze_reference_audio` |
| `transcribe_stem` | Audio Maestro `transcribe_audio`, then reviewed `notes_to_beats` and the DAW's MIDI writer |
| `analyze_drums` | Audio Maestro `analyze_drums`; Live still accepts its `analysis.json` |
| Listening/model setup | Removed from Live; Qwen is not part of Audio Maestro |
| `get_audio_job` / `cancel_audio_job` | Use the server that owns that job |
| `capture_audio` / `compare_audio` | Remain in Live Maestro |

Existing audio, JSON and MIDI remain ordinary files. The migration does not
remove old model caches or change installed MCP client configurations. Add the
new server using the provided examples and update Live to the changed build when
ready. The released Live version remains unchanged until a new release is made.
