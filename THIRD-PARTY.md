# Third-party software and models

Audio Maestro code is AGPL-3.0-or-later, copyright Roman Stark. The initial
numerical analysis, drum workflow, job management and transcription cleanup were
adapted from Roman Stark's Live Maestro under the same license.

The server uses the MCP Python SDK (MIT), Pydantic (MIT), NumPy (BSD-3-Clause),
SciPy (BSD-3-Clause) and SoundFile (BSD-3-Clause; libsndfile has its own LGPL
license). Model provisioning uses uv, an external tool.

Basic Pitch 0.4.0 is [Apache-2.0](https://github.com/spotify/basic-pitch),
with ONNX Runtime (MIT) in a separate environment. Its dependencies retain their
own licenses. Model activation is not a measured MIDI velocity.

The optional audio.cpp CLI is external software with its own upstream license.
SheetSage2 and its merged MERT2 backbone weights retain
[CC BY-NC 4.0](https://huggingface.co/audio-cpp/SheetSage2-GGUF#license).
The GGUF conversion is a modification, not an official upstream release.
Attribution and provenance are provided by the model publisher and the recorded
runtime/model hashes. No CLI, SheetSage weights or Qwen weights are distributed
in this package.
