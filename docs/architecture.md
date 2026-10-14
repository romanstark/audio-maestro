# Architecture

The stdio MCP server validates requests and queues serial background jobs.
Each job runs in an isolated subprocess and writes to a unique directory.
Reports and output files are verified before a job reports success; manifests
include SHA-256 fingerprints. Cancellation stops only the owned worker tree.
Interrupted persisted jobs remain explicitly interrupted after restart.

Numerical reference analysis and identified kick/snare analysis need only the
server's NumPy, SciPy and SoundFile dependencies. Basic Pitch runs in a separate
Python 3.10/ONNX environment. Optional SheetSage uses the caller's registered
audio.cpp CLI/model. Neither backend is imported into a DAW process. No model
setup or network transfer happens during discovery or inference.

Audio Maestro has no dependency on Live Maestro or UVR Maestro. Sources can be
files from any separator or DAW. Consumers can import MIDI or map canonical
original-source seconds through reviewed beat anchors. Numerical observations,
model estimates and caller-supplied roles/tempo remain separately identified.

The server does not control a DAW, separate sources or generate songs.
Those capabilities belong to independent applications or MCPs.
