# Contributing

Keep analysis independent of Live and UVR. Canonical note times are original
source seconds. Model estimates must retain provenance and a review requirement.
Do not silently download dependencies during tool discovery or inference.

Run ruff, pyright, pyrefly and pytest using the commands in the README. Tests
must run without a DAW, GPU or downloaded model. Changes to inference adapters
also need a bounded real-model check. Include MIDI, offsets, cancellation and
packaging checks when those contracts change.
