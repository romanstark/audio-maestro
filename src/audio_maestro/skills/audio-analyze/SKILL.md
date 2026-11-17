---
name: audio-analyze
description: Analyze local reference audio, transcribe isolated pitched parts with Basic Pitch or optional SheetSage, and export source-timed notes and MIDI independently of a DAW.
---

Choose a representative short section and preserve every file's original source
offset. Supplied stem roles are labels, not detected identities. Inspect audio_status
before model use. Setup is explicit; tools never fetch weights during inference.

Use analyze_reference_audio for numerical evidence. Chroma energy is not a chord
transcription. Onset periodicity cannot establish meter or downbeats.

Use transcribe_audio on an isolated pitched part. Basic Pitch is the CPU baseline;
SheetSage can help with a lead-sheet melody but can fail on bass or chords. Compare
models when the part warrants it. Review attacks, pitch, octave and note ends.
Keep raw estimates: harmonic removal, lowest-voice reduction and merging can erase
real music. Activation is not MIDI velocity; supply a deliberate constant.

Lead sheet first: settle one line per section and the voice leading, then add
parts. SheetSage is the first choice for melody and harmony, Basic Pitch the second
source (it misses fast repeated notes and splits or invents notes, worst on deep
bass). MIDI that came with the source is a harmonic hint only: check it for ghost
notes and late timing, and expect supplied drum MIDI to be unusable. Use seconds
events, never a bar count from notation text, which drifts after meter changes.
Line up repeated sections and settle outliers by majority across repetitions.

Model note times can drift late section by section. track_voice follows one voice
per register on the beat grid; transcribe_audio with align_to_audio moves each note
to the pitch change of that line and reports the shift per section. An energy
threshold alone misplaces attacks that sidechain ducking holds down.

Digital silence bypasses inference. Known kick/snare stems use analyze_drums; a
mixed kit uses analyze_drum_kit, which classifies band rises on reviewed beats and
yields a pattern rather than a sound. Other-stem activity is not quantified leakage.

Poll get_audio_job and read get_audio_report in bounded pages. Cancellation stops
owned workers and retains partial evidence. Notes remain model estimates even when
the supplied timing map is reviewed.

Use notes_to_beats with reviewed source-second/quarter-beat anchors covering the
notes before placing them in an existing song. Preserve excerpt offsets. MIDI
without anchors uses a neutral 120-BPM carrier to preserve seconds, not inferred
tempo. Use any DAW's existing note writer/importer to audition and reconstruct.
Live Maestro and UVR Maestro are optional consumers/producers of these files;
their presence is not required. No exact patch or effects-chain recovery is implied.
