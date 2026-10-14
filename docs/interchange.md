# File interchange

All paths are local to the executing computer. A source fingerprint identifies
bytes, not audio content or instrument identity. Copy the files explicitly when
crossing computers; a Windows path is not a Mac path.

For every source, retain its SHA-256, sample rate, channel count, source frames
and `source_offset_seconds`. That offset is the original recording time of the
file's first sample. `start_seconds`/`end_seconds` select **file-local** seconds
for a transcription. Its canonical note time is:

`original seconds = source offset + actual excerpt start + excerpt-local note time`

`maestro.transcription`, version 1, contains `raw_notes`, cleaned `notes`, backend
provenance, a source fingerprint and timing information. Note rows use integer
MIDI pitch, `start_seconds`, `end_seconds` and model activation. Activation is
not a recovered MIDI velocity. `notes_to_beats` accepts reviewed anchors with
`seconds` and quarter-note `beat`; anchors must cover every note. It returns
clip-local `pitch`, `start_time`, `duration` and explicit constant `velocity`.
These values can be supplied to any compatible MIDI writer, including Live's
`write_arrangement_notes`. No importing of a Maestro Python package is needed.

`notes.mid` is SMF type 0, channel 1, 960 PPQ. Reviewed anchor intervals generate
piecewise tempo events. Without anchors, the neutral 120-BPM carrier preserves
source seconds; it supplies no detected song tempo. Tick rounding is bounded by
the file's PPQ and tempo. The JSON retains the original floating-point seconds.
When a reviewed map begins later than source zero, the MIDI origin is explicitly
reported instead of inventing an earlier tempo.

`maestro.drum-events`, version 1, retains the existing portable drum contract:
file-local samples, original-source seconds, optional original-rate sample
positions and tempo anchors. `drums.mid` uses channel 10. `tempo-map.mid` is a
carrier for tempo import. Live's `write_drum_clip` and `apply_tempo_map` accept
this JSON from any producer. One-shot records retain attack sample and offset.

`maestro.reference`, version 1, retains measured excerpts and fingerprints,
caller-supplied roles, numerical features and reviewed or proposed timing.
Neither schema certifies instrument labels, estimated notes or the quality of
a reconstruction. SheetSage native event timestamps remain excerpt-local and
their chords, keys and beats are unreviewed hypotheses.
