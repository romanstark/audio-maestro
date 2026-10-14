# Limits

Input analysis accepts finite mono/stereo audio decodable by SoundFile. Select
sections of at most 120 seconds. Roles are caller-supplied. FFT chroma energy is
not identified notes or chords. Energy-rise periodicity cannot establish meter,
downbeats or the intended half/double-time interpretation.

Basic Pitch works best on an isolated pitched instrument. Fast lines, harmonics,
repeated notes, reverb and note-release tails can create errors. Optional harmonic
cleanup, lowest-note monophonic reduction and merging can also remove real voices.
Keep and review raw predictions. Neither backend recovers physical note-off or
played velocity. Supplied tempo anchors improve placement, not pitch accuracy.

SheetSage2 produces a lead-sheet reduction. It can improve a melody while being
wrong on bass and chords. It is not a universal all-instrument transcriber.
Its native events can invent beats/chords on unsuitable input. Digital silence
bypasses inference. CPU is available with a compatible supplied CLI; device
options do not promise equal output or performance across computers. The adapter
uses bounded native memory settings and caps inference at ten minutes per job.

Drum analysis requires identified kick/snare sources, not a mixed kit. Detector
calibration is caller-supplied. One-shots can contain other hits, reverb or leakage.
Simultaneous energy in another stem is activity, not measured contamination.

The server does not recover instrument patches, orchestrations, effect chains,
singing lyrics or subjective musical quality. It does not separate sources,
generate music or control a DAW. No Qwen backend is included.
