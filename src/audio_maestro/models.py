"""DAW-neutral musical note values."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Note:
    pitch: int
    start_time: float
    duration: float
    velocity: float = 100
