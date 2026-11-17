"""Isolated numerical analysis and transcription worker."""

import importlib.util
import json
import sys
from pathlib import Path

if "audio_maestro" not in sys.modules:
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "audio_maestro", root / "__init__.py", submodule_search_locations=[str(root)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Worker package unavailable.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["audio_maestro"] = module
    spec.loader.exec_module(module)


def main() -> None:
    try:
        request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        if request["kind"] == "reference":
            from audio_maestro.reference import analyze

            report = analyze(request)
        elif request["kind"] == "transcription":
            from audio_maestro.pipeline import transcribe

            report = transcribe(request)
        elif request["kind"] == "kit":
            from audio_maestro.kit import analyze_kit

            report = analyze_kit(request)
        elif request["kind"] == "voice":
            from audio_maestro.lines import track_voice

            report = track_voice(request)
        elif request["kind"] == "sound":
            from audio_maestro.sound import describe

            report = describe(request)
        else:
            raise ValueError("Unsupported worker kind.")
        target = Path(request["output_dir"]) / "report.json"
        target.write_text(json.dumps(report, allow_nan=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "event": "result",
                    "ok": True,
                    "report_path": str(target),
                    "audio_paths": report.get("audio_paths", []),
                    "midi_path": report.get("midi_path"),
                    "data_paths": report.get("data_paths", []),
                    "review_required": report["review_required"],
                }
            ),
            flush=True,
        )
    except Exception as error:  # noqa: BLE001 -- report worker failures on the protocol channel
        import traceback

        traceback.print_exc(file=sys.stderr)
        print(
            json.dumps(
                {"event": "result", "ok": False, "error": str(error), "code": "analysis_failed"}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
