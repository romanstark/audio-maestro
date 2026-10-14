"""Real stdio discovery/dispatch and isolated file jobs without a DAW or models."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl

from audio_maestro import server


def test_typed_tool_discovery_has_no_unresolved_refs():
    tools = asyncio.run(server.mcp.list_tools())
    names = {t.name for t in tools}
    assert {
        "transcribe_audio",
        "analyze_drums",
        "analyze_reference_audio",
        "notes_to_beats",
    } <= names
    assert not {"describe_audio", "capture_audio", "start_separation"} & names
    for tool in tools:
        text = json.dumps(tool.inputSchema)
        assert "$ref" not in text and "$defs" not in text, tool.name


def test_real_stdio_reference_job_and_standalone_resource(tmp_path: Path):
    source = tmp_path / "notes.wav"
    rate = 16000
    sf.write(source, 0.1 * np.sin(2 * np.pi * 220 * np.arange(rate * 2) / rate), rate)

    async def probe():
        env = {**os.environ, "AUDIO_MAESTRO_DATA_DIR": str(tmp_path / "state")}
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "audio_maestro.server"], env=env, cwd=str(tmp_path)
        )
        async with (
            stdio_client(params) as (reader, writer),
            ClientSession(reader, writer) as session,
        ):
            await session.initialize()
            resource = await session.read_resource(AnyUrl("audio://workflow"))
            assert resource.contents
            answer = await session.call_tool(
                "analyze_reference_audio",
                {
                    "reference_path": str(source),
                    "output_dir": str(tmp_path / "runs"),
                    "end_seconds": 1,
                    "confirm": True,
                },
            )
            started = answer.structuredContent
            assert started and started["ok"], answer
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                reply = await session.call_tool("get_audio_job", {"job_id": started["id"]})
                job = reply.structuredContent
                assert job
                if job["state"] in {"succeeded", "failed", "cancelled"}:
                    assert job["state"] == "succeeded", job
                    break
                await asyncio.sleep(0.05)
            else:
                raise AssertionError("Reference worker did not finish.")
            manifest = json.loads(Path(job["manifest"]).read_text())
            assert manifest["output_sha256"]
            page = await session.call_tool(
                "get_audio_report", {"job_id": started["id"], "section": "sources"}
            )
            assert page.structuredContent and page.structuredContent["total"] == 1

    asyncio.run(probe())
