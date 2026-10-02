"""Hugging Face ZeroGPU MiniMax-H3 video adapter.

Public Gradio Space: multimodalart/minimax-h3
API endpoint: /generate
Produces video with synchronized soundtrack. No HF token required for the
currently public Space, but anonymous/ZeroGPU quota and queue limits apply.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)

_SPACE = "https://multimodalart-minimax-h3.hf.space"
_INFO_URL = f"{_SPACE}/gradio_api/info"
_CALL_URL = f"{_SPACE}/gradio_api/call/generate"
_EVENT_URL = f"{_SPACE}/gradio_api/call/generate/{{event_id}}"

_CANVAS_BY_RATIO = {
    "16:9": "960x544 · 16:9 fast",
    "9:16": "544x960 · 9:16 fast",
    "1:1": "544x544 · 1:1 fast",
    "4:3": "768x576 · 4:3 fast",
    "3:4": "576x768 · 3:4 fast",
    "21:9": "1152x512 · 21:9 fast",
}


class HFMiniMaxVideo(BaseTool):
    name = "hf_minimax_video"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "hf_minimax"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API
    dependencies = []
    agent_skills = ["ai-video-gen", "create-video"]

    capabilities = ["text_to_video"]
    supports = {
        "text_to_video": True,
        "image_to_video": False,
        "native_audio": True,
        "aspect_ratio": True,
        "seed": True,
        "free_hosted": True,
    }
    best_for = [
        "free ZeroGPU text-to-video with synchronized soundtrack",
        "short cinematic clips when public Hugging Face quota is available",
    ]
    not_good_for = [
        "guaranteed availability or low latency",
        "private or SLA-backed production workloads",
    ]
    install_instructions = "No API key is required while the public Hugging Face Space remains anonymous-accessible."
    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=500, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=0)
    idempotency_key_fields = ["prompt", "duration", "aspect_ratio", "seed"]
    side_effects = ["writes video file to output_path", "calls a public Hugging Face ZeroGPU Space"]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "operation": {"type": "string", "enum": ["text_to_video"], "default": "text_to_video"},
            "duration": {"type": "integer", "minimum": 2, "maximum": 14, "default": 5},
            "aspect_ratio": {
                "type": "string",
                "enum": ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
                "default": "16:9",
            },
            "steps": {"type": "integer", "minimum": 10, "maximum": 40, "default": 28},
            "seed": {"type": "integer", "default": 42},
            "upsample_prompt": {"type": "boolean", "default": False},
            "output_path": {"type": "string"},
            "timeout_seconds": {"type": "integer", "minimum": 30, "default": 900},
        },
    }

    def get_status(self) -> ToolStatus:
        try:
            response = requests.get(_INFO_URL, timeout=5)
            if response.status_code == 200:
                info = response.json()
                if "/generate" in (info.get("named_endpoints") or {}):
                    return ToolStatus.AVAILABLE
            return ToolStatus.UNAVAILABLE
        except Exception:
            return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 180.0

    @staticmethod
    def _extract_file_url(payload: Any) -> str | None:
        if isinstance(payload, dict):
            url = payload.get("url")
            if isinstance(url, str) and url:
                return url
            path = payload.get("path")
            if isinstance(path, str) and path.startswith("http"):
                return path
            for value in payload.values():
                found = HFMiniMaxVideo._extract_file_url(value)
                if found:
                    return found
        elif isinstance(payload, list):
            for value in payload:
                found = HFMiniMaxVideo._extract_file_url(value)
                if found:
                    return found
        elif isinstance(payload, str) and payload.startswith("http"):
            return payload
        return None

    def _submit(self, inputs: dict[str, Any]) -> str:
        canvas = _CANVAS_BY_RATIO[str(inputs.get("aspect_ratio", "16:9"))]
        data = [
            str(inputs["prompt"]),
            None,
            None,
            canvas,
            int(inputs.get("duration", 5)),
            int(inputs.get("steps", 28)),
            int(inputs.get("seed", 42)),
            bool(inputs.get("upsample_prompt", False)),
        ]
        response = requests.post(_CALL_URL, json={"data": data}, timeout=30)
        response.raise_for_status()
        event_id = response.json().get("event_id")
        if not event_id:
            raise RuntimeError(f"MiniMax H3 returned no event_id: {response.text[:500]}")
        return str(event_id)

    def _wait(self, event_id: str, timeout: int) -> Any:
        with requests.get(_EVENT_URL.format(event_id=event_id), stream=True, timeout=timeout) as response:
            response.raise_for_status()
            event = None
            data_lines: list[str] = []
            for raw in response.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                line = raw.strip()
                if not line:
                    if event == "complete" and data_lines:
                        text = "\n".join(data_lines)
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            return text
                    if event in {"error", "cancelled"}:
                        raise RuntimeError("MiniMax H3 request failed: " + "\n".join(data_lines))
                    event = None
                    data_lines = []
                    continue
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data_lines.append(line.split(":", 1)[1].strip())
        raise RuntimeError("MiniMax H3 stream ended without a completed result")

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        if inputs.get("operation", "text_to_video") != "text_to_video":
            return ToolResult(success=False, error="HF MiniMax H3 currently supports text_to_video only in StoryMind.")
        start = time.time()
        try:
            event_id = self._submit(inputs)
            result = self._wait(event_id, int(inputs.get("timeout_seconds", 900)))
            url = self._extract_file_url(result)
            if not url:
                return ToolResult(success=False, error=f"MiniMax H3 returned no video URL: {str(result)[:1000]}")
            video = requests.get(url, timeout=180)
            video.raise_for_status()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            body = exc.response.text[:1000] if exc.response is not None else ""
            return ToolResult(success=False, error=f"Hugging Face MiniMax H3 HTTP {status}: {body}")
        except Exception as exc:
            return ToolResult(success=False, error=f"Hugging Face MiniMax H3 failed: {exc}")

        output_path = Path(inputs.get("output_path") or "hf_minimax_video.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(video.content)
        try:
            from tools.video._shared import probe_output
            probed = probe_output(output_path)
        except Exception:
            probed = {}
        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": "MiniMax-H3 ZeroGPU",
                "output": str(output_path),
                "output_path": str(output_path),
                "format": "mp4",
                **probed,
            },
            artifacts=[str(output_path)],
            cost_usd=0.0,
            duration_seconds=round(time.time() - start, 2),
            model="MiniMax-H3 ZeroGPU",
        )
