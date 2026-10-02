"""Agnes Video 2.5 / 2.5 Flash adapter.

Current public API (verified 2026-10):
- Base URL: https://apihub.agnes-ai.com
- Submit: POST /v1/videos
- Poll: GET /agnesapi?video_id=<id>&model_name=<model>
- Flash model is currently promotional $0/sec, 720P only, 4-12 seconds.
"""

from __future__ import annotations

import os
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

_BASE_URL = "https://apihub.agnes-ai.com"
_POLL_INTERVAL = 4
_MAX_WAIT_SECONDS = 900


class _AgnesVideoMixin:
    tier = ToolTier.GENERATE
    capability = "video_generation"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API
    dependencies = []
    agent_skills = ["ai-video-gen", "create-video"]
    capabilities = ["text_to_video", "image_to_video", "reference_to_video"]
    supports = {
        "text_to_video": True,
        "image_to_video": True,
        "reference_to_video": True,
        "reference_image": True,
        "multiple_reference_images": True,
        "aspect_ratio": True,
        "seed": True,
    }
    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=500, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["rate_limit", "timeout", "queue_full"])
    idempotency_key_fields = ["prompt", "operation", "duration", "aspect_ratio", "seed"]
    side_effects = ["writes video file to output_path", "calls Agnes API"]
    install_instructions = "Set AGNES_API_KEY in API Credentials."

    model_id = ""
    provider = ""
    price_per_second = 0.0
    allowed_sizes: tuple[str, ...] = ("720P",)
    max_reference_images = 5
    max_reference_audio = 3

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "operation": {
                "type": "string",
                "enum": ["text_to_video", "image_to_video", "reference_to_video"],
                "default": "text_to_video",
            },
            "duration": {"type": "integer", "minimum": 4, "maximum": 12, "default": 5},
            "aspect_ratio": {
                "type": "string",
                "enum": ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
                "default": "16:9",
            },
            "resolution": {"type": "string", "default": "720P"},
            "image_url": {"type": "string"},
            "reference_image_url": {"type": "string"},
            "reference_image_urls": {"type": "array", "items": {"type": "string"}},
            "reference_audio_urls": {"type": "array", "items": {"type": "string"}},
            "seed": {"type": "integer"},
            "output_path": {"type": "string"},
            "poll_interval_seconds": {"type": "integer", "minimum": 1, "default": 4},
            "timeout_seconds": {"type": "integer", "minimum": 30, "default": 900},
        },
    }

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE if os.environ.get("AGNES_API_KEY") else ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return round(self.price_per_second * int(inputs.get("duration", 5)), 4)

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 150.0

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, inputs: dict[str, Any]) -> dict[str, Any]:
        operation = str(inputs.get("operation", "text_to_video"))
        duration = max(4, min(12, int(inputs.get("duration", 5))))
        resolution = str(inputs.get("resolution", "720P")).upper()
        if resolution not in self.allowed_sizes:
            raise ValueError(
                f"{self.model_id} supports only: {', '.join(self.allowed_sizes)}"
            )

        payload: dict[str, Any] = {
            "model": self.model_id,
            "prompt": str(inputs["prompt"]),
            "seconds": str(duration),
            "size": resolution,
            "aspect_ratio": str(inputs.get("aspect_ratio", "16:9")),
            "n": 1,
        }
        if inputs.get("seed") is not None:
            payload["seed"] = int(inputs["seed"])

        image_url = inputs.get("image_url") or inputs.get("reference_image_url")
        refs = list(inputs.get("reference_image_urls") or [])
        audios = list(inputs.get("reference_audio_urls") or [])

        if operation == "text_to_video":
            payload["mode"] = "text"
        elif operation == "image_to_video":
            if not image_url:
                raise ValueError("image_to_video requires image_url or reference_image_url")
            payload["mode"] = "keyframe"
            payload["first_frame"] = image_url
        elif operation == "reference_to_video":
            if image_url:
                refs.insert(0, image_url)
            if not refs and not audios:
                raise ValueError("reference_to_video requires at least one image or audio URL")
            if len(refs) > self.max_reference_images:
                raise ValueError(f"{self.model_id} accepts at most {self.max_reference_images} reference images")
            if len(audios) > self.max_reference_audio:
                raise ValueError(f"{self.model_id} accepts at most {self.max_reference_audio} reference audio clips")
            payload["mode"] = "reference"
            if refs:
                payload["images"] = refs
            if audios:
                payload["audios"] = audios
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        return payload

    @staticmethod
    def _video_id(data: dict[str, Any]) -> str | None:
        return data.get("video_id") or data.get("id") or data.get("task_id")

    @staticmethod
    def _extract_url(data: dict[str, Any]) -> str | None:
        metadata = data.get("metadata") or {}
        if isinstance(metadata, dict):
            url = metadata.get("url") or metadata.get("video_url")
            if url:
                return str(url)
        output = data.get("output") or data.get("result") or {}
        if isinstance(output, dict):
            url = output.get("url") or output.get("video_url")
            if url:
                return str(url)
        return data.get("video_url") or data.get("url") or data.get("remixed_from_video_id")

    def _poll(self, video_id: str, headers: dict[str, str], interval: int, timeout: int) -> tuple[str, dict[str, Any]]:
        deadline = time.time() + timeout
        last: dict[str, Any] = {}
        while time.time() < deadline:
            response = requests.get(
                f"{_BASE_URL}/agnesapi",
                headers={"Authorization": headers["Authorization"]},
                params={"video_id": video_id, "model_name": self.model_id},
                timeout=30,
            )
            response.raise_for_status()
            last = response.json()
            status = str(last.get("status") or (last.get("data") or {}).get("status") or "").lower()
            url = self._extract_url(last)
            if url and status in {"", "completed", "success", "succeeded", "done"}:
                return str(url), last
            if status in {"failed", "error", "cancelled", "canceled"}:
                detail = last.get("error") or last.get("message") or last
                raise RuntimeError(f"Agnes task failed: {detail}")
            time.sleep(interval)
        raise TimeoutError(f"Agnes task {video_id} did not complete within {timeout}s")

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = os.environ.get("AGNES_API_KEY")
        if not api_key:
            return ToolResult(success=False, error="AGNES_API_KEY not set. " + self.install_instructions)
        start = time.time()
        headers = self._headers(api_key)
        try:
            payload = self._build_payload(inputs)
            response = requests.post(f"{_BASE_URL}/v1/videos", headers=headers, json=payload, timeout=60)
            if response.status_code >= 400:
                body = response.text[:1000]
                if "video_queue_full" in body or "queue" in body.lower() and "full" in body.lower():
                    return ToolResult(success=False, error=f"Agnes queue full: {body}")
                response.raise_for_status()
            submitted = response.json()
            url = self._extract_url(submitted)
            result_data = submitted
            if not url:
                video_id = self._video_id(submitted)
                if not video_id:
                    return ToolResult(success=False, error=f"Agnes returned no video_id: {submitted}")
                url, result_data = self._poll(
                    str(video_id),
                    headers,
                    int(inputs.get("poll_interval_seconds", _POLL_INTERVAL)),
                    int(inputs.get("timeout_seconds", _MAX_WAIT_SECONDS)),
                )
            video = requests.get(str(url), timeout=180)
            video.raise_for_status()
        except requests.HTTPError as exc:
            body = exc.response.text[:1000] if exc.response is not None else ""
            status = exc.response.status_code if exc.response is not None else "unknown"
            return ToolResult(success=False, error=f"Agnes HTTP {status}: {body}")
        except Exception as exc:
            return ToolResult(success=False, error=f"Agnes video generation failed: {exc}")

        output_path = Path(inputs.get("output_path") or f"{self.provider}_video.mp4")
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
                "model": self.model_id,
                "operation": inputs.get("operation", "text_to_video"),
                "output": str(output_path),
                "output_path": str(output_path),
                "format": "mp4",
                "remote_result": result_data,
                **probed,
            },
            artifacts=[str(output_path)],
            cost_usd=self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
            model=self.model_id,
        )


class AgnesVideoFlash(_AgnesVideoMixin, BaseTool):
    name = "agnes_video_flash"
    version = "0.1.0"
    provider = "agnes"
    model_id = "agnes-video-2.5-flash"
    price_per_second = 0.0
    allowed_sizes = ("720P",)
    max_reference_images = 5
    max_reference_audio = 3
    best_for = [
        "free promotional 720P video generation",
        "4-12 second text-to-video",
        "image/keyframe animation and small reference sets",
    ]
    not_good_for = ["guaranteed low-latency generation", "1080P or higher output"]


class AgnesVideoPaid(_AgnesVideoMixin, BaseTool):
    name = "agnes_video_paid"
    version = "0.1.0"
    provider = "agnes_paid"
    model_id = "agnes-video-2.5"
    price_per_second = 0.025
    allowed_sizes = ("720P", "1080P", "1K", "2K")
    max_reference_images = 8
    max_reference_audio = 3
    best_for = [
        "Agnes 2.5 paid video when Flash queue is congested",
        "720P through 2K output",
        "larger reference sets than Flash",
    ]
