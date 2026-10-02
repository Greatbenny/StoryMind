"""Capability-level StoryMind music selector with saved provider preferences."""

from __future__ import annotations

import json
import os
from typing import Any

from tools.base_tool import BaseTool, ToolResult, ToolRuntime, ToolStability, ToolStatus, ToolTier


class MusicSelector(BaseTool):
    name = "music_selector"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "music_generation"
    provider = "selector"
    stability = ToolStability.BETA
    runtime = ToolRuntime.HYBRID
    agent_skills = ["music", "acestep"]
    capabilities = ["generate_background_music", "generate_song", "generate_instrumental", "provider_selection"]
    input_schema = {"type":"object","required":["prompt"],"properties":{"prompt":{"type":"string"},"caption":{"type":"string"},"lyrics":{"type":"string"},"instrumental":{"type":"boolean"},"vocal_language":{"type":"string"},"duration_seconds":{"type":"number"},"bpm":{"type":"number"},"key":{"type":"string"},"quality":{"type":"string"},"seed":{"type":"integer"},"preferred_provider":{"type":"string","default":"auto"},"allowed_providers":{"type":"array","items":{"type":"string"}},"operation":{"type":"string","enum":["generate","rank"],"default":"generate"},"output_path":{"type":"string"}}}

    def _with_runtime_preferences(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if inputs.get("allowed_providers") or inputs.get("preferred_provider") not in (None, "", "auto"):
            return inputs
        raw = os.environ.get("STORYMIND_ROUTING_PREFERENCES_JSON", "").strip()
        if not raw:
            return inputs
        try:
            cap = json.loads(raw).get("capabilities", {}).get("music_generation", {})
            selected = cap.get("selectedProviders") or []
            if not selected:
                return inputs
            merged = dict(inputs); merged["allowed_providers"] = selected
            pref = cap.get("preferredProvider")
            if isinstance(pref, str) and pref in selected: merged["preferred_provider"] = pref
            return merged
        except Exception:
            return inputs

    def _providers(self):
        from tools.tool_registry import registry
        registry.ensure_discovered(); return [t for t in registry.get_by_capability("music_generation") if t.name != self.name]
    @property
    def fallback_tools(self): return [t.name for t in self._providers()]
    def get_status(self): return ToolStatus.AVAILABLE if any(t.get_status() == ToolStatus.AVAILABLE for t in self._providers()) else ToolStatus.UNAVAILABLE
    def estimate_cost(self, inputs):
        inputs = self._with_runtime_preferences(inputs); tool, _ = self._select(inputs, self._providers()); return tool.estimate_cost(inputs) if tool else 0.0
    def execute(self, inputs):
        inputs = self._with_runtime_preferences(inputs)
        from lib.scoring import rank_providers
        providers = self._providers()
        if inputs.get("operation") == "rank":
            available = [t for t in providers if t.get_status() == ToolStatus.AVAILABLE]
            rankings = rank_providers(available, inputs)
            return ToolResult(success=True, data={"rankings": [r.to_dict() if hasattr(r, "to_dict") else str(r) for r in rankings]})
        tool, score = self._select(inputs, providers)
        if tool is None: return ToolResult(success=False, error="No music provider is available.")
        adapted = dict(inputs); adapted.pop("preferred_provider", None); adapted.pop("allowed_providers", None); adapted.pop("operation", None)
        result = tool.execute(adapted)
        if result.success:
            result.data.setdefault("selected_tool", tool.name); result.data["selected_provider"] = tool.provider
            result.data["selection_reason"] = score.explain() if score is not None and hasattr(score, "explain") else f"Selected configured {tool.provider} music provider."
        return result
    def _select(self, inputs, candidates):
        from lib.scoring import rank_providers
        allowed = set(inputs.get("allowed_providers") or []); preferred = inputs.get("preferred_provider", "auto")
        if allowed: candidates = [t for t in candidates if t.provider in allowed]
        available = [t for t in candidates if t.get_status() == ToolStatus.AVAILABLE]
        if not available: return None, None
        rankings = rank_providers(available, inputs); by_provider = {t.provider: t for t in available}
        if preferred != "auto":
            tool = by_provider.get(str(preferred))
            if tool is None: return None, None
            score = next((r for r in rankings if getattr(r, "provider", None) == preferred), None)
            return tool, score
        for score in rankings:
            tool = by_provider.get(getattr(score, "provider", ""))
            if tool is not None: return tool, score
        return available[0], None
