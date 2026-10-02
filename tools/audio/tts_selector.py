"""Capability-level text-to-speech selector that chooses among provider tools.

Provider discovery is automatic — any BaseTool with capability="tts" is picked up
from the registry. Saved StoryMind provider preferences are applied per user.
"""

from __future__ import annotations

import json
import os
from typing import Any

from tools.base_tool import BaseTool, ToolResult, ToolRuntime, ToolStability, ToolTier, ToolStatus


class TTSSelector(BaseTool):
    name = "tts_selector"
    version = "0.2.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "selector"
    stability = ToolStability.BETA
    runtime = ToolRuntime.HYBRID
    agent_skills = ["text-to-speech", "elevenlabs", "openai-docs"]
    capabilities = ["text_to_speech", "provider_selection"]
    supports = {"user_preference_routing": True, "offline_fallback": True, "multilingual": True}
    best_for = ["preflight tool selection", "user-facing recommendation flows"]
    input_schema = {
        "type": "object", "required": ["text"],
        "properties": {
            "text": {"type": "string"}, "voice_id": {"type": "string"}, "model_id": {"type": "string"},
            "stability": {"type": "number", "minimum": 0, "maximum": 1}, "similarity_boost": {"type": "number", "minimum": 0, "maximum": 1},
            "style": {"type": "number", "minimum": 0, "maximum": 1}, "instructions": {"type": "string"},
            "speaking_rate": {"type": "number", "minimum": 0.25, "maximum": 2.0}, "speed": {"type": "number", "minimum": 0.25, "maximum": 4.0},
            "pitch": {"type": "number", "minimum": -50, "maximum": 50}, "input_type": {"type": "string", "enum": ["text", "ssml"], "default": "text"},
            "voice_performance": {"type": "object"}, "sample_mode": {"type": "boolean", "default": False}, "output_format": {"type": "string"},
            "preferred_provider": {"type": "string", "default": "auto"}, "allowed_providers": {"type": "array", "items": {"type": "string"}},
            "operation": {"type": "string", "enum": ["generate", "rank"], "default": "generate"}, "output_path": {"type": "string"},
        },
    }

    def _with_runtime_preferences(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if inputs.get("allowed_providers") or inputs.get("preferred_provider") not in (None, "", "auto"):
            return inputs
        raw = os.environ.get("STORYMIND_ROUTING_PREFERENCES_JSON", "").strip()
        if not raw:
            return inputs
        try:
            cap = json.loads(raw).get("capabilities", {}).get("tts", {})
            selected = cap.get("selectedProviders") or []
            if not selected:
                return inputs
            merged = dict(inputs); merged["allowed_providers"] = selected
            pref = cap.get("preferredProvider")
            if isinstance(pref, str) and pref in selected: merged["preferred_provider"] = pref
            return merged
        except Exception:
            return inputs

    def _providers(self) -> list[BaseTool]:
        from tools.tool_registry import registry
        registry.ensure_discovered(); return [t for t in registry.get_by_capability("tts") if t.name != self.name]

    @property
    def fallback_tools(self): return [t.name for t in self._providers()]
    @property
    def provider_matrix(self): return {t.provider: {"tool": t.name, "strength": ", ".join(t.best_for) if t.best_for else t.name} for t in self._providers()}
    def get_status(self): return ToolStatus.AVAILABLE if any(t.get_status() == ToolStatus.AVAILABLE for t in self._providers()) else ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        inputs = self._with_runtime_preferences(inputs); candidates = self._providers()
        if not candidates: return 0.0
        tool, _ = self._select_best_tool(inputs, candidates, self._prepare_task_context(inputs))
        return tool.estimate_cost(inputs) if tool else 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        inputs = self._with_runtime_preferences(inputs)
        from lib.scoring import rank_providers
        task_context = self._prepare_task_context(inputs); candidates = self._providers()
        if inputs.get("operation") == "rank":
            rankings = rank_providers(candidates, task_context)
            return ToolResult(success=True, data={"rankings": self._serialize_rankings(candidates, rankings), "explanation": "\n".join(r.explain() for r in rankings[:5]), "normalized_task_context": task_context})
        tool, score = self._select_best_tool(inputs, candidates, task_context)
        if tool is None: return ToolResult(success=False, error="No TTS provider available.")
        adapted = dict(inputs); adapted.pop("preferred_provider", None); adapted.pop("allowed_providers", None)
        result = tool.execute(adapted)
        if result.success:
            result.data.setdefault("selected_tool", tool.name); result.data["selected_provider"] = tool.provider
            result.data["selection_reason"] = score.explain() if score else f"Selected {tool.provider} ({tool.name})"
            if score: result.data["provider_score"] = score.to_dict()
            result.data.update(self._tool_context_payload(tool))
        return result

    def _select_best_tool(self, inputs, candidates, task_context):
        from lib.scoring import rank_providers
        preferred = inputs.get("preferred_provider", "auto"); allowed = set(inputs.get("allowed_providers") or [])
        if allowed: candidates = [t for t in candidates if t.provider in allowed]
        rankings = rank_providers(candidates, task_context)
        by_provider = {t.provider: t for t in candidates if t.get_status() == ToolStatus.AVAILABLE}
        if preferred != "auto":
            for score in rankings:
                if score.provider == preferred and score.provider in by_provider: return by_provider[score.provider], score
        for score in rankings:
            if score.provider in by_provider: return by_provider[score.provider], score
        return None, None

    def _prepare_task_context(self, inputs):
        from lib.scoring import normalize_task_context
        return normalize_task_context(inputs.get("task_context", {}), prompt=inputs.get("text", ""), capability=self.capability, operation=inputs.get("operation", "generate"))

    @staticmethod
    def _tool_context_payload(tool):
        info = tool.get_info(); return {"selected_tool_agent_skills": info.get("agent_skills", []), "required_agent_skills": info.get("agent_skills", []), "selected_tool_usage_location": info.get("usage_location"), "selected_tool_best_for": info.get("best_for", [])}

    def _serialize_rankings(self, candidates, rankings):
        by_name={t.name:t for t in candidates}; out=[]
        for score in rankings:
            item=score.to_dict(); tool=by_name.get(score.tool_name)
            if tool:
                info=tool.get_info(); item.update({"agent_skills":info.get("agent_skills",[]),"usage_location":info.get("usage_location"),"best_for":info.get("best_for",[]),"status":str(tool.get_status())})
            out.append(item)
        return out
