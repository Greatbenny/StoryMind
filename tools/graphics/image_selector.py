"""Capability-level image selector that routes between generation and stock providers.

Provider discovery is automatic — any BaseTool with capability="image_generation"
is picked up from the registry. Adding a new image provider requires only creating
the tool file in tools/graphics/; no changes to this selector are needed.
"""

from __future__ import annotations

import json
import os
from typing import Any

from tools.base_tool import BaseTool, ToolResult, ToolRuntime, ToolStability, ToolStatus, ToolTier


class ImageSelector(BaseTool):
    name = "image_selector"
    version = "0.2.0"
    tier = ToolTier.GENERATE
    capability = "image_generation"
    provider = "selector"
    stability = ToolStability.BETA
    runtime = ToolRuntime.HYBRID
    agent_skills = ["flux-best-practices", "bfl-api"]

    capabilities = [
        "generate_image", "search_image", "download_image",
        "provider_selection", "text_to_image", "stock_image",
    ]
    supports = {
        "user_preference_routing": True,
        "offline_fallback": True,
        "stock_fallback": True,
    }
    best_for = [
        "preflight routing — pick the best image provider for the task",
        "switching between generated and stock images",
        "automatic fallback when preferred provider is unavailable",
    ]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string", "description": "Image description (used as prompt for generation or query for stock)"},
            "negative_prompt": {"type": "string"},
            "width": {"type": "integer"}, "height": {"type": "integer"}, "seed": {"type": "integer"}, "n": {"type": "integer"},
            "aspect_ratio": {"type": "string"}, "resolution": {"type": "string"},
            "generation_mode": {"type": "string", "enum": ["generate", "edit"], "default": "generate"},
            "image_url": {"type": "string"}, "image_path": {"type": "string"},
            "image_urls": {"type": "array", "items": {"type": "string"}},
            "image_paths": {"type": "array", "items": {"type": "string"}},
            "preferred_provider": {"type": "string", "default": "auto"},
            "allowed_providers": {"type": "array", "items": {"type": "string"}},
            "operation": {"type": "string", "enum": ["generate", "rank"], "default": "generate"},
            "workflow_json": {"type": "string"}, "workflow_path": {"type": "string"}, "output_node": {"type": "string"},
            "workflow_name": {"type": "string"}, "workflow_model": {"type": "string"},
            "workflow_model_stack": {"type": "array", "items": {"type": "object"}}, "output_path": {"type": "string"},
        },
    }

    def _with_runtime_preferences(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if inputs.get("allowed_providers") or inputs.get("preferred_provider") not in (None, "", "auto"):
            return inputs
        raw = os.environ.get("STORYMIND_ROUTING_PREFERENCES_JSON", "").strip()
        if not raw:
            return inputs
        try:
            cap = json.loads(raw).get("capabilities", {}).get("image_generation", {})
            selected = cap.get("selectedProviders") or []
            if not selected:
                return inputs
            merged = dict(inputs)
            merged["allowed_providers"] = selected
            pref = cap.get("preferredProvider")
            if isinstance(pref, str) and pref in selected:
                merged["preferred_provider"] = pref
            return merged
        except Exception:
            return inputs

    def _providers(self) -> list[BaseTool]:
        from tools.tool_registry import registry
        registry.ensure_discovered()
        return [t for t in registry.get_by_capability("image_generation") if t.name != self.name]

    @property
    def fallback_tools(self) -> list[str]:
        return [t.name for t in self._providers()]

    @property
    def provider_matrix(self) -> dict[str, dict[str, str]]:
        return {t.provider: {"tool": t.name, "strength": ", ".join(t.best_for) if t.best_for else t.name} for t in self._providers()}

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE if any(t.get_status() == ToolStatus.AVAILABLE for t in self._providers()) else ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        inputs = self._with_runtime_preferences(inputs)
        candidates = self._providers()
        if not candidates:
            return 0.0
        tool, _ = self._select_best_tool(inputs, candidates, self._prepare_task_context(inputs))
        return tool.estimate_cost(inputs) if tool else 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        inputs = self._with_runtime_preferences(inputs)
        import logging
        from lib.scoring import rank_providers
        logger = logging.getLogger(__name__)
        task_context = self._prepare_task_context(inputs)
        candidates = self._filter_candidates(inputs, self._providers())
        if inputs.get("operation") == "rank":
            rankings = rank_providers(candidates, task_context)
            return ToolResult(success=True, data={"rankings": self._serialize_rankings(candidates, rankings), "explanation": "\n".join(r.explain() for r in rankings[:5]), "normalized_task_context": task_context})
        tool, score = self._select_best_tool(inputs, candidates, task_context)
        if tool is None:
            return ToolResult(success=False, error="No image provider available.")
        adapted = dict(inputs)
        props = getattr(tool, "input_schema", {}).get("properties", {})
        if "query" in props and "query" not in adapted:
            adapted["query"] = adapted.get("prompt", "")
        adapted.pop("preferred_provider", None); adapted.pop("allowed_providers", None)
        for key in ("negative_prompt","width","height","seed","n","aspect_ratio","resolution","generation_mode","image_url","image_path","image_urls","image_paths","workflow_json","workflow_path","output_node","workflow_name","workflow_model","workflow_model_stack"):
            if key in adapted and key not in props:
                logger.warning("image_selector: stripped unsupported param for %s: %s", tool.name, key)
                adapted.pop(key)
        result = tool.execute(adapted)
        if result.success:
            result.data.setdefault("selected_tool", tool.name); result.data["selected_provider"] = tool.provider
            result.data["selection_reason"] = score.explain() if score else f"Selected {tool.provider} ({tool.name})"
            if score: result.data["provider_score"] = score.to_dict()
            result.data.update(self._tool_context_payload(tool))
        return result

    def _select_best_tool(self, inputs: dict[str, Any], candidates: list[BaseTool], task_context: dict[str, Any]):
        from lib.scoring import rank_providers
        preferred = inputs.get("preferred_provider", "auto")
        allowed = set(inputs.get("allowed_providers") or [])
        if allowed: candidates = [t for t in candidates if t.provider in allowed]
        candidates = self._filter_candidates(inputs, candidates)
        rankings = rank_providers(candidates, task_context)
        by_provider = {t.provider: t for t in candidates if self._tool_selectable(t, inputs)}
        if preferred != "auto":
            for score in rankings:
                if score.provider == preferred and score.provider in by_provider: return by_provider[score.provider], score
        for score in rankings:
            if score.provider in by_provider: return by_provider[score.provider], score
        return None, None

    def _prepare_task_context(self, inputs: dict[str, Any]):
        from lib.scoring import normalize_task_context
        return normalize_task_context(inputs.get("task_context", {}), prompt=inputs.get("prompt", ""), capability=self.capability, operation=inputs.get("generation_mode", inputs.get("operation", "generate")))

    @staticmethod
    def _tool_context_payload(tool: BaseTool):
        info = tool.get_info()
        return {"selected_tool_agent_skills": info.get("agent_skills", []), "required_agent_skills": info.get("agent_skills", []), "selected_tool_usage_location": info.get("usage_location"), "selected_tool_best_for": info.get("best_for", [])}

    def _serialize_rankings(self, candidates, rankings):
        by_name = {t.name: t for t in candidates}; out=[]
        for score in rankings:
            item=score.to_dict(); tool=by_name.get(score.tool_name)
            if tool:
                info=tool.get_info(); item.update({"agent_skills":info.get("agent_skills",[]),"usage_location":info.get("usage_location"),"best_for":info.get("best_for",[]),"supports":info.get("supports",{}),"status":str(tool.get_status())})
            out.append(item)
        return out

    def _filter_candidates(self, inputs, candidates):
        if self._has_custom_workflow(inputs): return [t for t in candidates if self._custom_workflow_eligible(t, inputs)]
        wants_edit = inputs.get("generation_mode") == "edit" or any(inputs.get(k) for k in ("image_url","image_path","image_urls","image_paths"))
        if not wants_edit: return candidates
        filtered=[]
        for tool in candidates:
            props=getattr(tool,"input_schema",{}).get("properties",{}); supports=getattr(tool,"supports",{})
            if supports.get("image_edit") or any(k in props for k in ("image","images","image_url","image_path","image_urls","image_paths")): filtered.append(tool)
        return filtered or candidates

    @staticmethod
    def _has_custom_workflow(inputs): return bool(inputs.get("workflow_json") or inputs.get("workflow_path"))
    def _custom_workflow_eligible(self, tool, inputs):
        return self._has_custom_workflow(inputs) and bool(inputs.get("output_node")) and bool(getattr(tool,"supports",{}).get("custom_workflow")) and tool.get_status()!=ToolStatus.UNAVAILABLE
    def _tool_selectable(self, tool, inputs): return tool.get_status()==ToolStatus.AVAILABLE or self._custom_workflow_eligible(tool, inputs)
