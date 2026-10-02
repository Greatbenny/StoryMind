"""Capability-level video selector that routes between generation and stock providers.

Provider discovery is automatic — any BaseTool with capability="video_generation"
is picked up from the registry.  Adding a new video provider requires only creating
the tool file in tools/video/; no changes to this selector are needed.
"""

from __future__ import annotations

import json
import os

from tools.base_tool import BaseTool, ToolResult, ToolRuntime, ToolStability, ToolStatus, ToolTier


class VideoSelector(BaseTool):
    name = "video_selector"
    version = "0.4.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "selector"
    stability = ToolStability.BETA
    runtime = ToolRuntime.HYBRID
    agent_skills = ["ai-video-gen", "create-video", "ltx2"]

    capabilities = [
        "text_to_video", "image_to_video", "stock_video",
        "provider_selection", "search_video", "download_video",
    ]
    supports = {
        "user_preference_routing": True,
        "approved_provider_pools": True,
        "approved_pool_retry": True,
        "offline_fallback": True,
        "reference_image": True,
        "stock_fallback": True,
    }
    best_for = [
        "preflight routing",
        "user-facing recommendation flows",
        "switching between cloud, local, and stock video tools",
    ]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "preferred_provider": {
                "type": "string",
                "description": "Provider name or 'auto'. Valid values are discovered at runtime from the registry.",
                "default": "auto",
            },
            "allowed_providers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "User-approved provider pool. One entry means fixed routing; multiple entries allow StoryMind to choose/retry only inside this pool.",
            },
            "routing_mode": {
                "type": "string",
                "enum": ["legacy_auto", "fixed", "approved_pool"],
                "description": "Optional routing mode. If omitted, one allowed provider implies fixed; multiple imply approved_pool; none preserves legacy auto selection.",
            },
            "retry_approved_pool": {
                "type": "boolean",
                "default": True,
                "description": "Retry another user-approved provider after execution failure when multiple providers are selected.",
            },
            "operation": {
                "type": "string",
                "enum": ["text_to_video", "image_to_video", "reference_to_video", "rank"],
                "default": "text_to_video",
            },
            "target_operation": {
                "type": "string",
                "enum": ["text_to_video", "image_to_video", "reference_to_video"],
                "description": "Operation to score when operation='rank'.",
                "default": "text_to_video",
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["16:9", "9:16", "1:1"],
                "default": "16:9",
                "description": "Video aspect ratio. Passed through to the selected provider.",
            },
            "duration": {
                "type": "string",
                "description": "Duration hint (e.g., '5', '10'). Passed through to the selected provider.",
            },
            "reference_image_path": {
                "type": "string",
                "description": "Local path to a reference image for image_to_video. Auto-uploaded if the provider requires a URL.",
            },
            "reference_image_url": {
                "type": "string",
                "description": "URL of a reference image for image_to_video.",
            },
            "reference_image_urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Reference image URLs for providers that support reference-conditioned video.",
            },
            "reference_image_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Local reference image paths for providers that support reference-conditioned video.",
            },
            "image_url": {
                "type": "string",
                "description": "Alias for reference_image_url (used by some providers like Kling via fal.ai).",
            },
            "resolution": {
                "type": "string",
                "description": "Resolution hint for providers that support named output resolutions.",
            },
            "workflow_json": {
                "type": "string",
                "description": (
                    "Optional full ComfyUI workflow JSON. Routes to a custom-workflow-capable "
                    "provider (e.g. comfyui_video) based on server availability, not bundled "
                    "model readiness. Requires output_node."
                ),
            },
            "workflow_path": {
                "type": "string",
                "description": (
                    "Optional path to a ComfyUI workflow JSON file. Routes to a custom-workflow-"
                    "capable provider based on server availability. Requires output_node."
                ),
            },
            "output_node": {
                "type": "string",
                "description": "ComfyUI output node ID for a custom workflow_json/workflow_path.",
            },
            "workflow_name": {
                "type": "string",
                "description": "Optional human-readable provenance label for a custom workflow.",
            },
            "workflow_model": {
                "type": "string",
                "description": "Optional model/provenance label for a custom workflow.",
            },
            "workflow_model_stack": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Optional provenance metadata for custom workflow dependencies.",
            },
            "output_path": {"type": "string"},
        },
    }

    def _providers(self) -> list[BaseTool]:
        """Auto-discover video generation providers from the registry."""
        from tools.tool_registry import registry
        registry.ensure_discovered()
        return [t for t in registry.get_by_capability("video_generation")
                if t.name != self.name]

    @property
    def fallback_tools(self) -> list[str]:
        """Dynamically built from discovered providers + image_selector as last resort."""
        return [t.name for t in self._providers()] + ["image_selector"]

    @property
    def provider_matrix(self) -> dict[str, dict[str, str]]:
        """Built at runtime from each provider's best_for field."""
        matrix = {}
        for tool in self._providers():
            strength = ", ".join(tool.best_for) if tool.best_for else tool.name
            matrix[tool.provider] = {"tool": tool.name, "strength": strength}
        return matrix

    def get_status(self) -> ToolStatus:
        if any(tool.get_status() == ToolStatus.AVAILABLE for tool in self._providers()):
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, object]) -> float:
        inputs = self._with_runtime_preferences(inputs)
        candidates = self._filter_candidates(inputs, self._providers())
        if not candidates:
            return 0.0
        tool, _ = self._select_best_tool(inputs, candidates, self._prepare_task_context(inputs))
        return tool.estimate_cost(inputs) if tool else 0.0

    def estimate_runtime(self, inputs: dict[str, object]) -> float:
        inputs = self._with_runtime_preferences(inputs)
        candidates = self._providers()
        if not candidates:
            return 0.0
        tool, _ = self._select_best_tool(inputs, candidates, self._prepare_task_context(inputs))
        return tool.estimate_runtime(inputs) if tool else 0.0

    def execute(self, inputs: dict[str, object]) -> ToolResult:
        from lib.scoring import rank_providers

        inputs = self._with_runtime_preferences(inputs)
        candidates = self._providers()

        # Rank mode — return scored provider rankings without generating
        if inputs.get("operation") == "rank":
            rank_inputs = self._rank_inputs(inputs)
            task_context = self._prepare_task_context(rank_inputs)
            candidates = self._apply_provider_pool(rank_inputs, candidates)
            candidates = self._filter_candidates(rank_inputs, candidates)
            rankings = rank_providers(candidates, task_context)
            return ToolResult(
                success=True,
                data={
                    "rankings": self._serialize_rankings(candidates, rankings),
                    "explanation": "\n".join(r.explain() for r in rankings[:5]),
                    "normalized_task_context": task_context,
                },
            )

        # Normal generation — user-approved pools are hard boundaries.
        task_context = self._prepare_task_context(inputs)
        candidates = self._apply_provider_pool(inputs, candidates)
        ranked = self._ranked_selectable_tools(inputs, candidates, task_context)
        if not ranked:
            if inputs.get("allowed_providers"):
                return ToolResult(success=False, error="No approved video provider is currently available for this operation.")
            return ToolResult(success=False, error="No video generation provider available.")

        mode = self._routing_mode(inputs)
        retry_allowed = mode == "approved_pool" and bool(inputs.get("retry_approved_pool", True))
        attempts = []

        for index, (tool, score) in enumerate(ranked):
            if index > 0 and not retry_allowed:
                break

            adapted = dict(inputs)
            props = getattr(tool, "input_schema", {}).get("properties", {})
            if "query" in props and "query" not in adapted:
                adapted["query"] = adapted.get("prompt", "")

            if adapted.get("operation") == "image_to_video" and adapted.get("reference_image_path"):
                if "image_url" in props and "image_url" not in adapted:
                    try:
                        from tools.video._shared import upload_image_fal
                        adapted["image_url"] = upload_image_fal(adapted["reference_image_path"])
                    except Exception as e:
                        attempts.append({"provider": tool.provider, "tool": tool.name, "success": False, "error": str(e)})
                        if retry_allowed:
                            continue
                        return ToolResult(success=False, error=f"Failed to upload reference image: {e}")

            result = tool.execute(adapted)
            if result.success:
                result.data.setdefault("selected_tool", tool.name)
                result.data["selected_provider"] = tool.provider
                result.data["selection_reason"] = score.explain() if score else f"Selected {tool.provider} ({tool.name})"
                result.data["routing_mode"] = mode
                result.data["approved_providers"] = list(inputs.get("allowed_providers") or [])
                result.data["attempts"] = attempts + [{"provider": tool.provider, "tool": tool.name, "success": True}]
                if score:
                    result.data["provider_score"] = score.to_dict()
                result.data.update(self._tool_context_payload(tool))
                result.data["alternatives_considered"] = [t.name for t, _ in ranked if t.name != tool.name]
                return result

            attempts.append({"provider": tool.provider, "tool": tool.name, "success": False, "error": result.error})
            if not retry_allowed:
                return result

        last_error = attempts[-1]["error"] if attempts else "No approved provider completed successfully."
        return ToolResult(success=False, error=str(last_error), data={
            "routing_mode": mode,
            "approved_providers": list(inputs.get("allowed_providers") or []),
            "attempts": attempts,
        })

    def _with_runtime_preferences(self, inputs: dict[str, object]) -> dict[str, object]:
        """Apply per-user StoryMind routing preferences injected by Codex-web.

        Explicit tool-call routing fields always win. Runtime preferences only fill
        fields the caller did not provide.
        """
        if (
            inputs.get("allowed_providers")
            or inputs.get("routing_mode")
            or inputs.get("preferred_provider") not in (None, "", "auto")
        ):
            return inputs
        raw = os.environ.get("STORYMIND_ROUTING_PREFERENCES_JSON", "").strip()
        if not raw:
            return inputs
        try:
            payload = json.loads(raw)
            video = payload.get("capabilities", {}).get("video_generation", {})
            selected = video.get("selectedProviders") or []
            if (
                not isinstance(selected, list)
                or not selected
                or not all(isinstance(item, str) for item in selected)
            ):
                return inputs
            merged = dict(inputs)
            merged["allowed_providers"] = selected
            merged["routing_mode"] = "fixed" if len(selected) == 1 else "approved_pool"
            preferred = video.get("preferredProvider")
            if isinstance(preferred, str) and preferred in selected:
                merged["preferred_provider"] = preferred
            return merged
        except (json.JSONDecodeError, TypeError, AttributeError):
            return inputs

    def _routing_mode(self, inputs: dict[str, object]) -> str:
        explicit = str(inputs.get("routing_mode") or "").strip()
        if explicit in {"legacy_auto", "fixed", "approved_pool"}:
            return explicit
        allowed = list(inputs.get("allowed_providers") or [])
        if len(allowed) == 1:
            return "fixed"
        if len(allowed) > 1:
            return "approved_pool"
        return "legacy_auto"

    def _apply_provider_pool(self, inputs: dict[str, object], candidates: list[BaseTool]) -> list[BaseTool]:
        allowed = [str(p).strip() for p in (inputs.get("allowed_providers") or []) if str(p).strip()]
        if not allowed:
            return candidates
        approved = set(allowed)
        return [tool for tool in candidates if tool.provider in approved or tool.name in approved]

    def _ranked_selectable_tools(
        self,
        inputs: dict[str, object],
        candidates: list[BaseTool],
        task_context: dict[str, object],
    ) -> list[tuple[BaseTool, object]]:
        from lib.scoring import rank_providers

        candidates = self._apply_provider_pool(inputs, candidates)
        candidates = self._filter_candidates(inputs, candidates)
        rankings = rank_providers(candidates, task_context)
        by_name = {tool.name: tool for tool in candidates if self._tool_selectable(tool, inputs)}

        preferred = str(inputs.get("preferred_provider", "auto"))
        if self._routing_mode(inputs) == "fixed" and inputs.get("allowed_providers"):
            preferred = str(list(inputs.get("allowed_providers") or [])[0])

        ranked = []
        seen = set()
        if preferred != "auto":
            for score in rankings:
                tool = by_name.get(score.tool_name)
                if tool and (tool.provider == preferred or tool.name == preferred):
                    ranked.append((tool, score))
                    seen.add(tool.name)
                    break

        for score in rankings:
            tool = by_name.get(score.tool_name)
            if tool and tool.name not in seen:
                ranked.append((tool, score))
                seen.add(tool.name)

        return ranked

    def _select_best_tool(
        self,
        inputs: dict[str, object],
        candidates: list[BaseTool],
        task_context: dict[str, object],
    ) -> tuple[BaseTool | None, object]:
        """Select the best provider using scored ranking.

        Respects preferred_provider and environment hints as tie-breakers,
        but the scoring engine drives the primary selection.
        """
        from lib.scoring import rank_providers, ProviderScore

        preferred = inputs.get("preferred_provider", "auto")
        allowed = set(inputs.get("allowed_providers") or [])
        if allowed:
            candidates = [tool for tool in candidates if tool.provider in allowed]
        candidates = self._filter_candidates(inputs, candidates)

        env_hint = os.environ.get("VIDEO_GEN_LOCAL_MODEL", "").lower()
        env_map = {
            "wan2.1-1.3b": "wan",
            "wan2.1-14b": "wan",
            "hunyuan-1.5": "hunyuan",
            "ltx2-local": "ltx",
            "cogvideo-5b": "cogvideo",
            "cogvideo-2b": "cogvideo",
        }
        if preferred == "auto" and env_hint in env_map:
            preferred = env_map[env_hint]

        rankings = rank_providers(candidates, task_context)

        # Build tool lookup: provider → tool (first selectable per provider)
        tool_by_provider: dict[str, BaseTool] = {}
        for tool in candidates:
            if tool.provider not in tool_by_provider and self._tool_selectable(tool, inputs):
                tool_by_provider[tool.provider] = tool

        # If a preferred provider is explicitly requested and available,
        # boost it to the top unless its score is drastically worse.
        if preferred != "auto":
            for score in rankings:
                if score.provider == preferred and score.provider in tool_by_provider:
                    return tool_by_provider[score.provider], score

        # Return the highest-scored available provider
        for score in rankings:
            if score.provider in tool_by_provider:
                return tool_by_provider[score.provider], score

        return None, None

    def _prepare_task_context(self, inputs: dict[str, object]) -> dict[str, object]:
        from lib.scoring import normalize_task_context

        return normalize_task_context(
            inputs.get("task_context", {}),
            prompt=str(inputs.get("prompt", "")),
            capability=self.capability,
            operation=str(inputs.get("operation", "text_to_video")),
        )

    @staticmethod
    def _rank_inputs(inputs: dict[str, object]) -> dict[str, object]:
        rank_inputs = dict(inputs)
        rank_inputs["operation"] = inputs.get("target_operation", "text_to_video")
        return rank_inputs

    @staticmethod
    def _tool_context_payload(tool: BaseTool) -> dict[str, object]:
        info = tool.get_info()
        return {
            "selected_tool_agent_skills": info.get("agent_skills", []),
            "required_agent_skills": info.get("agent_skills", []),
            "selected_tool_usage_location": info.get("usage_location"),
            "selected_tool_best_for": info.get("best_for", []),
        }

    def _serialize_rankings(self, candidates: list[BaseTool], rankings: list[object]) -> list[dict[str, object]]:
        tool_by_name = {tool.name: tool for tool in candidates}
        serialized: list[dict[str, object]] = []
        for score in rankings:
            item = score.to_dict()
            tool = tool_by_name.get(score.tool_name)
            if tool:
                info = tool.get_info()
                item["agent_skills"] = info.get("agent_skills", [])
                item["usage_location"] = info.get("usage_location")
                item["best_for"] = info.get("best_for", [])
                item["supports"] = info.get("supports", {})
                item["status"] = str(tool.get_status())
            serialized.append(item)
        return serialized

    def _filter_candidates(
        self,
        inputs: dict[str, object],
        candidates: list[BaseTool],
    ) -> list[BaseTool]:
        # A caller-supplied custom workflow is provider-specific (ComfyUI graph
        # JSON). Route it only to custom-workflow-capable providers whose server
        # is reachable — bundled-model readiness is irrelevant in that case.
        if self._has_custom_workflow(inputs):
            return [t for t in candidates if self._custom_workflow_eligible(t, inputs)]

        operation = inputs.get("operation", "text_to_video")
        if operation == "rank":
            operation = inputs.get("target_operation", "text_to_video")

        filtered: list[BaseTool] = []
        matched_operation = False
        for tool in candidates:
            supports = getattr(tool, "supports", {})
            props = getattr(tool, "input_schema", {}).get("properties", {})

            if operation == "image_to_video":
                if supports.get("image_to_video") or "image_url" in props or "reference_image_url" in props:
                    matched_operation = True
                    if self._operation_ready(tool, "image_to_video"):
                        filtered.append(tool)
                continue

            if operation == "reference_to_video":
                if supports.get("reference_to_video") or "reference_image_urls" in props:
                    matched_operation = True
                    filtered.append(tool)
                continue

            matched_operation = True
            if self._operation_ready(tool, str(operation)):
                filtered.append(tool)

        return filtered if matched_operation else candidates

    @staticmethod
    def _operation_ready(tool: BaseTool, operation: str) -> bool:
        checker = getattr(tool, "is_operation_available", None)
        if not callable(checker):
            return True
        return bool(checker(operation))

    @staticmethod
    def _has_custom_workflow(inputs: dict[str, object]) -> bool:
        return bool(inputs.get("workflow_json") or inputs.get("workflow_path"))

    def _custom_workflow_eligible(self, tool: BaseTool, inputs: dict[str, object]) -> bool:
        """Whether a tool can run the caller-supplied custom workflow.

        Eligibility is based on server availability, not bundled-model readiness:
        a provider qualifies when it advertises ``custom_workflow`` support, an
        ``output_node`` is supplied, and its backend is reachable (status is not
        UNAVAILABLE).
        """
        if not self._has_custom_workflow(inputs):
            return False
        if not inputs.get("output_node"):
            return False
        supports = getattr(tool, "supports", {})
        if not supports.get("custom_workflow"):
            return False
        return tool.get_status() != ToolStatus.UNAVAILABLE

    def _tool_selectable(self, tool: BaseTool, inputs: dict[str, object]) -> bool:
        """A provider is selectable if it is AVAILABLE, or if it can serve a
        caller-supplied custom workflow even while bundled models report DEGRADED."""
        if tool.get_status() == ToolStatus.AVAILABLE:
            return True

        return self._custom_workflow_eligible(tool, inputs)
