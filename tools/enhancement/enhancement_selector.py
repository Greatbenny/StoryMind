from __future__ import annotations
import json, os
from tools.base_tool import BaseTool, ToolResult, ToolRuntime, ToolStability, ToolStatus, ToolTier

class EnhancementSelector(BaseTool):
    name='enhancement_selector'; version='0.1.0'; tier=ToolTier.GENERATE; capability='enhancement'; provider='selector'; stability=ToolStability.BETA; runtime=ToolRuntime.HYBRID
    input_schema={'type':'object','properties':{'preferred_provider':{'type':'string','default':'auto'},'allowed_providers':{'type':'array','items':{'type':'string'}}}}
    def _providers(self):
        from tools.tool_registry import registry
        registry.ensure_discovered(); return [t for t in registry.get_by_capability(self.capability) if t.name!=self.name]
    def _prefs(self,inputs):
        if inputs.get('allowed_providers') or inputs.get('preferred_provider') not in (None,'','auto'): return inputs
        try:
            cap=json.loads(os.environ.get('STORYMIND_ROUTING_PREFERENCES_JSON','{}')).get('capabilities',{}).get(self.capability,{})
            sel=cap.get('selectedProviders') or []
            if not sel:return inputs
            x=dict(inputs); x['allowed_providers']=sel
            if cap.get('preferredProvider') in sel:x['preferred_provider']=cap['preferredProvider']
            return x
        except Exception:return inputs
    def get_status(self): return ToolStatus.AVAILABLE if any(t.get_status()==ToolStatus.AVAILABLE for t in self._providers()) else ToolStatus.UNAVAILABLE
    def estimate_cost(self,inputs):
        tool=self._select(self._prefs(inputs)); return tool.estimate_cost(inputs) if tool else 0.0
    def estimate_runtime(self,inputs):
        tool=self._select(self._prefs(inputs)); return tool.estimate_runtime(inputs) if tool else 0.0
    def _select(self,inputs):
        allowed=set(inputs.get('allowed_providers') or []); preferred=inputs.get('preferred_provider','auto')
        candidates=[t for t in self._providers() if (not allowed or t.provider in allowed) and t.get_status()==ToolStatus.AVAILABLE]
        if preferred!='auto':
            for t in candidates:
                if t.provider==preferred:return t
        return candidates[0] if candidates else None
    def execute(self,inputs):
        inputs=self._prefs(inputs); tool=self._select(inputs)
        if not tool:return ToolResult(success=False,error='No approved enhancement provider is available.')
        adapted=dict(inputs); adapted.pop('preferred_provider',None); adapted.pop('allowed_providers',None)
        result=tool.execute(adapted)
        if result.success:
            result.data.setdefault('selected_provider',tool.provider); result.data.setdefault('selected_tool',tool.name)
        return result
