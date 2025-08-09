from typing import Any, Dict, Optional

from llama_index.core.callbacks.base import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType


class ReActVerboseHandler(BaseCallbackHandler):
    """Custom handler to show ReAct agent reasoning steps."""

    def __init__(self) -> None:
        super().__init__([], [])

    def _should_log(self, event_type: CBEventType) -> bool:
        return True

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Start a trace - no-op implementation."""
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[dict] = None,
    ) -> None:
        """End a trace - no-op implementation."""
        pass

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        if event_type == CBEventType.LLM:
            # Debug: print payload structure to understand format
            print(f"\n🔍 LLM Start - Payload: {payload}")

        if event_type == CBEventType.FUNCTION_CALL:
            print(f"\n🔍 Function Call Start - Payload: {payload}")

        if event_type == CBEventType.AGENT_STEP:
            print(f"\n🔍 Agent Step Start - Payload: {payload}")
        return event_id or ""

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if event_type == CBEventType.LLM:
            print(f"\n🔍 LLM End - Payload: {payload}")
            # Try different possible payload structures
            response = ""
            if payload:
                response = (
                    payload.get("response", "")
                    or payload.get("output", "")
                    or str(payload.get("result", ""))
                )
            if response:
                print(f"\n💬 LLM Output:\n{response}")

        if event_type == CBEventType.FUNCTION_CALL:
            print(f"\n🔍 Function Call End - Payload: {payload}")

        if event_type == CBEventType.AGENT_STEP:
            print(f"\n🔍 Agent Step End - Payload: {payload}")
