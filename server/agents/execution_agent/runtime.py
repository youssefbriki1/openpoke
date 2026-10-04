"""Simplified Execution Agent Runtime."""

import inspect
import json
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field

from .agent import ExecutionAgent
from .tools import get_tool_schemas, get_tool_registry
from ...config import get_settings
from ...openrouter_client import request_chat_completion
from ...logging_config import logger


@dataclass
class ExecutionResult:
    """Result from an execution agent."""
    agent_name: str
    success: bool
    response: str
    error: Optional[str] = None
    tools_executed: List[str] = field(default_factory=list)
    observed_entities: List[str] = field(default_factory=list)
    open_items_add: Dict[str, str] = field(default_factory=dict)
    open_items_close: List[str] = field(default_factory=list)


_RESOURCE_ENTITY_FIELDS = {
    "thread_id": "thread",
    "draft_id": "draft",
    "message_id": "msg",
    "trigger_id": "trigger",
}


def _resource_entities(tool_name: str, *values: Any) -> Set[str]:
    """Extract stable resource ids from tool arguments/results, never their content."""
    found: Set[str] = set()
    trigger_tool = tool_name in {"createTrigger", "updateTrigger", "listTriggers"}

    def visit(value: Any, depth: int = 0) -> None:
        if depth > 12:
            return
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith(("{", "[")):
                try:
                    visit(json.loads(stripped), depth + 1)
                except (json.JSONDecodeError, RecursionError):
                    pass
            return
        if isinstance(value, dict):
            if value.get("error"):
                return
            for key, item in value.items():
                kind = _RESOURCE_ENTITY_FIELDS.get(str(key))
                if kind and isinstance(item, (str, int)) and str(item).strip():
                    found.add(f"{kind}:{str(item).strip()}")
                elif key == "id" and trigger_tool and isinstance(item, (str, int)):
                    found.add(f"trigger:{str(item).strip()}")
                else:
                    visit(item, depth + 1)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, depth + 1)

    for value in values:
        visit(value)
    if tool_name == "gmail_create_draft" and values:
        draft_id = _draft_result_id(values[-1])
        if draft_id:
            found.add(f"draft:{draft_id}")
    return found


def _draft_result_id(result: Any) -> Optional[str]:
    """Read the draft id from either the normalized Gmail result or its data wrapper."""
    if not isinstance(result, dict):
        return None
    for record in (result, result.get("data")):
        if not isinstance(record, dict):
            continue
        draft_id = record.get("draft_id") or record.get("id")
        if draft_id is not None and str(draft_id).strip():
            return str(draft_id).strip()
    return None


def _obligation_changes(tool_name: str, arguments: Dict[str, Any], result: Any) -> Tuple[Dict[str, str], Set[str]]:
    """Track reminders and drafts that should keep their workstream available."""
    if not isinstance(result, dict) or result.get("error"):
        return {}, set()
    additions: Dict[str, str] = {}
    closures: Set[str] = set()
    if tool_name == "createTrigger" and result.get("trigger_id") is not None:
        key = f"trigger:{result['trigger_id']}"
        if result.get("status", "active") == "active":
            additions[key] = "Scheduled reminder"
    elif tool_name == "updateTrigger" and result.get("trigger_id") is not None:
        key = f"trigger:{result['trigger_id']}"
        if result.get("status") in {"paused", "completed"}:
            closures.add(key)
        else:
            additions[key] = "Scheduled reminder"
    elif tool_name == "gmail_create_draft":
        draft_id = _draft_result_id(result)
        if draft_id:
            additions[f"draft:{draft_id}"] = "Email draft awaiting approval"
    elif tool_name in {"gmail_execute_draft", "gmail_delete_draft"} and arguments.get("draft_id"):
        closures.add(f"draft:{arguments['draft_id']}")
    return additions, closures


class ExecutionAgentRuntime:
    """Manages the execution of a single agent request."""

    MAX_TOOL_ITERATIONS = 8

    # Initialize execution agent runtime with settings, tools, and agent instance
    def __init__(self, agent_name: str):
        settings = get_settings()
        self.agent = ExecutionAgent(agent_name)
        self.api_key = settings.openrouter_api_key
        self.model = settings.execution_agent_model
        self.tool_registry = get_tool_registry(agent_name=agent_name)
        self.tool_schemas = get_tool_schemas()

        if not self.api_key:
            raise ValueError("OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable.")

    # Main execution loop for running agent with LLM calls and tool execution
    async def execute(self, instructions: str) -> ExecutionResult:
        """Execute the agent with given instructions."""
        tools_executed: List[str] = []
        observed_entities: Set[str] = set()
        open_items_add: Dict[str, str] = {}
        open_items_close: Set[str] = set()
        try:
            # Build system prompt with history
            system_prompt = self.agent.build_system_prompt_with_history()

            # Start conversation with the instruction
            messages = [{"role": "user", "content": instructions}]
            final_response: Optional[str] = None

            for iteration in range(self.MAX_TOOL_ITERATIONS):
                logger.info(
                    f"[{self.agent.name}] Requesting plan (iteration {iteration + 1})"
                )
                response = await self._make_llm_call(system_prompt, messages, with_tools=True)
                assistant_message = response.get("choices", [{}])[0].get("message", {})

                if not assistant_message:
                    raise RuntimeError("LLM response did not include an assistant message")

                raw_tool_calls = assistant_message.get("tool_calls", []) or []
                parsed_tool_calls = self._extract_tool_calls(raw_tool_calls)

                assistant_entry: Dict[str, Any] = {
                    "role": "assistant",
                    "content": assistant_message.get("content", "") or "",
                }
                if raw_tool_calls:
                    assistant_entry["tool_calls"] = raw_tool_calls
                messages.append(assistant_entry)

                if not parsed_tool_calls:
                    final_response = assistant_entry["content"] or "No action required."
                    break

                for tool_call in parsed_tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("arguments", {})
                    call_id = tool_call.get("id")

                    if not tool_name:
                        logger.warning("Tool call missing name: %s", tool_call)
                        failure = {"error": "Tool call missing name; unable to execute."}
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": call_id or "unknown_tool",
                            "content": self._format_tool_result(
                                tool_name or "<unknown>", False, failure, tool_args
                            ),
                        }
                        messages.append(tool_message)
                        continue

                    tools_executed.append(tool_name)
                    logger.info(f"[{self.agent.name}] Executing tool: {tool_name}")

                    success, result = await self._execute_tool(tool_name, tool_args)

                    if success:
                        logger.info(f"[{self.agent.name}] Tool {tool_name} completed successfully")
                        record_payload = self._safe_json_dump(result)
                        observed_entities.update(_resource_entities(tool_name, tool_args, result))
                        added, closed = _obligation_changes(tool_name, tool_args, result)
                        open_items_add.update(added)
                        open_items_close.update(closed)
                    else:
                        error_detail = result.get("error") if isinstance(result, dict) else str(result)
                        logger.warning(f"[{self.agent.name}] Tool {tool_name} failed: {error_detail}")
                        record_payload = error_detail

                    self.agent.record_tool_execution(
                        tool_name,
                        self._safe_json_dump(tool_args),
                        record_payload
                    )

                    tool_message = {
                        "role": "tool",
                        "tool_call_id": call_id or tool_name,
                        "content": self._format_tool_result(tool_name, success, result, tool_args),
                    }
                    messages.append(tool_message)

            else:
                raise RuntimeError("Reached tool iteration limit without final response")

            if final_response is None:
                raise RuntimeError("LLM did not return a final response")

            self.agent.record_response(final_response)

            return ExecutionResult(
                agent_name=self.agent.name,
                success=True,
                response=final_response,
                tools_executed=tools_executed,
                observed_entities=sorted(observed_entities),
                open_items_add=open_items_add,
                open_items_close=sorted(open_items_close),
            )

        except Exception as e:
            logger.error(f"[{self.agent.name}] Execution failed: {e}")
            error_msg = str(e)
            failure_text = f"Failed to complete task: {error_msg}"
            self.agent.record_response(f"Error: {error_msg}")

            return ExecutionResult(
                agent_name=self.agent.name,
                success=False,
                response=failure_text,
                error=error_msg,
                tools_executed=tools_executed,
                observed_entities=sorted(observed_entities),
                open_items_add=open_items_add,
                open_items_close=sorted(open_items_close),
            )

    # Execute OpenRouter API call with system prompt, messages, and optional tool schemas
    async def _make_llm_call(self, system_prompt: str, messages: List[Dict], with_tools: bool) -> Dict:
        """Make an LLM call."""
        tools_to_send = self.tool_schemas if with_tools else None
        logger.info(f"[{self.agent.name}] Calling LLM with model: {self.model}, tools: {len(tools_to_send) if tools_to_send else 0}")
        return await request_chat_completion(
            model=self.model,
            messages=messages,
            system=system_prompt,
            api_key=self.api_key,
            tools=tools_to_send
        )

    # Parse and validate tool calls from LLM response into structured format
    def _extract_tool_calls(self, raw_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract tool calls from an assistant message."""
        tool_calls: List[Dict[str, Any]] = []

        for tool in raw_tools:
            function = tool.get("function", {})
            name = function.get("name", "")
            args = function.get("arguments", "")

            if isinstance(args, str):
                try:
                    args = json.loads(args) if args else {}
                except json.JSONDecodeError:
                    args = {}

            if name:
                tool_calls.append({
                    "id": tool.get("id"),
                    "name": name,
                    "arguments": args,
                })

        return tool_calls

    # Safely convert objects to JSON with fallback to string representation
    def _safe_json_dump(self, payload: Any) -> str:
        """Serialize payload to JSON, falling back to string representation."""
        try:
            return json.dumps(payload, default=str)
        except TypeError:
            return str(payload)

    # Format tool execution results into JSON structure for LLM consumption
    def _format_tool_result(
        self,
        tool_name: str,
        success: bool,
        result: Any,
        arguments: Dict[str, Any],
    ) -> str:
        """Build a structured string for tool responses."""
        if success:
            payload: Dict[str, Any] = {
                "tool": tool_name,
                "status": "success",
                "arguments": arguments,
                "result": result,
            }
        else:
            error_detail = result.get("error") if isinstance(result, dict) else str(result)
            payload = {
                "tool": tool_name,
                "status": "error",
                "arguments": arguments,
                "error": error_detail,
            }
        return self._safe_json_dump(payload)

    # Execute tool function from registry with error handling and async support
    async def _execute_tool(self, tool_name: str, arguments: Dict) -> Tuple[bool, Any]:
        """Execute a tool. Returns (success, result)."""
        tool_func = self.tool_registry.get(tool_name)
        if not tool_func:
            return False, {"error": f"Unknown tool: {tool_name}"}

        try:
            result = tool_func(**arguments)
            if inspect.isawaitable(result):
                result = await result
            return True, result
        except Exception as e:
            return False, {"error": str(e)}
