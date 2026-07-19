from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Optional
from urllib.parse import urlparse

import requests

from run_agent import AIAgent
from agent.turn_finalizer import ToolTurnComplete

from . import __version__


FRAMEWORK = "hermes"
DEFAULT_MAX_HOPS_PER_STEP = 30
REALSHOP_TOOL_PREFIX = "realshop__"
REALSHOP_TOOL_ORIGIN = "realshop_env"
HERMES_TOOL_ORIGIN = "hermes_native"
IDEALAB_SESSION_HEADER = "x-idealab-session-id"
IDEALAB_HOST_SUFFIX = "idealab.alibaba-inc.com"
REALSHOP_SESSION_PREFIX = "realshop-"
REALSHOP_CAPABILITY_GUIDANCE = (
    "Use any available tools, write and execute code, persist useful memory, "
    "and improve skills when helpful to maximize final net_assets."
)
log = logging.getLogger(__name__)


class RealShopStaleStep(BaseException):
    """Raised when RealShop rejects an act turn because the env advanced."""


class RealShopToolTurnComplete(ToolTurnComplete):
    """Stop Hermes' inner loop after RealShop marks the hook complete."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        final_response: str = "",
    ) -> None:
        super().__init__(
            final_response or "RealShop step completed.",
            reason="tool_turn_complete(realshop_end_of_step)",
        )
        self.messages = messages


def _ensure_realshop_sdk_path() -> None:
    candidates = [
        os.environ.get("REALSHOP_AGENT_SDK_ROOT"),
        os.path.abspath(os.path.join(os.getcwd(), "agent")),
        os.path.abspath(os.path.join(os.getcwd(), "..", "realshop-dev", "agent")),
    ]
    for path in candidates:
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)
            return


_ensure_realshop_sdk_path()
from sdk.realshop_tool_client import RealShopToolClient  # noqa: E402


def _is_idealab_base_url(base_url: Optional[str]) -> bool:
    if not base_url:
        return False
    try:
        host = urlparse(str(base_url)).hostname or ""
    except Exception:
        host = ""
    host = host.lower()
    return host == IDEALAB_HOST_SUFFIX or host.endswith(f".{IDEALAB_HOST_SUFFIX}")


def _is_claude_model(model: Optional[str]) -> bool:
    return "claude" in str(model or "").lower()


def _realshop_request_overrides(base_url: Optional[str], run_id: str) -> dict[str, Any]:
    if not _is_idealab_base_url(base_url):
        return {}
    return {
        "extra_headers": {
            IDEALAB_SESSION_HEADER: str(run_id),
        },
    }


def _realshop_session_id(run_id: str) -> str:
    run_id = str(run_id or "").strip()
    return f"{REALSHOP_SESSION_PREFIX}{run_id or 'unknown'}"


def _realshop_session_db() -> Any | None:
    try:
        from hermes_state import SessionDB
    except Exception as exc:  # noqa: BLE001
        log.warning("Hermes SessionDB import failed; RealShop trace will still be recorded: %s", exc)
        return None
    try:
        return SessionDB()
    except Exception as exc:  # noqa: BLE001
        log.warning("Hermes SessionDB init failed; RealShop trace will still be recorded: %s", exc)
        return None


def _tool_call_name(tool_call: Any) -> str:
    function = getattr(tool_call, "function", None)
    if function is not None and getattr(function, "name", None) is not None:
        return str(function.name)
    if isinstance(tool_call, dict):
        function = tool_call.get("function") or {}
        return str(function.get("name") or tool_call.get("name") or "")
    return str(getattr(tool_call, "name", "") or "")


def _tool_call_arguments(tool_call: Any) -> str:
    function = getattr(tool_call, "function", None)
    value = None
    if function is not None and getattr(function, "arguments", None) is not None:
        value = function.arguments
    elif isinstance(tool_call, dict):
        function = tool_call.get("function") or {}
        value = function.get("arguments", tool_call.get("arguments"))
    else:
        value = getattr(tool_call, "arguments", None)

    if value is None or value == "":
        return "{}"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _tool_call_id(tool_call: Any, index: int) -> str:
    if getattr(tool_call, "id", None):
        return str(tool_call.id)
    if isinstance(tool_call, dict) and tool_call.get("id"):
        return str(tool_call["id"])
    return f"call_realshop_{int(time.time() * 1000)}_{index}"


def _openai_tool_call(
    tool_call: Any,
    index: int,
    *,
    realshop_tool_names: Optional[set[str]] = None,
) -> dict[str, Any]:
    name = _tool_call_name(tool_call)
    out = {
        "id": _tool_call_id(tool_call, index),
        "type": "function",
        "tool_origin": _tool_origin_for_name(name, realshop_tool_names),
        "function": {
            "name": name,
            "arguments": _tool_call_arguments(tool_call),
        },
    }
    extra_content = getattr(tool_call, "extra_content", None)
    if extra_content is None and isinstance(tool_call, dict):
        extra_content = tool_call.get("extra_content")
    if extra_content is not None:
        if hasattr(extra_content, "model_dump"):
            extra_content = extra_content.model_dump()
        out["extra_content"] = extra_content
    return out


def _end_of_step_tool_call() -> dict[str, Any]:
    return {
        "id": f"call_realshop_end_{int(time.time() * 1000)}",
        "type": "function",
        "tool_origin": REALSHOP_TOOL_ORIGIN,
        "function": {"name": "end_of_step", "arguments": "{}"},
    }


def _assistant_content(assistant_message: Any) -> Optional[str]:
    content = getattr(assistant_message, "content", None)
    if content is None and isinstance(assistant_message, dict):
        content = assistant_message.get("content")
    if content is None:
        return None
    return str(content)


def _assistant_reasoning(assistant_message: Any) -> Optional[str]:
    for attr in ("reasoning_content", "reasoning"):
        value = getattr(assistant_message, attr, None)
        if value:
            return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if isinstance(assistant_message, dict):
        for key in ("reasoning_content", "reasoning"):
            value = assistant_message.get(key)
            if value:
                return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return None


def _tool_origin_for_name(
    name: str,
    realshop_tool_names: Optional[set[str]] = None,
) -> str:
    return (
        REALSHOP_TOOL_ORIGIN
        if _is_realshop_tool_name(name, realshop_tool_names)
        else HERMES_TOOL_ORIGIN
    )


def _is_realshop_tool_name(
    name: str,
    realshop_tool_names: Optional[set[str]] = None,
) -> bool:
    name = str(name or "")
    return (
        name == "end_of_step"
        or name.startswith(REALSHOP_TOOL_PREFIX)
        or name in (realshop_tool_names or set())
    )


def _realshop_env_tool_name(name: str) -> str:
    name = str(name or "")
    if name.startswith(REALSHOP_TOOL_PREFIX):
        return name[len(REALSHOP_TOOL_PREFIX):]
    return name


def _exposed_realshop_tool(tool: dict[str, Any]) -> dict[str, Any]:
    out = dict(tool)
    fn = dict(out.get("function") or {})
    env_name = str(fn.get("name") or "")
    fn["name"] = env_name
    fn["description"] = f"[RealShop env] {fn.get('description') or ''}".strip()
    fn["x-tool-origin"] = REALSHOP_TOOL_ORIGIN
    fn["x-realshop-tool-name"] = env_name
    out["function"] = fn
    out["tool_origin"] = REALSHOP_TOOL_ORIGIN
    out["x-realshop-tool-name"] = env_name
    return out


def _realshop_act_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    out = dict(tool_call)
    fn = dict(out.get("function") or {})
    hermes_name = str(fn.get("name") or "")
    env_name = _realshop_env_tool_name(hermes_name)
    fn["name"] = env_name
    out["function"] = fn
    out["tool_origin"] = REALSHOP_TOOL_ORIGIN
    if hermes_name != env_name:
        out["hermes_tool_name"] = hermes_name
    else:
        out.pop("hermes_tool_name", None)
    return out


def _with_tool_calls(assistant_message: Any, tool_calls: list[Any]) -> None:
    if isinstance(assistant_message, dict):
        assistant_message["tool_calls"] = tool_calls
    else:
        assistant_message.tool_calls = tool_calls


def _token_usage(
    assistant_message: Any,
    *,
    provider: Optional[str] = None,
    api_mode: Optional[str] = None,
) -> Optional[dict[str, int]]:
    usage = getattr(assistant_message, "usage", None)
    if usage is None and isinstance(assistant_message, dict):
        usage = assistant_message.get("usage")
    if usage is None:
        return None
    try:
        from agent.usage_pricing import normalize_usage
        normalized = normalize_usage(usage, provider=provider, api_mode=api_mode)
        return {
            "input": int(normalized.input_tokens or 0),
            "output": int(normalized.output_tokens or 0),
            "cache_read": int(normalized.cache_read_tokens or 0),
            "cache_write": int(normalized.cache_write_tokens or 0),
            "reasoning": int(normalized.reasoning_tokens or 0),
            "total": int(normalized.total_tokens or 0),
        }
    except Exception:
        if isinstance(usage, dict):
            prompt = int(usage.get("prompt_tokens", usage.get("input", 0)) or 0)
            completion = int(usage.get("completion_tokens", usage.get("output", 0)) or 0)
            cached = int(usage.get("cached_tokens", usage.get("cached", 0)) or 0)
        else:
            prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion = int(getattr(usage, "completion_tokens", 0) or 0)
            cached = int(getattr(usage, "cached_tokens", 0) or 0)
    return {
        "input": max(0, prompt - cached),
        "output": completion,
        "cache_read": cached,
        "cache_write": 0,
        "reasoning": 0,
        "total": prompt + completion,
    }


def _raw_prompt_tokens(assistant_message: Any) -> int:
    usage = getattr(assistant_message, "usage", None)
    if usage is None and isinstance(assistant_message, dict):
        usage = assistant_message.get("usage")
    if usage is None:
        return 0
    try:
        if isinstance(usage, dict):
            return int(usage.get("prompt_tokens") or 0)
        return int(getattr(usage, "prompt_tokens", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _compressor_count(agent: Any) -> int:
    compressor = getattr(agent, "context_compressor", None)
    try:
        return int(getattr(compressor, "compression_count", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _compressor_prompt_tokens(agent: Any) -> int:
    compressor = getattr(agent, "context_compressor", None)
    try:
        return int(getattr(compressor, "last_prompt_tokens", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _realshop_context(
    agent: Any,
    assistant_message: Optional[Any] = None,
) -> Optional[dict[str, Any]]:
    prompt_tokens = (
        _raw_prompt_tokens(assistant_message)
        if assistant_message is not None
        else 0
    )
    if prompt_tokens <= 0:
        prompt_tokens = _compressor_prompt_tokens(agent)
    current_count = _compressor_count(agent)
    reported_count = int(
        getattr(agent, "_realshop_reported_compression_count", 0) or 0
    )
    context: dict[str, Any] = {}
    if prompt_tokens > 0:
        context["tokens"] = prompt_tokens
    if current_count > reported_count:
        context["compacted"] = True
    return context or None


def _mark_realshop_context_reported(agent: Any) -> None:
    current_count = _compressor_count(agent)
    reported_count = int(
        getattr(agent, "_realshop_reported_compression_count", 0) or 0
    )
    if current_count > reported_count:
        agent._realshop_reported_compression_count = current_count


def _usage_snapshot(agent: Any) -> dict[str, int]:
    return {
        "input": int(getattr(agent, "session_input_tokens", 0) or 0),
        "output": int(getattr(agent, "session_output_tokens", 0) or 0),
        "cache_read": int(getattr(agent, "session_cache_read_tokens", 0) or 0),
        "cache_write": int(getattr(agent, "session_cache_write_tokens", 0) or 0),
        "reasoning": int(getattr(agent, "session_reasoning_tokens", 0) or 0),
        "total": int(getattr(agent, "session_total_tokens", 0) or 0),
    }


def _usage_delta(before: dict[str, int], agent: Any) -> Optional[dict[str, int]]:
    after = _usage_snapshot(agent)
    delta = {
        key: max(0, after.get(key, 0) - int(before.get(key, 0) or 0))
        for key in ("input", "output", "cache_read", "cache_write", "reasoning", "total")
    }
    return delta if any(delta.values()) else None


def _message_signature(message: dict[str, Any]) -> str:
    return json.dumps(message, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _trace_message(message: dict[str, Any]) -> Optional[dict[str, Any]]:
    role = message.get("role")
    if role not in ("assistant", "tool"):
        return None
    out = dict(message)
    if role == "assistant":
        out.setdefault("tool_origin", HERMES_TOOL_ORIGIN)
        normalized_tool_calls = []
        for index, tc in enumerate(out.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            normalized = dict(tc)
            name = str((normalized.get("function") or {}).get("name") or normalized.get("name") or "")
            normalized.setdefault("id", f"call_hermes_{int(time.time() * 1000)}_{index}")
            normalized.setdefault("type", "function")
            normalized.setdefault(
                "tool_origin",
                _tool_origin_for_name(name),
            )
            normalized_tool_calls.append(normalized)
        if normalized_tool_calls:
            out["tool_calls"] = normalized_tool_calls
    elif role == "tool":
        out.setdefault("tool_origin", HERMES_TOOL_ORIGIN)
    return out


def _new_trace_messages(
    prior_history: list[dict[str, Any]],
    current_history: list[dict[str, Any]],
    *,
    exclude_messages: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    prior_counts: dict[str, int] = {}
    for msg in prior_history:
        if isinstance(msg, dict):
            sig = _message_signature(msg)
            prior_counts[sig] = prior_counts.get(sig, 0) + 1
    for msg in exclude_messages or []:
        if isinstance(msg, dict):
            sig = _message_signature(msg)
            prior_counts[sig] = prior_counts.get(sig, 0) + 1

    out: list[dict[str, Any]] = []
    for msg in current_history:
        if not isinstance(msg, dict):
            continue
        sig = _message_signature(msg)
        if prior_counts.get(sig, 0) > 0:
            prior_counts[sig] -= 1
            continue
        trace_msg = _trace_message(msg)
        if trace_msg is not None:
            out.append(trace_msg)
    return out


def _reorder_end_of_step_last(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normal_calls = [
        tc for tc in tool_calls
        if _realshop_env_tool_name((tc.get("function") or {}).get("name")) != "end_of_step"
    ]
    end_calls = [
        tc for tc in tool_calls
        if _realshop_env_tool_name((tc.get("function") or {}).get("name")) == "end_of_step"
    ]
    if not end_calls:
        return tool_calls
    return [*normal_calls, end_calls[-1]]


def _append_tool_results(
    messages: list[dict[str, Any]],
    act_resp: dict[str, Any],
    *,
    name_by_tool_call_id: Optional[dict[str, str]] = None,
) -> None:
    for result in act_resp.get("tool_results", []):
        tool_call_id = result["tool_call_id"]
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": (name_by_tool_call_id or {}).get(tool_call_id, result["name"]),
            "tool_origin": result.get("tool_origin", REALSHOP_TOOL_ORIGIN),
            "content": result["content"],
        })


class RealShopHermesAgent(AIAgent):
    def __init__(
        self,
        *,
        realshop_client: RealShopToolClient,
        model: str,
        base_url: Optional[str],
        api_key: Optional[str],
        provider: Optional[str],
        quiet: bool,
        max_iterations: int,
    ) -> None:
        run_id = str(getattr(realshop_client, "run_id", "") or "")
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            provider=provider,
            model=model,
            max_iterations=max_iterations,
            quiet_mode=quiet,
            save_trajectories=False,
            skip_context_files=True,
            session_id=_realshop_session_id(run_id),
            session_db=_realshop_session_db(),
            platform="realshop",
            request_overrides=_realshop_request_overrides(
                base_url,
                run_id,
            ),
        )
        # RealShop runs headlessly, so streaming adds an SSE failure surface
        # without any user-visible benefit. Retry transient API failures using
        # the normal non-streaming path instead.
        self._disable_streaming = True
        self._api_max_retries = 6
        self.realshop_client = realshop_client
        self._native_tools = list(self.tools or [])
        self._native_tool_names = {
            str((tool.get("function") or {}).get("name"))
            for tool in self._native_tools
            if (tool.get("function") or {}).get("name")
        }
        self._realshop_step_done = False
        self._realshop_last_act: Optional[dict[str, Any]] = None
        self._realshop_trace_msgs_for_act: list[dict[str, Any]] = []
        self._realshop_reported_compression_count = _compressor_count(self)
        self.refresh_realshop_tools()

    def refresh_realshop_tools(self) -> None:
        realshop_tools = [
            _exposed_realshop_tool(tool)
            for tool in self.realshop_client.tools()
        ]
        raw_realshop_names = {
            str((tool.get("function") or {}).get("name"))
            for tool in realshop_tools
            if (tool.get("function") or {}).get("name")
        }
        collisions = raw_realshop_names & self._native_tool_names
        if collisions:
            names = ", ".join(sorted(collisions))
            raise ValueError(
                "RealShop env tool names collide with Hermes native tools: "
                f"{names}"
            )
        legacy_prefixed_aliases = {
            f"{REALSHOP_TOOL_PREFIX}{name}"
            for name in raw_realshop_names
        }
        self._realshop_tool_names = {
            *raw_realshop_names,
            *legacy_prefixed_aliases,
        }
        tools = [*self._native_tools, *realshop_tools]
        self.tools = tools
        self.valid_tool_names = {
            str((tool.get("function") or {}).get("name"))
            for tool in tools
            if (tool.get("function") or {}).get("name")
        } | legacy_prefixed_aliases

    def _anthropic_prompt_cache_policy(self, *args, **kwargs) -> tuple[bool, bool]:
        base_url = kwargs.get("base_url", getattr(self, "base_url", None))
        model = kwargs.get("model", getattr(self, "model", None))
        if _is_idealab_base_url(base_url) and _is_claude_model(model):
            return True, False

        parent_policy = getattr(super(), "_anthropic_prompt_cache_policy", None)
        if parent_policy is None:
            return False, False
        return parent_policy(*args, **kwargs)

    def queue_realshop_trace_messages(self, messages: list[dict[str, Any]]) -> None:
        self._realshop_trace_msgs_for_act = list(messages)

    def _execute_tool_calls(
        self,
        assistant_message: Any,
        messages: list[dict[str, Any]],
        effective_task_id: str,
        api_call_count: int = 0,
    ) -> None:
        raw_tool_calls = getattr(assistant_message, "tool_calls", None) or []
        realshop_tool_names = getattr(self, "_realshop_tool_names", set())
        tool_calls = _reorder_end_of_step_last([
            _openai_tool_call(
                tool_call,
                index,
                realshop_tool_names=realshop_tool_names,
            )
            for index, tool_call in enumerate(raw_tool_calls)
        ])
        realshop_tool_calls = [
            tc for tc in tool_calls
            if _is_realshop_tool_name(
                (tc.get("function") or {}).get("name", ""),
                realshop_tool_names,
            )
        ]
        native_raw_tool_calls = [
            tool_call for tool_call in raw_tool_calls
            if not _is_realshop_tool_name(
                _tool_call_name(tool_call),
                realshop_tool_names,
            )
        ]
        native_tool_calls = [
            tc for tc in tool_calls
            if not _is_realshop_tool_name(
                (tc.get("function") or {}).get("name", ""),
                realshop_tool_names,
            )
        ]
        assistant_msg = {
            "role": "assistant",
            "content": _assistant_content(assistant_message),
            "tool_calls": tool_calls,
            "tool_origin": (
                REALSHOP_TOOL_ORIGIN
                if realshop_tool_calls and not native_tool_calls
                else HERMES_TOOL_ORIGIN
                if native_tool_calls and not realshop_tool_calls
                else "mixed"
            ),
        }
        reasoning = _assistant_reasoning(assistant_message)
        if reasoning:
            assistant_msg["reasoning_content"] = reasoning

        if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "assistant":
            messages[-1]["tool_calls"] = tool_calls
            messages[-1]["tool_origin"] = assistant_msg["tool_origin"]

        if native_raw_tool_calls:
            native_assistant_msg = dict(assistant_msg)
            native_assistant_msg["tool_calls"] = native_tool_calls
            native_assistant_msg["tool_origin"] = HERMES_TOOL_ORIGIN
            native_start_idx = len(messages)
            saved_tool_calls = raw_tool_calls
            try:
                _with_tool_calls(assistant_message, native_raw_tool_calls)
                super()._execute_tool_calls(
                    assistant_message,
                    messages,
                    effective_task_id,
                    api_call_count,
                )
            finally:
                _with_tool_calls(assistant_message, saved_tool_calls)
            native_results = [
                {**msg, "tool_origin": HERMES_TOOL_ORIGIN}
                for msg in messages[native_start_idx:]
                if isinstance(msg, dict) and msg.get("role") == "tool"
            ]
            trace_messages = [
                *self._realshop_trace_msgs_for_act,
                native_assistant_msg,
                *native_results,
            ]
            try:
                native_token_usage = None
                if not realshop_tool_calls:
                    native_token_usage = _token_usage(
                        assistant_message,
                        provider=getattr(self, "provider", None),
                        api_mode=getattr(self, "api_mode", None),
                    )
                native_context = _realshop_context(self, assistant_message)
                self.realshop_client.act(
                    token_usage=native_token_usage,
                    messages=trace_messages,
                    context=native_context,
                )
                _mark_realshop_context_reported(self)
            except requests.HTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status == 425:
                    self._realshop_trace_msgs_for_act = []
                    raise RealShopStaleStep() from exc
                raise
            self._realshop_trace_msgs_for_act = []

        if not realshop_tool_calls:
            self._realshop_last_act = {"ok": True, "step_done": False, "trace_only": True}
            return

        realshop_act_calls = [_realshop_act_tool_call(tc) for tc in realshop_tool_calls]
        realshop_assistant_msg = dict(assistant_msg)
        realshop_assistant_msg["tool_calls"] = realshop_act_calls
        realshop_assistant_msg["tool_origin"] = REALSHOP_TOOL_ORIGIN
        name_by_id = {
            tc["id"]: (tc.get("function") or {}).get("name", "")
            for tc in realshop_tool_calls
        }
        trace_msgs = list(self._realshop_trace_msgs_for_act)
        try:
            context = _realshop_context(self, assistant_message)
            act_resp = self.realshop_client.act(
                token_usage=_token_usage(
                    assistant_message,
                    provider=getattr(self, "provider", None),
                    api_mode=getattr(self, "api_mode", None),
                ),
                messages=[*trace_msgs, realshop_assistant_msg],
                context=context,
            )
            _mark_realshop_context_reported(self)
        except requests.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 425:
                self._realshop_trace_msgs_for_act = []
                raise RealShopStaleStep() from exc
            raise

        self._realshop_trace_msgs_for_act = []
        self._realshop_last_act = act_resp
        self._realshop_step_done = bool(act_resp.get("step_done"))
        _append_tool_results(messages, act_resp, name_by_tool_call_id=name_by_id)
        if self._realshop_step_done:
            try:
                self._flush_messages_to_session_db(messages)
            except Exception as exc:
                log.warning("Hermes SessionDB flush failed after RealShop end_of_step: %s", exc)
            raise RealShopToolTurnComplete(
                messages,
                final_response=_assistant_content(assistant_message),
            )


def _tick_step(obs: dict[str, Any]) -> int:
    tick = obs.get("tick") or {}
    if tick.get("step") is not None:
        return int(tick["step"])
    day = int(tick.get("day", 0) or 0)
    hour = int(tick.get("hour", 0) or 0)
    return (day - 1) * 24 + hour if day > 0 else 0


def _force_end_of_step(
    client: RealShopToolClient,
    history: list[dict[str, Any]],
    trace_messages: list[dict[str, Any]],
    reason: str,
    *,
    token_usage: Optional[dict[str, int]] = None,
    context: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    outbound_messages = list(trace_messages)
    merge_with_last_assistant = bool(
        history
        and isinstance(history[-1], dict)
        and history[-1].get("role") == "assistant"
        and not history[-1].get("tool_calls")
    )
    if merge_with_last_assistant:
        assistant_msg = dict(history[-1])
        content = assistant_msg.get("content")
        assistant_msg["content"] = (
            f"{content.rstrip()}\n\n{reason}"
            if isinstance(content, str) and content.strip()
            else reason
        )
        assistant_msg["tool_origin"] = REALSHOP_TOOL_ORIGIN
        assistant_msg["tool_calls"] = [_end_of_step_tool_call()]
        history[-1] = assistant_msg
        if (
            outbound_messages
            and isinstance(outbound_messages[-1], dict)
            and outbound_messages[-1].get("role") == "assistant"
            and not outbound_messages[-1].get("tool_calls")
        ):
            outbound_messages[-1] = dict(assistant_msg)
        else:
            outbound_messages.append(dict(assistant_msg))
    else:
        assistant_msg = {
            "role": "assistant",
            "content": reason,
            "tool_origin": REALSHOP_TOOL_ORIGIN,
            "tool_calls": [_end_of_step_tool_call()],
        }
        history.append(assistant_msg)
        outbound_messages.append(assistant_msg)
    act_resp = client.act(
        messages=outbound_messages,
        token_usage=token_usage,
        context=context,
    )
    _append_tool_results(history, act_resp)
    return history


def _build_agent(args: argparse.Namespace, client: RealShopToolClient) -> RealShopHermesAgent:
    model = args.model or os.environ.get("MODEL_NAME", "")
    return RealShopHermesAgent(
        realshop_client=client,
        model=model,
        base_url=args.openai_base_url or os.environ.get("OPENAI_BASE_URL"),
        api_key=args.openai_api_key or os.environ.get("OPENAI_API_KEY"),
        provider=args.provider or os.environ.get("HERMES_PROVIDER"),
        quiet=args.quiet,
        max_iterations=max(1, int(args.max_hops_per_step)),
    )


def run(args: argparse.Namespace) -> int:
    client = RealShopToolClient(
        args.base_url,
        args.run_id,
        args.agent_id,
        timeout=args.timeout,
        observation_timeout=args.observation_timeout,
    )
    agent = _build_agent(args, client)
    client.register(
        framework=FRAMEWORK,
        model=args.model or os.environ.get("MODEL_NAME"),
        version=__version__,
        extra={
            "adapter": "realshop_adapter",
            "hermes_root": os.getcwd(),
            "max_hops_per_step": args.max_hops_per_step,
        },
    )

    system_prompt: Optional[str] = None
    history: list[dict[str, Any]] = []
    step_count = 0

    while True:
        try:
            obs = client.observation()
        except requests.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 410:
                if not args.quiet:
                    print("[realshop-adapter] run finished", file=sys.stderr)
                return 0
            if not args.quiet:
                print(f"[realshop-adapter] observation error: {exc}", file=sys.stderr)
            time.sleep(args.retry_sleep)
            continue

        if args.max_steps is not None and _tick_step(obs) >= int(args.max_steps):
            if not args.quiet:
                print(f"[realshop-adapter] reached max_steps={args.max_steps}", file=sys.stderr)
            return 0

        brief = obs.get("brief") or {}
        if system_prompt is None and brief.get("system_prompt"):
            system_prompt = (
                f"{str(brief['system_prompt']).rstrip()}\n\n"
                f"{REALSHOP_CAPABILITY_GUIDANCE}"
            )

        observation_msg = {"role": "user", "content": obs.get("text", "") or ""}
        history_len_before_step = len(history)
        agent._realshop_step_done = False
        agent._realshop_last_act = None

        stale_step = False
        no_tool_token_usage = None
        agent.refresh_realshop_tools()
        agent.queue_realshop_trace_messages([])
        history_before_step = list(history)
        usage_before_step = _usage_snapshot(agent)
        try:
            result = agent.run_conversation(
                observation_msg["content"],
                system_message=system_prompt,
                conversation_history=history,
            )
        except RealShopStaleStep:
            history = history[:history_len_before_step]
            stale_step = True
        else:
            history = list(result.get("messages") or history)

        if stale_step:
            continue

        if not agent._realshop_step_done:
            if agent._realshop_last_act is None:
                agent._realshop_trace_msgs_for_act = [
                    *agent._realshop_trace_msgs_for_act,
                    *_new_trace_messages(
                        history_before_step,
                        history,
                        exclude_messages=[observation_msg],
                    ),
                ]
                no_tool_token_usage = _usage_delta(usage_before_step, agent)
            fallback_context = _realshop_context(agent)
            history = _force_end_of_step(
                client,
                history,
                agent._realshop_trace_msgs_for_act,
                "[fallback] Hermes did not call end_of_step; releasing the hook.",
                token_usage=no_tool_token_usage,
                context=fallback_context,
            )
            _mark_realshop_context_reported(agent)
            agent._realshop_trace_msgs_for_act = []

        step_count += 1
        if args.max_observations is not None and step_count >= int(args.max_observations):
            return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Hermes as a RealShop agent.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--agent-id", default="agent_0")
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME"))
    parser.add_argument("--provider", default=None)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-observations", type=int, default=None)
    parser.add_argument("--max-hops-per-step", type=int, default=DEFAULT_MAX_HOPS_PER_STEP)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--observation-timeout", type=float, default=30.0)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run(args)
