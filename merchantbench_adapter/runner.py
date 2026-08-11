from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
import uuid
from typing import Any, Optional
from urllib.parse import urlparse

import requests

from run_agent import AIAgent
from agent.turn_finalizer import ToolTurnComplete

from . import __version__


FRAMEWORK = "hermes"
DEFAULT_MAX_HOPS_PER_STEP = 30
MERCHANTBENCH_REVIEW_INTERVAL_USER_TURNS = 100
MERCHANTBENCH_TOOL_PREFIX = "merchantbench__"
MERCHANTBENCH_TOOL_ORIGIN = "merchantbench_env"
HERMES_TOOL_ORIGIN = "hermes_native"
IDEALAB_SESSION_HEADER = "x-idealab-session-id"
IDEALAB_HOST_SUFFIX = "idealab.alibaba-inc.com"
MERCHANTBENCH_SESSION_PREFIX = "merchantbench-"
MERCHANTBENCH_CAPABILITY_GUIDANCE = (
    "Use any available tools, write and execute code, persist useful memory, "
    "and improve skills when helpful to maximize final net_assets."
)
log = logging.getLogger(__name__)


class MerchantBenchStaleStep(BaseException):
    """Raised when MerchantBench rejects an act turn because the env advanced."""


class MerchantBenchToolTurnComplete(ToolTurnComplete):
    """Stop Hermes' inner loop after MerchantBench marks the hook complete."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        final_response: str = "",
    ) -> None:
        super().__init__(
            final_response or "MerchantBench step completed.",
            reason="tool_turn_complete(merchantbench_end_of_step)",
        )
        self.messages = messages


def _ensure_merchantbench_sdk_path() -> None:
    candidates = [
        os.environ.get("MERCHANTBENCH_AGENT_SDK_ROOT"),
        os.path.abspath(os.path.join(os.getcwd(), "..", "merchantbench", "agent")),
    ]
    for path in candidates:
        sdk_entry = os.path.join(str(path or ""), "sdk", "merchantbench_tool_client.py")
        if path and os.path.isfile(sdk_entry):
            if path not in sys.path:
                sys.path.append(path)
            return


_ensure_merchantbench_sdk_path()
from sdk.merchantbench_tool_client import MerchantBenchToolClient  # noqa: E402


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


def _merchantbench_request_overrides(base_url: Optional[str], run_id: str) -> dict[str, Any]:
    if not _is_idealab_base_url(base_url):
        return {}
    return {
        "extra_headers": {
            IDEALAB_SESSION_HEADER: str(run_id),
        },
    }


def _merchantbench_session_id(run_id: str) -> str:
    run_id = str(run_id or "").strip()
    return f"{MERCHANTBENCH_SESSION_PREFIX}{run_id or 'unknown'}"


def _merchantbench_session_db() -> Any | None:
    try:
        from hermes_state import SessionDB
    except Exception as exc:  # noqa: BLE001
        log.warning("Hermes SessionDB import failed; MerchantBench trace will still be recorded: %s", exc)
        return None
    try:
        return SessionDB()
    except Exception as exc:  # noqa: BLE001
        log.warning("Hermes SessionDB init failed; MerchantBench trace will still be recorded: %s", exc)
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
    return f"call_merchantbench_{int(time.time() * 1000)}_{index}"


def _openai_tool_call(
    tool_call: Any,
    index: int,
    *,
    merchantbench_tool_names: Optional[set[str]] = None,
) -> dict[str, Any]:
    name = _tool_call_name(tool_call)
    out = {
        "id": _tool_call_id(tool_call, index),
        "type": "function",
        "tool_origin": _tool_origin_for_name(name, merchantbench_tool_names),
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
    direct_signature = getattr(tool_call, "thoughtSignature", None)
    model_extra = getattr(tool_call, "model_extra", None)
    if direct_signature is None and isinstance(tool_call, dict):
        direct_signature = tool_call.get("thoughtSignature")
    if direct_signature is None and isinstance(model_extra, dict):
        direct_signature = model_extra.get("thoughtSignature")
    if isinstance(direct_signature, str) and direct_signature:
        out["thoughtSignature"] = direct_signature
    return out


def _end_of_step_tool_call() -> dict[str, Any]:
    return {
        "id": f"call_merchantbench_end_{int(time.time() * 1000)}",
        "type": "function",
        "tool_origin": MERCHANTBENCH_TOOL_ORIGIN,
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
    merchantbench_tool_names: Optional[set[str]] = None,
) -> str:
    return (
        MERCHANTBENCH_TOOL_ORIGIN
        if _is_merchantbench_tool_name(name, merchantbench_tool_names)
        else HERMES_TOOL_ORIGIN
    )


def _is_merchantbench_tool_name(
    name: str,
    merchantbench_tool_names: Optional[set[str]] = None,
) -> bool:
    name = str(name or "")
    return (
        name == "end_of_step"
        or name.startswith(MERCHANTBENCH_TOOL_PREFIX)
        or name in (merchantbench_tool_names or set())
    )


def _merchantbench_env_tool_name(name: str) -> str:
    name = str(name or "")
    if name.startswith(MERCHANTBENCH_TOOL_PREFIX):
        return name[len(MERCHANTBENCH_TOOL_PREFIX):]
    return name


def _exposed_merchantbench_tool(tool: dict[str, Any]) -> dict[str, Any]:
    out = dict(tool)
    fn = dict(out.get("function") or {})
    env_name = str(fn.get("name") or "")
    fn["name"] = env_name
    fn["description"] = f"[MerchantBench env] {fn.get('description') or ''}".strip()
    fn["x-tool-origin"] = MERCHANTBENCH_TOOL_ORIGIN
    fn["x-merchantbench-tool-name"] = env_name
    out["function"] = fn
    out["tool_origin"] = MERCHANTBENCH_TOOL_ORIGIN
    out["x-merchantbench-tool-name"] = env_name
    return out


def _merchantbench_act_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    out = dict(tool_call)
    fn = dict(out.get("function") or {})
    hermes_name = str(fn.get("name") or "")
    env_name = _merchantbench_env_tool_name(hermes_name)
    fn["name"] = env_name
    out["function"] = fn
    out["tool_origin"] = MERCHANTBENCH_TOOL_ORIGIN
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


def _merchantbench_context(
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
        getattr(agent, "_merchantbench_reported_compression_count", 0) or 0
    )
    context: dict[str, Any] = {}
    if prompt_tokens > 0:
        context["tokens"] = prompt_tokens
    if current_count > reported_count:
        context["compacted"] = True
    return context or None


def _mark_merchantbench_context_reported(agent: Any) -> None:
    current_count = _compressor_count(agent)
    reported_count = int(
        getattr(agent, "_merchantbench_reported_compression_count", 0) or 0
    )
    if current_count > reported_count:
        agent._merchantbench_reported_compression_count = current_count


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


_TOKEN_USAGE_KEYS = (
    "input",
    "output",
    "cache_read",
    "cache_write",
    "reasoning",
    "total",
)


def _canonical_token_usage(
    usage: Optional[dict[str, int]],
) -> Optional[dict[str, int]]:
    if not usage:
        return None
    normalized = {
        key: max(0, int(usage.get(key, 0) or 0))
        for key in _TOKEN_USAGE_KEYS
    }
    if "total" not in usage:
        normalized["total"] = sum(
            normalized[key]
            for key in ("input", "output", "cache_read", "cache_write")
        )
    return normalized if any(normalized.values()) else None


def _summary_usage_snapshot(agent: Any) -> dict[str, int]:
    compressor = getattr(agent, "context_compressor", None)
    snapshotter = getattr(compressor, "summary_usage_snapshot", None)
    if not callable(snapshotter):
        return {key: 0 for key in _TOKEN_USAGE_KEYS}
    try:
        snapshot = snapshotter() or {}
        return {
            key: max(0, int(snapshot.get(key, 0) or 0))
            for key in _TOKEN_USAGE_KEYS
        }
    except (AttributeError, TypeError, ValueError):
        return {key: 0 for key in _TOKEN_USAGE_KEYS}


def _summary_usage_records_snapshot(agent: Any) -> list[dict[str, Any]]:
    compressor = getattr(agent, "context_compressor", None)
    snapshotter = getattr(compressor, "summary_usage_records_snapshot", None)
    if not callable(snapshotter):
        return []
    try:
        return [dict(record) for record in (snapshotter() or [])]
    except (AttributeError, TypeError, ValueError):
        return []


def _merchantbench_token_usage(
    agent: Any,
    foreground_usage: Optional[dict[str, int]],
) -> Optional[dict[str, int]]:
    """Return foreground-only usage; auxiliary work has its own ledger."""
    return _canonical_token_usage(foreground_usage)


def _post_auxiliary_usage(
    client: MerchantBenchToolClient,
    batch: dict[str, Any],
    *,
    attempts: int,
) -> bool:
    last_error: Optional[Exception] = None
    for attempt in range(max(1, attempts)):
        try:
            client.record_usage(
                batch["token_usage"],
                usage_id=batch["usage_id"],
                source=batch["source"],
                step=batch.get("step"),
                model=batch.get("model"),
                provider=batch.get("provider"),
                cost_usd=batch.get("cost_usd"),
                cost_status=batch.get("cost_status"),
                cost_source=batch.get("cost_source"),
            )
            return True
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max(1, attempts):
                time.sleep(2 ** attempt)
    if last_error is not None:
        log.warning(
            "Could not record auxiliary usage %s: %s",
            batch.get("usage_id"),
            last_error,
        )
    return False


def _flush_pending_summary_usage(
    agent: Any,
    client: MerchantBenchToolClient,
    *,
    attempts: int,
) -> None:
    records = _summary_usage_records_snapshot(agent)
    reported = int(
        getattr(agent, "_merchantbench_reported_summary_record_count", 0) or 0
    )
    reported_steps = getattr(agent, "_merchantbench_summary_usage_steps", None)
    if not isinstance(reported_steps, dict):
        reported_steps = {}
        agent._merchantbench_summary_usage_steps = reported_steps
    usage_session_id = str(
        getattr(agent, "_merchantbench_usage_session_id", "") or ""
    )
    if not usage_session_id:
        usage_session_id = uuid.uuid4().hex
        agent._merchantbench_usage_session_id = usage_session_id
    while reported < len(records):
        record = records[reported]
        usage = _canonical_token_usage(record.get("token_usage"))
        if usage is None:
            reported += 1
            agent._merchantbench_reported_summary_record_count = reported
            continue
        usage_step = reported_steps.setdefault(
            reported,
            client.latest_env_t(),
        )
        digest = hashlib.sha256(
            json.dumps(
                {
                    "session": usage_session_id,
                    "index": reported,
                    "step": usage_step,
                    "record": record,
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:20]
        batch = {
            **record,
            "usage_id": f"compression-{reported}-{digest}",
            "source": "context_compression",
            "token_usage": usage,
            "step": usage_step,
        }
        if not _post_auxiliary_usage(client, batch, attempts=attempts):
            return
        reported += 1
        agent._merchantbench_reported_summary_record_count = reported
        reported_steps.pop(reported - 1, None)


def _flush_pending_review_usage(
    agent: Any,
    client: MerchantBenchToolClient,
    *,
    attempts: int,
) -> None:
    lock = getattr(agent, "_merchantbench_review_usage_lock", None)
    if lock is None:
        return
    with lock:
        batches = [dict(batch) for batch in agent._merchantbench_review_usage_batches]
    for batch in batches:
        if not _post_auxiliary_usage(client, batch, attempts=attempts):
            return
        with lock:
            agent._merchantbench_review_usage_batches = [
                queued
                for queued in agent._merchantbench_review_usage_batches
                if queued.get("usage_id") != batch.get("usage_id")
            ]


def _flush_pending_auxiliary_usage(
    agent: Any,
    client: MerchantBenchToolClient,
    *,
    attempts: int = 3,
) -> None:
    """Persist all unreported auxiliary usage through idempotent entries."""
    _flush_pending_summary_usage(agent, client, attempts=attempts)
    _flush_pending_review_usage(agent, client, attempts=attempts)


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


def _sanitize_merchantbench_history(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove MerchantBench hook-control messages from the next LLM prompt.

    ``end_of_step`` is a transport-level acknowledgement that must remain in
    the MerchantBench trace, but repeatedly replaying it to the model can trigger
    provider loop detection.  Also discard orphaned tool results left behind
    by compaction so the retained OpenAI message sequence stays valid.
    """
    eos_call_ids: set[str] = set()
    sanitized: list[dict[str, Any]] = []

    for raw in history:
        if not isinstance(raw, dict):
            continue
        msg = raw
        if msg.get("role") != "assistant":
            sanitized.append(msg)
            continue

        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            sanitized.append(msg)
            continue

        kept_calls: list[dict[str, Any]] = []
        removed_eos = False
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") or {}
            if _merchantbench_env_tool_name(fn.get("name")) == "end_of_step":
                removed_eos = True
                call_id = call.get("id")
                if call_id:
                    eos_call_ids.add(str(call_id))
                continue
            kept_calls.append(call)

        if not removed_eos:
            sanitized.append(msg)
            continue
        msg = dict(raw)
        if kept_calls:
            msg["tool_calls"] = kept_calls
            sanitized.append(msg)
            continue

        msg.pop("tool_calls", None)
        msg.pop("tool_origin", None)
        content = msg.get("content")
        drop_with_observation = False
        if isinstance(content, str):
            for marker in ("\n\n[fallback]", "\n\n[llm-error]"):
                marker_idx = content.find(marker)
                if marker_idx >= 0:
                    content = content[:marker_idx].rstrip()
            if content.startswith(("[fallback]", "[llm-error]")):
                content = ""
            msg["content"] = content
        if not content and not msg.get("reasoning_content"):
            # A synthetic/empty acknowledgement contains no model decision.
            # Drop its observation too so the next full observation does not
            # create adjacent user roles.
            drop_with_observation = True
            msg["_merchantbench_drop_with_observation"] = True
        if content or msg.get("reasoning_content"):
            sanitized.append(msg)
        elif drop_with_observation:
            sanitized.append(msg)

    valid_tool_call_ids = {
        str(call.get("id"))
        for msg in sanitized
        if msg.get("role") == "assistant"
        for call in (msg.get("tool_calls") or [])
        if isinstance(call, dict) and call.get("id")
    }
    filtered = [
        msg
        for msg in sanitized
        if msg.get("role") != "tool"
        or (
            str(msg.get("tool_call_id") or "") not in eos_call_ids
            and str(msg.get("tool_call_id") or "") in valid_tool_call_ids
        )
    ]
    repaired: list[dict[str, Any]] = []
    for msg in filtered:
        if msg.pop("_merchantbench_drop_with_observation", False):
            if repaired and repaired[-1].get("role") == "user":
                repaired.pop()
            continue
        repaired.append(msg)
    return repaired


def _sanitize_merchantbench_agent_history(
    agent: Any,
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sanitize provider history and reconcile SessionDB identity tracking."""
    sanitized = _sanitize_merchantbench_history(history)
    flushed_ids = getattr(agent, "_flushed_db_message_ids", None)
    if isinstance(flushed_ids, set):
        flushed_ids.intersection_update(
            id(msg) for msg in sanitized if isinstance(msg, dict)
        )
    flush_cursor = getattr(agent, "_last_flushed_db_idx", None)
    if isinstance(flush_cursor, int):
        agent._last_flushed_db_idx = min(flush_cursor, len(sanitized))
    return sanitized


def _reorder_end_of_step_last(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normal_calls = [
        tc for tc in tool_calls
        if _merchantbench_env_tool_name((tc.get("function") or {}).get("name")) != "end_of_step"
    ]
    end_calls = [
        tc for tc in tool_calls
        if _merchantbench_env_tool_name((tc.get("function") or {}).get("name")) == "end_of_step"
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
            "tool_origin": result.get("tool_origin", MERCHANTBENCH_TOOL_ORIGIN),
            "content": result["content"],
        })


class MerchantBenchHermesAgent(AIAgent):
    def __init__(
        self,
        *,
        merchantbench_client: MerchantBenchToolClient,
        model: str,
        base_url: Optional[str],
        api_key: Optional[str],
        provider: Optional[str],
        quiet: bool,
        max_iterations: int,
    ) -> None:
        run_id = str(getattr(merchantbench_client, "run_id", "") or "")
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            provider=provider,
            model=model,
            max_iterations=max_iterations,
            quiet_mode=quiet,
            save_trajectories=False,
            skip_context_files=True,
            session_id=_merchantbench_session_id(run_id),
            session_db=_merchantbench_session_db(),
            platform="merchantbench",
            request_overrides=_merchantbench_request_overrides(
                base_url,
                run_id,
            ),
        )
        # MerchantBench runs headlessly, so streaming adds an SSE failure surface
        # without any user-visible benefit. Retry transient API failures using
        # the normal non-streaming path instead.
        self._disable_streaming = True
        self._api_max_retries = 6
        # MerchantBench uses one observation/hook as one Hermes user turn.  Align
        # both self-improvement surfaces to the memory cadence instead of
        # letting skill review fire on inner tool-loop iterations.
        self._memory_nudge_interval = MERCHANTBENCH_REVIEW_INTERVAL_USER_TURNS
        self._skill_nudge_interval = 0
        self.merchantbench_client = merchantbench_client
        self._native_tools = list(self.tools or [])
        self._native_tool_names = {
            str((tool.get("function") or {}).get("name"))
            for tool in self._native_tools
            if (tool.get("function") or {}).get("name")
        }
        self._merchantbench_step_done = False
        self._merchantbench_last_act: Optional[dict[str, Any]] = None
        self._merchantbench_trace_msgs_for_act: list[dict[str, Any]] = []
        self._merchantbench_reported_compression_count = _compressor_count(self)
        self._merchantbench_reported_summary_record_count = len(
            _summary_usage_records_snapshot(self)
        )
        # Stable for this adapter process so retries reuse an idempotency key,
        # but distinct after a process restart so a resumed run cannot collide
        # with an earlier compression record that had the same index/payload.
        self._merchantbench_usage_session_id = uuid.uuid4().hex
        self._merchantbench_summary_usage_steps: dict[int, Optional[int]] = {}
        self._merchantbench_review_usage_lock = threading.Lock()
        self._merchantbench_review_usage_batches: list[dict[str, Any]] = []
        self.refresh_merchantbench_tools()

    def _record_background_review_usage(self, usage: dict[str, Any]) -> None:
        normalized = _canonical_token_usage(usage)
        if normalized is None:
            return
        trigger_step = self.merchantbench_client.latest_env_t()
        identity = {
            "turn": int(getattr(self, "_user_turn_count", 0) or 0),
            "request_index": int(usage.get("request_index", 0) or 0),
            "step": trigger_step,
            "usage": normalized,
            "model": str(usage.get("model") or ""),
            "provider": str(usage.get("provider") or ""),
            "cost_usd": usage.get("cost_usd"),
            "cost_status": str(usage.get("cost_status") or "unknown"),
            "cost_source": str(usage.get("cost_source") or "none"),
        }
        digest = hashlib.sha256(
            json.dumps(
                identity,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:20]
        batch = {
            "usage_id": (
                f"checkpoint-review-turn-{identity['turn']}-"
                f"call-{identity['request_index']}-{digest}"
            ),
            "source": "checkpoint_review",
            "token_usage": normalized,
            "step": trigger_step,
            "model": identity["model"],
            "provider": identity["provider"],
            "cost_usd": identity["cost_usd"],
            "cost_status": identity["cost_status"],
            "cost_source": identity["cost_source"],
        }
        with self._merchantbench_review_usage_lock:
            if not any(
                queued.get("usage_id") == batch["usage_id"]
                for queued in self._merchantbench_review_usage_batches
            ):
                self._merchantbench_review_usage_batches.append(batch)

    def _spawn_background_review(
        self,
        messages_snapshot: list[dict[str, Any]],
        review_memory: bool = False,
        review_skills: bool = False,
    ) -> None:
        """Run each MerchantBench checkpoint review to completion before returning.

        The normal Hermes review remains asynchronous.  MerchantBench instead uses
        the ten-user-turn memory trigger as a combined memory + skills
        checkpoint, and waits for that review before requesting the next
        observation.  The review still runs in its own thread so Hermes'
        thread-local tool and approval isolation is preserved.
        """
        from agent.background_review import spawn_background_review_thread
        from tools.thread_context import propagate_context_to_thread

        target, _prompt = spawn_background_review_thread(
            self,
            messages_snapshot,
            review_memory=True,
            review_skills=True,
            usage_callback=self._record_background_review_usage,
        )
        review_thread = threading.Thread(
            target=propagate_context_to_thread(target),
            daemon=True,
            name="merchantbench-checkpoint-review",
        )
        review_thread.start()
        review_thread.join()
        _flush_pending_review_usage(
            self,
            self.merchantbench_client,
            attempts=3,
        )

    def refresh_merchantbench_tools(self) -> None:
        merchantbench_tools = [
            _exposed_merchantbench_tool(tool)
            for tool in self.merchantbench_client.tools()
        ]
        raw_merchantbench_names = {
            str((tool.get("function") or {}).get("name"))
            for tool in merchantbench_tools
            if (tool.get("function") or {}).get("name")
        }
        collisions = raw_merchantbench_names & self._native_tool_names
        if collisions:
            names = ", ".join(sorted(collisions))
            raise ValueError(
                "MerchantBench env tool names collide with Hermes native tools: "
                f"{names}"
            )
        legacy_prefixed_aliases = {
            f"{MERCHANTBENCH_TOOL_PREFIX}{name}"
            for name in raw_merchantbench_names
        }
        self._merchantbench_tool_names = {
            *raw_merchantbench_names,
            *legacy_prefixed_aliases,
        }
        tools = [*self._native_tools, *merchantbench_tools]
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

    def queue_merchantbench_trace_messages(self, messages: list[dict[str, Any]]) -> None:
        self._merchantbench_trace_msgs_for_act = list(messages)

    def _execute_tool_calls(
        self,
        assistant_message: Any,
        messages: list[dict[str, Any]],
        effective_task_id: str,
        api_call_count: int = 0,
    ) -> None:
        # Compression/review calls are independent billable requests. Flush
        # them through the idempotent cost endpoint before recording this
        # foreground turn so a lost /act response can never duplicate them.
        _flush_pending_auxiliary_usage(
            self,
            self.merchantbench_client,
            attempts=1,
        )
        raw_tool_calls = getattr(assistant_message, "tool_calls", None) or []
        merchantbench_tool_names = getattr(self, "_merchantbench_tool_names", set())
        tool_calls = _reorder_end_of_step_last([
            _openai_tool_call(
                tool_call,
                index,
                merchantbench_tool_names=merchantbench_tool_names,
            )
            for index, tool_call in enumerate(raw_tool_calls)
        ])
        merchantbench_tool_calls = [
            tc for tc in tool_calls
            if _is_merchantbench_tool_name(
                (tc.get("function") or {}).get("name", ""),
                merchantbench_tool_names,
            )
        ]
        native_raw_tool_calls = [
            tool_call for tool_call in raw_tool_calls
            if not _is_merchantbench_tool_name(
                _tool_call_name(tool_call),
                merchantbench_tool_names,
            )
        ]
        native_tool_calls = [
            tc for tc in tool_calls
            if not _is_merchantbench_tool_name(
                (tc.get("function") or {}).get("name", ""),
                merchantbench_tool_names,
            )
        ]
        assistant_msg = {
            "role": "assistant",
            "content": _assistant_content(assistant_message),
            "tool_calls": tool_calls,
            "tool_origin": (
                MERCHANTBENCH_TOOL_ORIGIN
                if merchantbench_tool_calls and not native_tool_calls
                else HERMES_TOOL_ORIGIN
                if native_tool_calls and not merchantbench_tool_calls
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
                *self._merchantbench_trace_msgs_for_act,
                native_assistant_msg,
                *native_results,
            ]
            try:
                native_token_usage = None
                if not merchantbench_tool_calls:
                    native_token_usage = _merchantbench_token_usage(
                        self,
                        _token_usage(
                            assistant_message,
                            provider=getattr(self, "provider", None),
                            api_mode=getattr(self, "api_mode", None),
                        ),
                    )
                native_context = _merchantbench_context(self, assistant_message)
                self.merchantbench_client.act(
                    token_usage=native_token_usage,
                    messages=trace_messages,
                    context=native_context,
                )
                _mark_merchantbench_context_reported(self)
            except requests.HTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status == 425:
                    self._merchantbench_trace_msgs_for_act = []
                    raise MerchantBenchStaleStep() from exc
                raise
            self._merchantbench_trace_msgs_for_act = []

        if not merchantbench_tool_calls:
            self._merchantbench_last_act = {"ok": True, "step_done": False, "trace_only": True}
            return

        merchantbench_act_calls = [_merchantbench_act_tool_call(tc) for tc in merchantbench_tool_calls]
        merchantbench_assistant_msg = dict(assistant_msg)
        merchantbench_assistant_msg["tool_calls"] = merchantbench_act_calls
        merchantbench_assistant_msg["tool_origin"] = MERCHANTBENCH_TOOL_ORIGIN
        name_by_id = {
            tc["id"]: (tc.get("function") or {}).get("name", "")
            for tc in merchantbench_tool_calls
        }
        trace_msgs = list(self._merchantbench_trace_msgs_for_act)
        try:
            context = _merchantbench_context(self, assistant_message)
            act_resp = self.merchantbench_client.act(
                token_usage=_merchantbench_token_usage(
                    self,
                    _token_usage(
                        assistant_message,
                        provider=getattr(self, "provider", None),
                        api_mode=getattr(self, "api_mode", None),
                    ),
                ),
                messages=[*trace_msgs, merchantbench_assistant_msg],
                context=context,
            )
            _mark_merchantbench_context_reported(self)
        except requests.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 425:
                self._merchantbench_trace_msgs_for_act = []
                raise MerchantBenchStaleStep() from exc
            raise

        self._merchantbench_trace_msgs_for_act = []
        self._merchantbench_last_act = act_resp
        self._merchantbench_step_done = bool(act_resp.get("step_done"))
        _append_tool_results(messages, act_resp, name_by_tool_call_id=name_by_id)
        if self._merchantbench_step_done:
            try:
                self._flush_messages_to_session_db(messages)
            except Exception as exc:
                log.warning("Hermes SessionDB flush failed after MerchantBench end_of_step: %s", exc)
            raise MerchantBenchToolTurnComplete(
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
    client: MerchantBenchToolClient,
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
        assistant_msg["tool_origin"] = MERCHANTBENCH_TOOL_ORIGIN
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
            "tool_origin": MERCHANTBENCH_TOOL_ORIGIN,
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


def _build_agent(args: argparse.Namespace, client: MerchantBenchToolClient) -> MerchantBenchHermesAgent:
    model = args.model or os.environ.get("MODEL_NAME", "")
    return MerchantBenchHermesAgent(
        merchantbench_client=client,
        model=model,
        base_url=args.openai_base_url or os.environ.get("OPENAI_BASE_URL"),
        api_key=args.openai_api_key or os.environ.get("OPENAI_API_KEY"),
        provider=args.provider or os.environ.get("HERMES_PROVIDER"),
        quiet=args.quiet,
        max_iterations=max(1, int(args.max_hops_per_step)),
    )


def run(args: argparse.Namespace) -> int:
    client = MerchantBenchToolClient(
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
            "adapter": "merchantbench_adapter",
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
                _flush_pending_auxiliary_usage(agent, client)
                if not args.quiet:
                    print("[merchantbench-adapter] run finished", file=sys.stderr)
                return 0
            if not args.quiet:
                print(f"[merchantbench-adapter] observation error: {exc}", file=sys.stderr)
            time.sleep(args.retry_sleep)
            continue

        if args.max_steps is not None and _tick_step(obs) >= int(args.max_steps):
            _flush_pending_auxiliary_usage(agent, client)
            if not args.quiet:
                print(f"[merchantbench-adapter] reached max_steps={args.max_steps}", file=sys.stderr)
            return 0

        brief = obs.get("brief") or {}
        if system_prompt is None and brief.get("system_prompt"):
            system_prompt = (
                f"{str(brief['system_prompt']).rstrip()}\n\n"
                f"{MERCHANTBENCH_CAPABILITY_GUIDANCE}"
            )

        observation_msg = {"role": "user", "content": obs.get("text", "") or ""}
        history = _sanitize_merchantbench_agent_history(agent, history)
        history_len_before_step = len(history)
        agent._merchantbench_step_done = False
        agent._merchantbench_last_act = None

        stale_step = False
        no_tool_token_usage = None
        agent.refresh_merchantbench_tools()
        agent.queue_merchantbench_trace_messages([])
        history_before_step = list(history)
        usage_before_step = _usage_snapshot(agent)
        try:
            result = agent.run_conversation(
                observation_msg["content"],
                system_message=system_prompt,
                conversation_history=history,
            )
        except MerchantBenchStaleStep:
            history = history[:history_len_before_step]
            stale_step = True
        else:
            history = list(result.get("messages") or history)

        if stale_step:
            continue

        if not agent._merchantbench_step_done:
            if agent._merchantbench_last_act is None:
                agent._merchantbench_trace_msgs_for_act = [
                    *agent._merchantbench_trace_msgs_for_act,
                    *_new_trace_messages(
                        history_before_step,
                        history,
                        exclude_messages=[observation_msg],
                    ),
                ]
                no_tool_token_usage = _usage_delta(usage_before_step, agent)
            fallback_context = _merchantbench_context(agent)
            _flush_pending_auxiliary_usage(
                agent,
                client,
                attempts=3,
            )
            try:
                history = _force_end_of_step(
                    client,
                    history,
                    agent._merchantbench_trace_msgs_for_act,
                    "[fallback] Hermes did not call end_of_step; releasing the hook.",
                    token_usage=no_tool_token_usage,
                    context=fallback_context,
                )
            except requests.HTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status != 425:
                    raise
                # The hook advanced while the synthetic acknowledgement was
                # in flight. Discard this entire stale turn and obtain a new
                # observation; replaying the acknowledgement would attach it
                # to the wrong simulation step.
                history = history[:history_len_before_step]
                agent._merchantbench_trace_msgs_for_act = []
                continue
            _mark_merchantbench_context_reported(agent)
            agent._merchantbench_trace_msgs_for_act = []

        # MerchantBench's hook acknowledgement belongs in the environment trace,
        # not in the next provider request.  Sanitize only after the current
        # hook has been fully recorded.
        history = _sanitize_merchantbench_agent_history(agent, history)

        step_count += 1
        if args.max_observations is not None and step_count >= int(args.max_observations):
            _flush_pending_auxiliary_usage(agent, client)
            return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Hermes as a MerchantBench agent.")
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
