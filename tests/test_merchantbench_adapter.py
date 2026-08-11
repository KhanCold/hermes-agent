import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import requests


class StubAIAgent:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.base_url = kwargs.get("base_url")
        self.model = kwargs.get("model")
        self.provider = kwargs.get("provider")
        self.api_mode = "chat_completions"
        self.request_overrides = kwargs.get("request_overrides") or {}
        self.tools = [{
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Run a shell command.",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        self.valid_tool_names = {"terminal"}

    def _execute_tool_calls(self, assistant_message, messages, effective_task_id, api_call_count=0):
        return None


stub_run_agent = ModuleType("run_agent")
stub_run_agent.AIAgent = StubAIAgent
sys.modules.setdefault("run_agent", stub_run_agent)

from merchantbench_adapter.runner import (
    MerchantBenchHermesAgent,
    MerchantBenchToolTurnComplete,
    build_arg_parser,
    run,
    _force_end_of_step,
    _flush_pending_auxiliary_usage,
    _new_trace_messages,
    _merchantbench_token_usage,
    _sanitize_merchantbench_agent_history,
    _sanitize_merchantbench_history,
    _token_usage,
    _usage_delta,
)


class FakeMerchantBenchClient:
    def __init__(self):
        self.run_id = "run-1"
        self.agent_id = "agent_0"
        self.act_calls = []
        self.usage_calls = []

    def latest_env_t(self):
        return 12

    def record_usage(self, usage, **kwargs):
        self.usage_calls.append((dict(usage), dict(kwargs)))
        return {"ok": True, "recorded": True}

    def tools(self):
        return [{
            "type": "function",
            "function": {
                "name": "search_products",
                "description": "Search the MerchantBench catalog.",
                "parameters": {"type": "object", "properties": {}},
            },
        }]

    def act(self, assistant_message=None, token_usage=None, *, messages=None, context=None):
        self.act_calls.append({
            "assistant_message": assistant_message,
            "token_usage": token_usage,
            "messages": list(messages or [assistant_message]),
            "context": context,
        })
        return {"ok": True, "tool_results": [], "step_done": False}


def test_arg_parser_defaults_to_react160k_hop_budget():
    args = build_arg_parser().parse_args([
        "--run-id", "run-1",
        "--base-url", "http://127.0.0.1:5050",
    ])

    assert args.max_hops_per_step == 30


def test_merchantbench_uses_non_streaming_with_six_api_attempts():
    agent = MerchantBenchHermesAgent(
        merchantbench_client=FakeMerchantBenchClient(),
        model="gpt-5.5-0424-global",
        base_url="https://idealab.alibaba-inc.com/api/openai/v1",
        api_key="test-key",
        provider=None,
        quiet=True,
        max_iterations=30,
    )

    assert agent._disable_streaming is True
    assert agent._api_max_retries == 6


def test_merchantbench_aligns_combined_review_to_ten_user_turn_memory_cadence():
    agent = MerchantBenchHermesAgent(
        merchantbench_client=FakeMerchantBenchClient(),
        model="gpt-5.5-0424-global",
        base_url="https://idealab.alibaba-inc.com/api/openai/v1",
        api_key="test-key",
        provider=None,
        quiet=True,
        max_iterations=30,
    )

    assert agent._memory_nudge_interval == 100
    assert agent._skill_nudge_interval == 0


def test_merchantbench_checkpoint_review_is_combined_and_synchronous():
    agent = MerchantBenchHermesAgent.__new__(MerchantBenchHermesAgent)
    agent.merchantbench_client = FakeMerchantBenchClient()
    agent._merchantbench_review_usage_lock = threading.Lock()
    agent._merchantbench_review_usage_batches = []
    review_started = threading.Event()
    release_review = threading.Event()
    review_completed = threading.Event()
    method_returned = threading.Event()
    captured = {}

    def build_review_target(
        actual_agent,
        messages_snapshot,
        *,
        review_memory,
        review_skills,
        usage_callback,
    ):
        captured.update({
            "agent": actual_agent,
            "messages_snapshot": messages_snapshot,
            "review_memory": review_memory,
            "review_skills": review_skills,
            "usage_callback": usage_callback,
        })

        def target():
            review_started.set()
            assert release_review.wait(timeout=1)
            review_completed.set()

        return target, "combined-review"

    def invoke_review():
        agent._spawn_background_review(
            messages_snapshot=[{"role": "user", "content": "checkpoint"}],
            review_memory=True,
            review_skills=False,
        )
        method_returned.set()

    with (
        patch(
            "agent.background_review.spawn_background_review_thread",
            side_effect=build_review_target,
        ),
        patch(
            "tools.thread_context.propagate_context_to_thread",
            side_effect=lambda target: target,
        ),
    ):
        caller = threading.Thread(target=invoke_review)
        caller.start()
        assert review_started.wait(timeout=1)
        assert not method_returned.wait(timeout=0.05)
        release_review.set()
        caller.join(timeout=1)

    assert not caller.is_alive()
    assert review_completed.is_set()
    assert method_returned.is_set()
    assert captured["agent"] is agent
    assert captured["messages_snapshot"] == [
        {"role": "user", "content": "checkpoint"}
    ]
    assert captured["review_memory"] is True
    assert captured["review_skills"] is True
    assert captured["usage_callback"].__self__ is agent


def test_idealab_runs_add_session_header_to_llm_requests():
    agent = MerchantBenchHermesAgent(
        merchantbench_client=FakeMerchantBenchClient(),
        model="gpt-5.5-0424-global",
        base_url="https://idealab.alibaba-inc.com/api/openai/v1",
        api_key="test-key",
        provider=None,
        quiet=True,
        max_iterations=30,
    )

    assert agent.init_kwargs["request_overrides"]["extra_headers"] == {
        "x-idealab-session-id": "run-1",
    }


def test_merchantbench_runs_enable_hermes_session_db_recording():
    class FakeSessionDB:
        pass

    fake_hermes_state = ModuleType("hermes_state")
    fake_hermes_state.SessionDB = FakeSessionDB

    with patch.dict(sys.modules, {"hermes_state": fake_hermes_state}):
        agent = MerchantBenchHermesAgent(
            merchantbench_client=FakeMerchantBenchClient(),
            model="claude-opus-4-8",
            base_url="https://idealab.alibaba-inc.com/api/openai/v1",
            api_key="test-key",
            provider=None,
            quiet=True,
            max_iterations=30,
        )

    assert agent.init_kwargs["session_id"] == "merchantbench-run-1"
    assert agent.init_kwargs["platform"] == "merchantbench"
    assert agent.init_kwargs["skip_context_files"] is True
    assert isinstance(agent.init_kwargs["session_db"], FakeSessionDB)


def test_idealab_claude_enables_official_prompt_cache_markers():
    agent = MerchantBenchHermesAgent.__new__(MerchantBenchHermesAgent)
    agent.base_url = "https://idealab.alibaba-inc.com/api/openai/v1"
    agent.model = "claude-opus-4-8"

    assert agent._anthropic_prompt_cache_policy() == (True, False)


def test_idealab_non_claude_does_not_force_prompt_cache_markers():
    agent = MerchantBenchHermesAgent.__new__(MerchantBenchHermesAgent)
    agent.base_url = "https://idealab.alibaba-inc.com/api/openai/v1"
    agent.model = "gpt-5.5-0424-global"

    assert agent._anthropic_prompt_cache_policy() == (False, False)


def test_refresh_merchantbench_tools_exposes_env_names_without_dropping_native_tools():
    agent = MerchantBenchHermesAgent.__new__(MerchantBenchHermesAgent)
    agent.merchantbench_client = FakeMerchantBenchClient()
    agent._native_tools = [{
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run a shell command.",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    agent._native_tool_names = {"terminal"}
    agent.tools = list(agent._native_tools)
    agent.valid_tool_names = {"terminal"}

    agent.refresh_merchantbench_tools()

    names = [tool["function"]["name"] for tool in agent.tools]
    assert names == ["terminal", "search_products"]
    assert agent.valid_tool_names == {
        "terminal",
        "search_products",
        "merchantbench__search_products",
    }
    assert agent._merchantbench_tool_names == {
        "search_products",
        "merchantbench__search_products",
    }
    merchantbench_tool = agent.tools[1]
    assert merchantbench_tool["x-merchantbench-tool-name"] == "search_products"
    assert merchantbench_tool["function"]["x-tool-origin"] == "merchantbench_env"
    assert merchantbench_tool["function"]["description"].startswith("[MerchantBench env]")


def test_refresh_merchantbench_tools_rejects_native_name_collisions():
    agent = MerchantBenchHermesAgent.__new__(MerchantBenchHermesAgent)
    agent.merchantbench_client = FakeMerchantBenchClient()
    agent._native_tools = [{
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "A conflicting native tool.",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    agent._native_tool_names = {"search_products"}

    with pytest.raises(
        ValueError,
        match="MerchantBench env tool names collide with Hermes native tools: search_products",
    ):
        agent.refresh_merchantbench_tools()


def test_legacy_prefixed_end_of_step_alias_dispatches_to_raw_env_name():
    class ClientWithEndOfStep(FakeMerchantBenchClient):
        def tools(self):
            return [
                *super().tools(),
                {
                    "type": "function",
                    "function": {
                        "name": "end_of_step",
                        "description": "Release the current hook.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ]

    client = ClientWithEndOfStep()
    agent = MerchantBenchHermesAgent.__new__(MerchantBenchHermesAgent)
    agent.merchantbench_client = client
    agent.provider = "openai"
    agent.api_mode = "chat_completions"
    agent._merchantbench_step_done = False
    agent._merchantbench_last_act = None
    agent._merchantbench_trace_msgs_for_act = []
    agent._native_tools = []
    agent._native_tool_names = set()
    agent.tools = []
    agent.valid_tool_names = set()
    agent.refresh_merchantbench_tools()

    assert [tool["function"]["name"] for tool in agent.tools] == [
        "search_products",
        "end_of_step",
    ]
    assert "merchantbench__end_of_step" in agent.valid_tool_names

    assistant_message = SimpleNamespace(
        content="Finish this step.",
        usage=None,
        tool_calls=[
            SimpleNamespace(
                id="call_end",
                function=SimpleNamespace(
                    name="merchantbench__end_of_step",
                    arguments="{}",
                ),
            )
        ],
    )
    messages = [{"role": "assistant", "content": "Finish this step."}]

    agent._execute_tool_calls(assistant_message, messages, "task-1", 1)

    sent_call = client.act_calls[0]["messages"][-1]["tool_calls"][0]
    assert sent_call["function"]["name"] == "end_of_step"
    assert sent_call["hermes_tool_name"] == "merchantbench__end_of_step"


def test_token_usage_uses_hermes_normalized_cache_and_reasoning_buckets():
    assistant_message = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=100,
            prompt_tokens_details=SimpleNamespace(cached_tokens=300),
            cache_creation_input_tokens=50,
            output_tokens_details=SimpleNamespace(reasoning_tokens=25),
        )
    )

    assert _token_usage(assistant_message, provider="openai", api_mode="chat_completions") == {
        "input": 650,
        "output": 100,
        "cache_read": 300,
        "cache_write": 50,
        "reasoning": 25,
        "total": 1100,
    }


def test_foreground_act_usage_excludes_auxiliary_usage():
    agent = SimpleNamespace()
    foreground = {
        "input": 100,
        "output": 20,
        "cache_read": 40,
        "cache_write": 0,
        "reasoning": 5,
        "total": 160,
    }

    assert _merchantbench_token_usage(agent, foreground) == foreground
    assert _merchantbench_token_usage(agent, None) is None


def test_checkpoint_review_usage_uses_idempotent_auxiliary_ledger():
    agent = MerchantBenchHermesAgent.__new__(MerchantBenchHermesAgent)
    agent._user_turn_count = 10
    agent._merchantbench_review_usage_lock = threading.Lock()
    agent._merchantbench_review_usage_batches = []
    client = FakeMerchantBenchClient()
    agent.merchantbench_client = client

    review_usage = {
        "input": 700,
        "output": 40,
        "cache_read": 200,
        "cache_write": 25,
        "reasoning": 10,
        "total": 975,
        "model": "cheap-review-model",
        "provider": "review-provider",
        "cost_usd": 0.123,
        "cost_status": "estimated",
        "cost_source": "official_docs_snapshot",
    }
    agent._record_background_review_usage({**review_usage, "request_index": 1})
    # Identical token counts in one review are still distinct billable calls.
    agent._record_background_review_usage({**review_usage, "request_index": 2})
    _flush_pending_auxiliary_usage(agent, client)
    _flush_pending_auxiliary_usage(agent, client)

    assert len(client.usage_calls) == 2
    usage, kwargs = client.usage_calls[0]
    assert usage["total"] == 975
    assert kwargs["source"] == "checkpoint_review"
    assert kwargs["model"] == "cheap-review-model"
    assert kwargs["cost_usd"] == 0.123
    assert kwargs["usage_id"].startswith("checkpoint-review-turn-10-call-1-")
    assert client.usage_calls[1][1]["usage_id"].startswith(
        "checkpoint-review-turn-10-call-2-"
    )
    assert client.usage_calls[0][1]["usage_id"] != (
        client.usage_calls[1][1]["usage_id"]
    )
    assert agent._merchantbench_review_usage_batches == []


def test_final_checkpoint_review_usage_flushes_without_another_act():
    class UsageClient:
        run_id = "run-1"
        agent_id = "agent_0"

        def __init__(self):
            self.calls = []
            self.current_step = 2148
            self.fail = True

        def latest_env_t(self):
            return self.current_step

        def record_usage(self, usage, **kwargs):
            self.calls.append((dict(usage), dict(kwargs)))
            if self.fail:
                self.current_step = 2160
                raise ConnectionError("response lost after server commit")
            return {"ok": True, "recorded": True}

    client = UsageClient()
    agent = MerchantBenchHermesAgent.__new__(MerchantBenchHermesAgent)
    agent.merchantbench_client = client
    agent.context_compressor = None
    agent._merchantbench_reported_summary_record_count = 0
    agent._user_turn_count = 180
    agent._merchantbench_review_usage_lock = threading.Lock()
    agent._merchantbench_review_usage_batches = []
    agent._record_background_review_usage({
        "input": 700,
        "output": 40,
        "cache_read": 200,
        "cache_write": 25,
        "reasoning": 10,
        "total": 975,
        "model": "main-model",
        "provider": "custom",
    })
    _flush_pending_auxiliary_usage(agent, client, attempts=1)
    client.fail = False
    _flush_pending_auxiliary_usage(agent, client, attempts=1)

    assert len(client.calls) == 2
    assert client.calls[0][1]["usage_id"] == client.calls[1][1]["usage_id"]
    usage, kwargs = client.calls[-1]
    assert usage["total"] == 975
    assert kwargs["source"] == "checkpoint_review"
    assert client.calls[0][1]["step"] == 2148
    assert client.calls[1][1]["step"] == 2148
    assert kwargs["usage_id"].startswith("checkpoint-review-turn-180-")


def test_compression_retry_keeps_the_original_observation_step():
    record = {
        "token_usage": {"input": 80, "output": 10, "total": 90},
        "model": "summary-model",
    }

    class UsageClient(FakeMerchantBenchClient):
        def __init__(self):
            super().__init__()
            self.current_step = 12
            self.fail = True

        def latest_env_t(self):
            return self.current_step

        def record_usage(self, usage, **kwargs):
            self.usage_calls.append((dict(usage), dict(kwargs)))
            if self.fail:
                self.current_step = 24
                raise ConnectionError("response lost")
            return {"ok": True, "recorded": True}

    agent = SimpleNamespace(
        context_compressor=SimpleNamespace(
            summary_usage_records_snapshot=lambda: [dict(record)],
        ),
        _merchantbench_reported_summary_record_count=0,
        _merchantbench_summary_usage_steps={},
        _merchantbench_review_usage_lock=threading.Lock(),
        _merchantbench_review_usage_batches=[],
    )
    client = UsageClient()

    _flush_pending_auxiliary_usage(agent, client, attempts=1)
    client.fail = False
    _flush_pending_auxiliary_usage(agent, client, attempts=1)

    assert len(client.usage_calls) == 2
    assert client.usage_calls[0][1]["step"] == 12
    assert client.usage_calls[1][1]["step"] == 12
    assert client.usage_calls[0][1]["usage_id"] == (
        client.usage_calls[1][1]["usage_id"]
    )


def test_compression_usage_id_changes_after_adapter_restart():
    record = {
        "token_usage": {"input": 80, "output": 10, "total": 90},
        "model": "summary-model",
    }

    def make_agent(session_id):
        return SimpleNamespace(
            context_compressor=SimpleNamespace(
                summary_usage_records_snapshot=lambda: [dict(record)],
            ),
            _merchantbench_reported_summary_record_count=0,
            _merchantbench_summary_usage_steps={},
            _merchantbench_usage_session_id=session_id,
            _merchantbench_review_usage_lock=threading.Lock(),
            _merchantbench_review_usage_batches=[],
        )

    first_client = FakeMerchantBenchClient()
    resumed_client = FakeMerchantBenchClient()
    _flush_pending_auxiliary_usage(
        make_agent("adapter-process-1"), first_client, attempts=1
    )
    _flush_pending_auxiliary_usage(
        make_agent("adapter-process-2"), resumed_client, attempts=1
    )

    first_kwargs = first_client.usage_calls[0][1]
    resumed_kwargs = resumed_client.usage_calls[0][1]
    assert first_kwargs["step"] == resumed_kwargs["step"] == 12
    assert first_kwargs["usage_id"] != resumed_kwargs["usage_id"]


def test_merchantbench_reports_compression_separately_from_foreground_act():
    client = FakeMerchantBenchClient()
    summary_record = {
        "token_usage": {
            "input": 80, "output": 10, "cache_read": 0,
            "cache_write": 0, "reasoning": 0, "total": 90,
        },
        "model": "summary-model",
        "provider": "summary-provider",
        "cost_usd": 0.05,
        "cost_status": "estimated",
        "cost_source": "model_catalog",
    }
    agent = MerchantBenchHermesAgent.__new__(MerchantBenchHermesAgent)
    agent.merchantbench_client = client
    agent.provider = "openai"
    agent.api_mode = "chat_completions"
    agent._merchantbench_step_done = False
    agent._merchantbench_last_act = None
    agent._merchantbench_trace_msgs_for_act = []
    agent._native_tools = []
    agent._merchantbench_tool_names = {
        "search_products",
        "merchantbench__search_products",
    }
    agent.tools = []
    agent.valid_tool_names = set()
    agent.context_compressor = SimpleNamespace(
        compression_count=0,
        last_prompt_tokens=0,
        summary_usage_records_snapshot=lambda: [dict(summary_record)],
    )
    agent._merchantbench_reported_compression_count = 0
    agent._merchantbench_reported_summary_record_count = 0
    agent._merchantbench_review_usage_lock = threading.Lock()
    agent._merchantbench_review_usage_batches = []

    def assistant(call_id):
        return SimpleNamespace(
            content="Search the catalog.",
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
            tool_calls=[SimpleNamespace(
                id=call_id,
                function=SimpleNamespace(
                    name="search_products",
                    arguments='{"query":"toy"}',
                ),
            )],
        )

    agent._execute_tool_calls(
        assistant("call_env_1"),
        [{"role": "assistant", "content": "Search the catalog."}],
        "task-1",
        1,
    )
    agent._execute_tool_calls(
        assistant("call_env_2"),
        [{"role": "assistant", "content": "Search again."}],
        "task-1",
        2,
    )

    assert len(client.usage_calls) == 1
    assert client.usage_calls[0][0]["total"] == 90
    assert client.usage_calls[0][1]["source"] == "context_compression"
    assert client.act_calls[0]["token_usage"] == {
        "input": 100,
        "output": 20,
        "cache_read": 0,
        "cache_write": 0,
        "reasoning": 0,
        "total": 120,
    }
    assert client.act_calls[1]["token_usage"] == {
        "input": 100,
        "output": 20,
        "cache_read": 0,
        "cache_write": 0,
        "reasoning": 0,
        "total": 120,
    }


def test_codex_normalized_response_usage_reaches_merchantbench_cost_buckets():
    from agent.transports.codex import ResponsesApiTransport

    raw_usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=100,
        input_tokens_details=SimpleNamespace(
            cached_tokens=300,
            cache_creation_tokens=50,
        ),
        output_tokens_details=SimpleNamespace(reasoning_tokens=25),
    )
    response = SimpleNamespace(
        output=[SimpleNamespace(
            type="function_call",
            call_id="call_env_0",
            name="merchantbench__search_products",
            arguments='{"query":"toy"}',
            id="fc_env_0",
            status="completed",
        )],
        status="completed",
        incomplete_details=None,
        usage=raw_usage,
    )

    assistant_message = ResponsesApiTransport().normalize_response(response)

    assert _token_usage(
        assistant_message,
        provider="openai",
        api_mode="codex_responses",
    ) == {
        "input": 650,
        "output": 100,
        "cache_read": 300,
        "cache_write": 50,
        "reasoning": 25,
        "total": 1100,
    }


def test_native_only_tool_call_is_sent_to_merchantbench_act_immediately():
    client = FakeMerchantBenchClient()
    agent = MerchantBenchHermesAgent.__new__(MerchantBenchHermesAgent)
    agent.merchantbench_client = client
    agent.provider = "openai"
    agent.api_mode = "chat_completions"
    agent._merchantbench_step_done = False
    agent._merchantbench_last_act = None
    agent._merchantbench_trace_msgs_for_act = []
    agent._native_tools = []
    agent._native_tool_names = set()
    agent._merchantbench_tool_names = {
        "search_products",
        "merchantbench__search_products",
    }
    agent.tools = []
    agent.valid_tool_names = set()

    assistant_message = SimpleNamespace(
        content="Inspect files.",
        usage=None,
        tool_calls=[
            SimpleNamespace(
                id="call_native_0",
                function=SimpleNamespace(name="terminal", arguments='{"command":"pwd"}'),
            )
        ],
    )
    messages = [{"role": "assistant", "content": "Inspect files."}]

    def fake_super_execute(
        _self,
        _assistant_message,
        target_messages,
        _task_id,
        _api_call_count=0,
    ):
        target_messages.append({
            "role": "tool",
            "tool_call_id": "call_native_0",
            "name": "terminal",
            "content": "/tmp/workspace",
        })

    with patch.object(
        MerchantBenchHermesAgent.__mro__[1],
        "_execute_tool_calls",
        fake_super_execute,
    ):
        agent._execute_tool_calls(assistant_message, messages, "task-1", 1)

    assert len(client.act_calls) == 1
    stored = client.act_calls[0]["messages"]
    assert [m["role"] for m in stored] == ["assistant", "tool"]
    assert [m["tool_origin"] for m in stored] == ["hermes_native", "hermes_native"]
    assert stored[1]["content"] == "/tmp/workspace"


def test_mixed_native_and_merchantbench_tool_call_reports_token_usage_once():
    client = FakeMerchantBenchClient()
    agent = MerchantBenchHermesAgent.__new__(MerchantBenchHermesAgent)
    agent.merchantbench_client = client
    agent.provider = "openai"
    agent.api_mode = "chat_completions"
    agent._merchantbench_step_done = False
    agent._merchantbench_last_act = None
    agent._merchantbench_trace_msgs_for_act = []
    agent._native_tools = []
    agent._native_tool_names = set()
    agent._merchantbench_tool_names = {
        "search_products",
        "merchantbench__search_products",
    }
    agent.tools = []
    agent.valid_tool_names = set()

    assistant_message = SimpleNamespace(
        content="Inspect files, then search products.",
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 20,
        },
        tool_calls=[
            SimpleNamespace(
                id="call_native_0",
                function=SimpleNamespace(name="terminal", arguments='{"command":"pwd"}'),
            ),
            SimpleNamespace(
                id="call_env_0",
                function=SimpleNamespace(name="search_products", arguments='{"query":"toy"}'),
            ),
        ],
    )
    messages = [{"role": "assistant", "content": "Inspect files, then search products."}]

    def fake_super_execute(
        _self,
        _assistant_message,
        target_messages,
        _task_id,
        _api_call_count=0,
    ):
        target_messages.append({
            "role": "tool",
            "tool_call_id": "call_native_0",
            "name": "terminal",
            "content": "/tmp/workspace",
        })

    with patch.object(
        MerchantBenchHermesAgent.__mro__[1],
        "_execute_tool_calls",
        fake_super_execute,
    ):
        try:
            agent._execute_tool_calls(assistant_message, messages, "task-1", 1)
        except MerchantBenchToolTurnComplete:
            pass

    assert len(client.act_calls) == 2
    native_trace_call, env_tool_call = client.act_calls
    assert native_trace_call["token_usage"] is None
    assert native_trace_call["context"] == {"tokens": 100}
    assert env_tool_call["token_usage"] == {
        "input": 100,
        "output": 20,
        "cache_read": 0,
        "cache_write": 0,
        "reasoning": 0,
        "total": 120,
    }
    assert env_tool_call["context"] == {"tokens": 100}
    assert [m["tool_origin"] for m in native_trace_call["messages"]] == [
        "hermes_native",
        "hermes_native",
    ]
    assert env_tool_call["messages"][-1]["tool_origin"] == "merchantbench_env"
    assert env_tool_call["messages"][-1]["tool_calls"] == [{
        "id": "call_env_0",
        "type": "function",
        "tool_origin": "merchantbench_env",
        "function": {"name": "search_products", "arguments": '{"query":"toy"}'},
    }]


def test_merchantbench_tool_call_preserves_assistant_message_identity():
    client = FakeMerchantBenchClient()
    agent = MerchantBenchHermesAgent.__new__(MerchantBenchHermesAgent)
    agent.merchantbench_client = client
    agent.provider = "openai"
    agent.api_mode = "chat_completions"
    agent._merchantbench_step_done = False
    agent._merchantbench_last_act = None
    agent._merchantbench_trace_msgs_for_act = []
    agent._native_tools = []
    agent.tools = []
    agent.valid_tool_names = set()

    assistant_message = SimpleNamespace(
        content="Search products.",
        usage=None,
        tool_calls=[
            SimpleNamespace(
                id="call_env_0",
                function=SimpleNamespace(
                    name="merchantbench__search_products",
                    arguments='{"query":"toy"}',
                ),
            )
        ],
    )
    persisted_assistant = {"role": "assistant", "content": "Search products."}
    messages = [persisted_assistant]

    agent._execute_tool_calls(assistant_message, messages, "task-1", 1)

    assert messages[-1] is persisted_assistant
    assert messages[-1]["tool_origin"] == "merchantbench_env"
    assert messages[-1]["tool_calls"][0]["tool_origin"] == "merchantbench_env"


def test_merchantbench_tool_call_preserves_gemini_thought_signature():
    client = FakeMerchantBenchClient()
    agent = MerchantBenchHermesAgent.__new__(MerchantBenchHermesAgent)
    agent.merchantbench_client = client
    agent.provider = ""
    agent.api_mode = "chat_completions"
    agent._merchantbench_step_done = False
    agent._merchantbench_last_act = None
    agent._merchantbench_trace_msgs_for_act = []
    agent._native_tools = []
    agent.tools = []
    agent.valid_tool_names = set()

    signature = {"google": {"thought_signature": "sig-123"}}
    assistant_message = SimpleNamespace(
        content=None,
        usage=None,
        tool_calls=[
            SimpleNamespace(
                id="call_env_0",
                extra_content=signature,
                function=SimpleNamespace(
                    name="merchantbench__search_products",
                    arguments='{"query":"toy"}',
                ),
            )
        ],
    )
    persisted_assistant = {"role": "assistant", "content": None}
    messages = [persisted_assistant]

    agent._execute_tool_calls(assistant_message, messages, "task-1", 1)

    assert persisted_assistant["tool_calls"][0]["extra_content"] == signature
    sent_tool_call = client.act_calls[0]["messages"][-1]["tool_calls"][0]
    assert sent_tool_call["extra_content"] == signature


def test_merchantbench_tool_call_preserves_direct_gemini_thought_signature():
    client = FakeMerchantBenchClient()
    agent = MerchantBenchHermesAgent.__new__(MerchantBenchHermesAgent)
    agent.merchantbench_client = client
    agent.provider = ""
    agent.api_mode = "chat_completions"
    agent._merchantbench_step_done = False
    agent._merchantbench_last_act = None
    agent._merchantbench_trace_msgs_for_act = []
    agent._native_tools = []
    agent.tools = []
    agent.valid_tool_names = set()

    assistant_message = SimpleNamespace(
        content=None,
        usage=None,
        tool_calls=[
            SimpleNamespace(
                id="call_env_direct",
                model_extra={"thoughtSignature": "sig-direct-123"},
                function=SimpleNamespace(
                    name="merchantbench__search_products",
                    arguments='{"query":"toy"}',
                ),
            )
        ],
    )
    persisted_assistant = {"role": "assistant", "content": None}
    messages = [persisted_assistant]

    agent._execute_tool_calls(assistant_message, messages, "task-1", 1)

    persisted_tool_call = persisted_assistant["tool_calls"][0]
    assert persisted_tool_call["thoughtSignature"] == "sig-direct-123"
    sent_tool_call = client.act_calls[0]["messages"][-1]["tool_calls"][0]
    assert sent_tool_call["thoughtSignature"] == "sig-direct-123"


def test_merchantbench_end_of_step_flushes_tool_results_to_session_db():
    class StepDoneClient(FakeMerchantBenchClient):
        def act(self, assistant_message=None, token_usage=None, *, messages=None, context=None):
            self.act_calls.append({
                "assistant_message": assistant_message,
                "token_usage": token_usage,
                "messages": list(messages or [assistant_message]),
                "context": context,
            })
            return {
                "ok": True,
                "tool_results": [{
                    "tool_call_id": "call_eos_0",
                    "name": "end_of_step",
                    "tool_origin": "merchantbench_env",
                    "content": '{"ok": true}',
                }],
                "step_done": True,
            }

    client = StepDoneClient()
    agent = MerchantBenchHermesAgent.__new__(MerchantBenchHermesAgent)
    agent.merchantbench_client = client
    agent.provider = "openai"
    agent.api_mode = "chat_completions"
    agent._merchantbench_step_done = False
    agent._merchantbench_last_act = None
    agent._merchantbench_trace_msgs_for_act = []
    agent._native_tools = []
    agent.tools = []
    agent.valid_tool_names = set()
    flushed = []

    def fake_flush(target_messages, conversation_history=None):
        flushed.append([dict(msg) for msg in target_messages])

    agent._flush_messages_to_session_db = fake_flush
    assistant_message = SimpleNamespace(
        content="Release the hook.",
        usage=None,
        tool_calls=[
            SimpleNamespace(
                id="call_eos_0",
                function=SimpleNamespace(name="end_of_step", arguments="{}"),
            ),
        ],
    )
    messages = [{"role": "assistant", "content": "Release the hook."}]

    with pytest.raises(MerchantBenchToolTurnComplete) as completed:
        agent._execute_tool_calls(assistant_message, messages, "task-1", 1)

    assert completed.value.final_response == "Release the hook."
    assert completed.value.reason == "tool_turn_complete(merchantbench_end_of_step)"
    assert flushed
    assert [m["role"] for m in flushed[0][-2:]] == ["assistant", "tool"]
    assert flushed[0][-1]["tool_call_id"] == "call_eos_0"
    assert flushed[0][-1]["content"] == '{"ok": true}'


def test_merchantbench_context_marks_compression_once_after_successful_act():
    client = FakeMerchantBenchClient()
    agent = MerchantBenchHermesAgent.__new__(MerchantBenchHermesAgent)
    agent.merchantbench_client = client
    agent.provider = "openai"
    agent.api_mode = "chat_completions"
    agent._merchantbench_step_done = False
    agent._merchantbench_last_act = None
    agent._merchantbench_trace_msgs_for_act = []
    agent._native_tools = []
    agent.tools = []
    agent.valid_tool_names = set()
    agent.context_compressor = SimpleNamespace(
        last_prompt_tokens=77,
        compression_count=1,
    )
    agent._merchantbench_reported_compression_count = 0

    def fake_super_execute(
        _self,
        _assistant_message,
        target_messages,
        _task_id,
        _api_call_count=0,
    ):
        target_messages.append({
            "role": "tool",
            "tool_call_id": "call_native_0",
            "name": "terminal",
            "content": "/tmp/workspace",
        })

    assistant_message = SimpleNamespace(
        content="Inspect files.",
        usage=None,
        tool_calls=[
            SimpleNamespace(
                id="call_native_0",
                function=SimpleNamespace(name="terminal", arguments='{"command":"pwd"}'),
            )
        ],
    )

    with patch.object(
        MerchantBenchHermesAgent.__mro__[1],
        "_execute_tool_calls",
        fake_super_execute,
    ):
        agent._execute_tool_calls(
            assistant_message,
            [{"role": "assistant", "content": "Inspect files."}],
            "task-1",
            1,
        )

    assert client.act_calls[0]["context"] == {
        "tokens": 77,
        "compacted": True,
    }
    assert agent._merchantbench_reported_compression_count == 1

    agent.context_compressor.last_prompt_tokens = 88
    agent._merchantbench_trace_msgs_for_act = []
    with patch.object(
        MerchantBenchHermesAgent.__mro__[1],
        "_execute_tool_calls",
        fake_super_execute,
    ):
        agent._execute_tool_calls(
            assistant_message,
            [{"role": "assistant", "content": "Inspect files again."}],
            "task-1",
            2,
        )

    assert client.act_calls[1]["context"] == {"tokens": 88}


def test_no_tool_assistant_trace_can_be_flushed_with_fallback_end_of_step():
    client = FakeMerchantBenchClient()
    prior_history = [{"role": "assistant", "content": "previous"}]
    observation_msg = {"role": "user", "content": "observation text"}
    current_history = [
        {"role": "assistant", "content": "previous"},
        observation_msg,
        {"role": "assistant", "content": "I need more data."},
    ]

    trace_messages = _new_trace_messages(
        prior_history,
        current_history,
        exclude_messages=[observation_msg],
    )
    history = list(current_history)
    _force_end_of_step(
        client,
        history,
        trace_messages,
        "[fallback] Hermes returned no tool call; releasing the hook.",
        token_usage={"input": 11, "output": 7, "cache_read": 3, "total": 21},
        context={"tokens": 55, "compacted": True},
    )

    stored = client.act_calls[0]["messages"]
    assert [m["role"] for m in stored] == ["assistant"]
    assert stored[0]["content"].startswith("I need more data.\n\n[fallback]")
    assert stored[0]["tool_origin"] == "merchantbench_env"
    assert stored[0]["tool_calls"][0]["function"]["name"] == "end_of_step"
    assert client.act_calls[0]["token_usage"] == {
        "input": 11,
        "output": 7,
        "cache_read": 3,
        "total": 21,
    }
    assert client.act_calls[0]["context"] == {
        "tokens": 55,
        "compacted": True,
    }


def test_new_trace_messages_can_exclude_observation_already_recorded_by_env():
    observation_msg = {"role": "user", "content": "observation text"}
    trace = _new_trace_messages(
        [{"role": "assistant", "content": "previous"}],
        [
            {"role": "assistant", "content": "previous"},
            observation_msg,
            {"role": "assistant", "content": "I need more data."},
        ],
        exclude_messages=[observation_msg],
    )

    assert [m["role"] for m in trace] == ["assistant"]
    assert trace[0]["content"] == "I need more data."
    assert trace[0]["tool_origin"] == "hermes_native"


def test_merchantbench_history_drops_pure_end_of_step_and_fallback_result():
    history = [
        {"role": "user", "content": "observation"},
        {
            "role": "assistant",
            "content": "[fallback] Hermes did not call end_of_step; releasing the hook.",
            "tool_origin": "merchantbench_env",
            "tool_calls": [{
                "id": "call_eos",
                "type": "function",
                "function": {"name": "end_of_step", "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call_eos",
            "name": "end_of_step",
            "content": '{"ok": true}',
        },
    ]

    assert _sanitize_merchantbench_history(history) == []


def test_merchantbench_history_keeps_business_call_but_removes_eos_and_orphans():
    history = [
        {
            "role": "assistant",
            "content": "Checked the catalog.",
            "tool_calls": [
                {
                    "id": "call_search",
                    "type": "function",
                    "function": {
                        "name": "search_products",
                        "arguments": '{"query":"sports"}',
                    },
                },
                {
                    "id": "call_eos",
                    "type": "function",
                    "function": {"name": "end_of_step", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_search",
            "name": "search_products",
            "content": '{"items": []}',
        },
        {
            "role": "tool",
            "tool_call_id": "call_eos",
            "name": "end_of_step",
            "content": '{"ok": true}',
        },
        {
            "role": "tool",
            "tool_call_id": "call_old",
            "name": "query_supply_chain_anomalies",
            "content": "stale result",
        },
    ]

    cleaned = _sanitize_merchantbench_history(history)

    assert [m["role"] for m in cleaned] == ["assistant", "tool"]
    assert [
        call["function"]["name"]
        for call in cleaned[0]["tool_calls"]
    ] == ["search_products"]
    assert cleaned[1]["tool_call_id"] == "call_search"


def test_merchantbench_history_preserves_semantic_text_from_pure_eos_turn():
    history = [{
        "role": "assistant",
        "content": "No changes needed.\n\n[fallback] releasing the hook.",
        "tool_calls": [{
            "id": "call_eos",
            "type": "function",
            "function": {"name": "merchantbench__end_of_step", "arguments": "{}"},
        }],
    }]

    assert _sanitize_merchantbench_history(history) == [{
        "role": "assistant",
        "content": "No changes needed.",
    }]


def test_merchantbench_history_reconciles_session_db_message_identities():
    retained = {"role": "system", "content": "system prompt"}
    observation = {"role": "user", "content": "observation"}
    eos = {
        "role": "assistant",
        "content": "[fallback] releasing the hook.",
        "tool_calls": [{
            "id": "call_eos",
            "type": "function",
            "function": {"name": "end_of_step", "arguments": "{}"},
        }],
    }
    eos_result = {
        "role": "tool",
        "tool_call_id": "call_eos",
        "name": "end_of_step",
        "content": '{"ok": true}',
    }
    history = [retained, observation, eos, eos_result]
    agent = SimpleNamespace(
        _flushed_db_message_ids={id(msg) for msg in history},
        _last_flushed_db_idx=len(history),
    )

    cleaned = _sanitize_merchantbench_agent_history(agent, history)

    assert cleaned == [retained]
    assert cleaned[0] is retained
    assert agent._flushed_db_message_ids == {id(retained)}
    assert agent._last_flushed_db_idx == 1


def test_run_does_not_send_observation_back_to_merchantbench_act(monkeypatch):
    class FakeLoopClient(FakeMerchantBenchClient):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.register_calls = []

        def register(self, **kwargs):
            self.register_calls.append(kwargs)

        def observation(self):
            return {
                "text": "Day 1, Hour 0\nObservation text",
                "tick": {"step": 0},
                "brief": {"system_prompt": "system prompt"},
            }

    class FakeLoopAgent:
        def __init__(self):
            self._merchantbench_step_done = False
            self._merchantbench_last_act = None
            self._merchantbench_trace_msgs_for_act = []
            self._merchantbench_reported_compression_count = 0
            self.system_messages = []
            self.context_compressor = SimpleNamespace(
                compression_count=0,
                last_prompt_tokens=0,
            )

        def refresh_merchantbench_tools(self):
            return None

        def queue_merchantbench_trace_messages(self, messages):
            self._merchantbench_trace_msgs_for_act = list(messages)

        def run_conversation(
            self,
            user_message,
            *,
            system_message=None,
            conversation_history=None,
        ):
            self.system_messages.append(system_message)
            return {
                "messages": [
                    *(conversation_history or []),
                    {"role": "assistant", "content": "Need one more check."},
                ],
            }

    created = {}

    def fake_client_factory(*args, **kwargs):
        created["client"] = FakeLoopClient()
        return created["client"]

    def fake_build_agent(args, client):
        created["agent"] = FakeLoopAgent()
        return created["agent"]

    monkeypatch.setattr(
        "merchantbench_adapter.runner.MerchantBenchToolClient",
        fake_client_factory,
    )
    monkeypatch.setattr("merchantbench_adapter.runner._build_agent", fake_build_agent)

    args = build_arg_parser().parse_args([
        "--run-id", "run-1",
        "--base-url", "http://127.0.0.1:5050",
        "--max-observations", "1",
        "--quiet",
    ])

    assert run(args) == 0

    sent_messages = created["client"].act_calls[0]["messages"]
    assert [m["role"] for m in sent_messages] == ["assistant"]
    assert sent_messages[0]["tool_calls"][0]["function"]["name"] == "end_of_step"
    assert all("Observation text" not in str(m.get("content", "")) for m in sent_messages)
    assert created["agent"].system_messages == [
        "system prompt\n\n"
        "Use any available tools, write and execute code, persist useful memory, "
        "and improve skills when helpful to maximize final net_assets."
    ]


def test_run_reobserves_when_fallback_end_of_step_is_stale(monkeypatch):
    class StaleFallbackClient(FakeMerchantBenchClient):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.observation_calls = 0

        def register(self, **kwargs):
            return {"ok": True}

        def observation(self):
            self.observation_calls += 1
            if self.observation_calls > 1:
                raise requests.HTTPError(
                    response=SimpleNamespace(status_code=410),
                )
            return {
                "text": "Day 1, Hour 0\nObservation text",
                "tick": {"step": 0},
                "brief": {"system_prompt": "system prompt"},
            }

        def act(self, assistant_message=None, token_usage=None, *, messages=None, context=None):
            self.act_calls.append({
                "assistant_message": assistant_message,
                "token_usage": token_usage,
                "messages": list(messages or [assistant_message]),
                "context": context,
            })
            raise requests.HTTPError(
                response=SimpleNamespace(status_code=425),
            )

    class NoToolAgent:
        def __init__(self):
            self._merchantbench_step_done = False
            self._merchantbench_last_act = None
            self._merchantbench_trace_msgs_for_act = []
            self._merchantbench_reported_compression_count = 0
            self.context_compressor = SimpleNamespace(
                compression_count=0,
                last_prompt_tokens=0,
            )

        def refresh_merchantbench_tools(self):
            return None

        def queue_merchantbench_trace_messages(self, messages):
            self._merchantbench_trace_msgs_for_act = list(messages)

        def run_conversation(self, user_message, **kwargs):
            return {
                "messages": [
                    *(kwargs.get("conversation_history") or []),
                    {"role": "assistant", "content": "No tool call."},
                ],
            }

    created = {}

    def fake_client_factory(*args, **kwargs):
        created["client"] = StaleFallbackClient()
        return created["client"]

    monkeypatch.setattr(
        "merchantbench_adapter.runner.MerchantBenchToolClient",
        fake_client_factory,
    )
    monkeypatch.setattr(
        "merchantbench_adapter.runner._build_agent",
        lambda args, client: NoToolAgent(),
    )

    args = build_arg_parser().parse_args([
        "--run-id", "run-1",
        "--base-url", "http://127.0.0.1:5050",
        "--quiet",
    ])

    assert run(args) == 0
    assert created["client"].observation_calls == 2
    assert len(created["client"].act_calls) == 1
    sent = created["client"].act_calls[0]["messages"]
    assert sent[-1]["tool_calls"][0]["function"]["name"] == "end_of_step"


def test_usage_delta_returns_canonical_session_token_buckets():
    agent = SimpleNamespace(
        session_input_tokens=120,
        session_output_tokens=45,
        session_cache_read_tokens=30,
        session_cache_write_tokens=5,
        session_reasoning_tokens=9,
        session_total_tokens=200,
    )

    assert _usage_delta(
        {
            "input": 100,
            "output": 40,
            "cache_read": 10,
            "cache_write": 0,
            "reasoning": 4,
            "total": 160,
        },
        agent,
    ) == {
        "input": 20,
        "output": 5,
        "cache_read": 20,
        "cache_write": 5,
        "reasoning": 5,
        "total": 40,
    }
