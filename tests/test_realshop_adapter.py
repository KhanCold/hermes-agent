import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest


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

from realshop_adapter.runner import (
    RealShopHermesAgent,
    RealShopToolTurnComplete,
    build_arg_parser,
    run,
    _force_end_of_step,
    _new_trace_messages,
    _token_usage,
    _usage_delta,
)


class FakeRealShopClient:
    def __init__(self):
        self.run_id = "run-1"
        self.act_calls = []

    def tools(self):
        return [{
            "type": "function",
            "function": {
                "name": "search_products",
                "description": "Search the RealShop catalog.",
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


def test_realshop_uses_non_streaming_with_six_api_attempts():
    agent = RealShopHermesAgent(
        realshop_client=FakeRealShopClient(),
        model="gpt-5.5-0424-global",
        base_url="https://idealab.alibaba-inc.com/api/openai/v1",
        api_key="test-key",
        provider=None,
        quiet=True,
        max_iterations=30,
    )

    assert agent._disable_streaming is True
    assert agent._api_max_retries == 6


def test_idealab_runs_add_session_header_to_llm_requests():
    agent = RealShopHermesAgent(
        realshop_client=FakeRealShopClient(),
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


def test_realshop_runs_enable_hermes_session_db_recording():
    class FakeSessionDB:
        pass

    fake_hermes_state = ModuleType("hermes_state")
    fake_hermes_state.SessionDB = FakeSessionDB

    with patch.dict(sys.modules, {"hermes_state": fake_hermes_state}):
        agent = RealShopHermesAgent(
            realshop_client=FakeRealShopClient(),
            model="claude-opus-4-8",
            base_url="https://idealab.alibaba-inc.com/api/openai/v1",
            api_key="test-key",
            provider=None,
            quiet=True,
            max_iterations=30,
        )

    assert agent.init_kwargs["session_id"] == "realshop-run-1"
    assert agent.init_kwargs["platform"] == "realshop"
    assert agent.init_kwargs["skip_context_files"] is True
    assert isinstance(agent.init_kwargs["session_db"], FakeSessionDB)


def test_idealab_claude_enables_official_prompt_cache_markers():
    agent = RealShopHermesAgent.__new__(RealShopHermesAgent)
    agent.base_url = "https://idealab.alibaba-inc.com/api/openai/v1"
    agent.model = "claude-opus-4-8"

    assert agent._anthropic_prompt_cache_policy() == (True, False)


def test_idealab_non_claude_does_not_force_prompt_cache_markers():
    agent = RealShopHermesAgent.__new__(RealShopHermesAgent)
    agent.base_url = "https://idealab.alibaba-inc.com/api/openai/v1"
    agent.model = "gpt-5.5-0424-global"

    assert agent._anthropic_prompt_cache_policy() == (False, False)


def test_refresh_realshop_tools_exposes_env_names_without_dropping_native_tools():
    agent = RealShopHermesAgent.__new__(RealShopHermesAgent)
    agent.realshop_client = FakeRealShopClient()
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

    agent.refresh_realshop_tools()

    names = [tool["function"]["name"] for tool in agent.tools]
    assert names == ["terminal", "search_products"]
    assert agent.valid_tool_names == {
        "terminal",
        "search_products",
        "realshop__search_products",
    }
    assert agent._realshop_tool_names == {
        "search_products",
        "realshop__search_products",
    }
    realshop_tool = agent.tools[1]
    assert realshop_tool["x-realshop-tool-name"] == "search_products"
    assert realshop_tool["function"]["x-tool-origin"] == "realshop_env"
    assert realshop_tool["function"]["description"].startswith("[RealShop env]")


def test_refresh_realshop_tools_rejects_native_name_collisions():
    agent = RealShopHermesAgent.__new__(RealShopHermesAgent)
    agent.realshop_client = FakeRealShopClient()
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
        match="RealShop env tool names collide with Hermes native tools: search_products",
    ):
        agent.refresh_realshop_tools()


def test_legacy_prefixed_end_of_step_alias_dispatches_to_raw_env_name():
    class ClientWithEndOfStep(FakeRealShopClient):
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
    agent = RealShopHermesAgent.__new__(RealShopHermesAgent)
    agent.realshop_client = client
    agent.provider = "openai"
    agent.api_mode = "chat_completions"
    agent._realshop_step_done = False
    agent._realshop_last_act = None
    agent._realshop_trace_msgs_for_act = []
    agent._native_tools = []
    agent._native_tool_names = set()
    agent.tools = []
    agent.valid_tool_names = set()
    agent.refresh_realshop_tools()

    assert [tool["function"]["name"] for tool in agent.tools] == [
        "search_products",
        "end_of_step",
    ]
    assert "realshop__end_of_step" in agent.valid_tool_names

    assistant_message = SimpleNamespace(
        content="Finish this step.",
        usage=None,
        tool_calls=[
            SimpleNamespace(
                id="call_end",
                function=SimpleNamespace(
                    name="realshop__end_of_step",
                    arguments="{}",
                ),
            )
        ],
    )
    messages = [{"role": "assistant", "content": "Finish this step."}]

    agent._execute_tool_calls(assistant_message, messages, "task-1", 1)

    sent_call = client.act_calls[0]["messages"][-1]["tool_calls"][0]
    assert sent_call["function"]["name"] == "end_of_step"
    assert sent_call["hermes_tool_name"] == "realshop__end_of_step"


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


def test_codex_normalized_response_usage_reaches_realshop_cost_buckets():
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
            name="realshop__search_products",
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


def test_native_only_tool_call_is_sent_to_realshop_act_immediately():
    client = FakeRealShopClient()
    agent = RealShopHermesAgent.__new__(RealShopHermesAgent)
    agent.realshop_client = client
    agent.provider = "openai"
    agent.api_mode = "chat_completions"
    agent._realshop_step_done = False
    agent._realshop_last_act = None
    agent._realshop_trace_msgs_for_act = []
    agent._native_tools = []
    agent._native_tool_names = set()
    agent._realshop_tool_names = {
        "search_products",
        "realshop__search_products",
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
        RealShopHermesAgent.__mro__[1],
        "_execute_tool_calls",
        fake_super_execute,
    ):
        agent._execute_tool_calls(assistant_message, messages, "task-1", 1)

    assert len(client.act_calls) == 1
    stored = client.act_calls[0]["messages"]
    assert [m["role"] for m in stored] == ["assistant", "tool"]
    assert [m["tool_origin"] for m in stored] == ["hermes_native", "hermes_native"]
    assert stored[1]["content"] == "/tmp/workspace"


def test_mixed_native_and_realshop_tool_call_reports_token_usage_once():
    client = FakeRealShopClient()
    agent = RealShopHermesAgent.__new__(RealShopHermesAgent)
    agent.realshop_client = client
    agent.provider = "openai"
    agent.api_mode = "chat_completions"
    agent._realshop_step_done = False
    agent._realshop_last_act = None
    agent._realshop_trace_msgs_for_act = []
    agent._native_tools = []
    agent._native_tool_names = set()
    agent._realshop_tool_names = {
        "search_products",
        "realshop__search_products",
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
        RealShopHermesAgent.__mro__[1],
        "_execute_tool_calls",
        fake_super_execute,
    ):
        try:
            agent._execute_tool_calls(assistant_message, messages, "task-1", 1)
        except RealShopToolTurnComplete:
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
    assert env_tool_call["messages"][-1]["tool_origin"] == "realshop_env"
    assert env_tool_call["messages"][-1]["tool_calls"] == [{
        "id": "call_env_0",
        "type": "function",
        "tool_origin": "realshop_env",
        "function": {"name": "search_products", "arguments": '{"query":"toy"}'},
    }]


def test_realshop_tool_call_preserves_assistant_message_identity():
    client = FakeRealShopClient()
    agent = RealShopHermesAgent.__new__(RealShopHermesAgent)
    agent.realshop_client = client
    agent.provider = "openai"
    agent.api_mode = "chat_completions"
    agent._realshop_step_done = False
    agent._realshop_last_act = None
    agent._realshop_trace_msgs_for_act = []
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
                    name="realshop__search_products",
                    arguments='{"query":"toy"}',
                ),
            )
        ],
    )
    persisted_assistant = {"role": "assistant", "content": "Search products."}
    messages = [persisted_assistant]

    agent._execute_tool_calls(assistant_message, messages, "task-1", 1)

    assert messages[-1] is persisted_assistant
    assert messages[-1]["tool_origin"] == "realshop_env"
    assert messages[-1]["tool_calls"][0]["tool_origin"] == "realshop_env"


def test_realshop_tool_call_preserves_gemini_thought_signature():
    client = FakeRealShopClient()
    agent = RealShopHermesAgent.__new__(RealShopHermesAgent)
    agent.realshop_client = client
    agent.provider = ""
    agent.api_mode = "chat_completions"
    agent._realshop_step_done = False
    agent._realshop_last_act = None
    agent._realshop_trace_msgs_for_act = []
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
                    name="realshop__search_products",
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


def test_realshop_end_of_step_flushes_tool_results_to_session_db():
    class StepDoneClient(FakeRealShopClient):
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
                    "tool_origin": "realshop_env",
                    "content": '{"ok": true}',
                }],
                "step_done": True,
            }

    client = StepDoneClient()
    agent = RealShopHermesAgent.__new__(RealShopHermesAgent)
    agent.realshop_client = client
    agent.provider = "openai"
    agent.api_mode = "chat_completions"
    agent._realshop_step_done = False
    agent._realshop_last_act = None
    agent._realshop_trace_msgs_for_act = []
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

    try:
        agent._execute_tool_calls(assistant_message, messages, "task-1", 1)
    except RealShopToolTurnComplete:
        pass

    assert flushed
    assert [m["role"] for m in flushed[0][-2:]] == ["assistant", "tool"]
    assert flushed[0][-1]["tool_call_id"] == "call_eos_0"
    assert flushed[0][-1]["content"] == '{"ok": true}'


def test_realshop_context_marks_compression_once_after_successful_act():
    client = FakeRealShopClient()
    agent = RealShopHermesAgent.__new__(RealShopHermesAgent)
    agent.realshop_client = client
    agent.provider = "openai"
    agent.api_mode = "chat_completions"
    agent._realshop_step_done = False
    agent._realshop_last_act = None
    agent._realshop_trace_msgs_for_act = []
    agent._native_tools = []
    agent.tools = []
    agent.valid_tool_names = set()
    agent.context_compressor = SimpleNamespace(
        last_prompt_tokens=77,
        compression_count=1,
    )
    agent._realshop_reported_compression_count = 0

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
        RealShopHermesAgent.__mro__[1],
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
    assert agent._realshop_reported_compression_count == 1

    agent.context_compressor.last_prompt_tokens = 88
    agent._realshop_trace_msgs_for_act = []
    with patch.object(
        RealShopHermesAgent.__mro__[1],
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
    client = FakeRealShopClient()
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
    assert [m["role"] for m in stored] == ["assistant", "assistant"]
    assert stored[0]["content"] == "I need more data."
    assert stored[0]["tool_origin"] == "hermes_native"
    assert stored[1]["tool_origin"] == "realshop_env"
    assert stored[1]["tool_calls"][0]["function"]["name"] == "end_of_step"
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


def test_run_does_not_send_observation_back_to_realshop_act(monkeypatch):
    class FakeLoopClient(FakeRealShopClient):
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
            self._realshop_step_done = False
            self._realshop_last_act = None
            self._realshop_trace_msgs_for_act = []
            self._realshop_reported_compression_count = 0
            self.context_compressor = SimpleNamespace(
                compression_count=0,
                last_prompt_tokens=0,
            )

        def refresh_realshop_tools(self):
            return None

        def queue_realshop_trace_messages(self, messages):
            self._realshop_trace_msgs_for_act = list(messages)

        def run_conversation(
            self,
            user_message,
            *,
            system_message=None,
            conversation_history=None,
        ):
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
        "realshop_adapter.runner.RealShopToolClient",
        fake_client_factory,
    )
    monkeypatch.setattr("realshop_adapter.runner._build_agent", fake_build_agent)

    args = build_arg_parser().parse_args([
        "--run-id", "run-1",
        "--base-url", "http://127.0.0.1:5050",
        "--max-observations", "1",
        "--quiet",
    ])

    assert run(args) == 0

    sent_messages = created["client"].act_calls[0]["messages"]
    assert [m["role"] for m in sent_messages] == ["assistant", "assistant"]
    assert all("Observation text" not in str(m.get("content", "")) for m in sent_messages)


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
