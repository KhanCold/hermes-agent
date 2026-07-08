import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


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

    def act(self, assistant_message=None, token_usage=None, *, messages=None):
        self.act_calls.append({
            "assistant_message": assistant_message,
            "token_usage": token_usage,
            "messages": list(messages or [assistant_message]),
        })
        return {"ok": True, "tool_results": [], "step_done": False}


def test_arg_parser_defaults_to_react160k_hop_budget():
    args = build_arg_parser().parse_args([
        "--run-id", "run-1",
        "--base-url", "http://127.0.0.1:5050",
    ])

    assert args.max_hops_per_step == 30


def test_idealab_runs_add_session_header_to_llm_requests():
    agent = RealShopHermesAgent(
        realshop_client=FakeRealShopClient(),
        model="gpt-5.5-0424-global",
        base_url="https://idealab.alibaba-inc.com/api/openai/v1",
        api_key="test-key",
        provider=None,
        quiet=True,
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
        )

    assert agent.init_kwargs["session_id"] == "realshop-run-1"
    assert agent.init_kwargs["platform"] == "realshop"
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


def test_refresh_realshop_tools_appends_prefixed_env_tools_without_dropping_native_tools():
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
    agent.tools = list(agent._native_tools)
    agent.valid_tool_names = {"terminal"}

    agent.refresh_realshop_tools()

    names = [tool["function"]["name"] for tool in agent.tools]
    assert names == ["terminal", "realshop__search_products"]
    assert agent.valid_tool_names == {"terminal", "realshop__search_products"}
    realshop_tool = agent.tools[1]
    assert realshop_tool["x-realshop-tool-name"] == "search_products"
    assert realshop_tool["function"]["x-tool-origin"] == "realshop_env"
    assert realshop_tool["function"]["description"].startswith("[RealShop env]")


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
                function=SimpleNamespace(name="realshop__search_products", arguments='{"query":"toy"}'),
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
    assert env_tool_call["token_usage"] == {
        "input": 100,
        "output": 20,
        "cache_read": 0,
        "cache_write": 0,
        "reasoning": 0,
        "total": 120,
    }
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
        "hermes_tool_name": "realshop__search_products",
    }]


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
